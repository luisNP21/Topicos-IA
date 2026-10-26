"""Entry point de la tarea 5a: metricas de transcripcion y auditoria de entidades.

Uso:
    python run_metricas.py                                   # entradas de config.yaml
    python run_metricas.py --transcripciones RUTA [RUTA ...] [--ner RUTA ...] [--nivel N] [--nombre S]
    python run_metricas.py --simular                         # prueba con datos simulados

Por cada (sistema, nivel):
- une las paginas de cada documento (ver paginas.py) y guarda ese cache por documento,
- CER y WER contra el texto visible y contra el texto completo,
- lectura de las entidades del gold que se ven en la imagen,
- si hay cache NER, el origen de cada entidad predicha (acierto, fp_ner, texto_extra,
  mal_leida, inventada) y la lista de candidatas para el juez visual.

Salidas en <PROJECT_ROOT>/<rutas.salida_dir>/ (o .../simulado/ con --simular).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for extra in (REPO_ROOT / "M2", REPO_ROOT / "M2" / "harness", SCRIPT_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import log  # noqa: E402  (M2/common.py)
from gold_loader import cargar_gold_set  # noqa: E402  (M2/harness/gold_loader.py)

from auditoria import (  # noqa: E402
    CATEGORIAS, clasificar_predichas, lectura_entidades, normalizar_entidad, normalizar_texto,
)
from cer_wer import comparar, sumar, tasas  # noqa: E402
from paginas import cargar_json, unir_paginas  # noqa: E402


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        log(f"PROJECT_ROOT no definido; se usa la raiz del repo: {REPO_ROOT}")
        return REPO_ROOT
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def _ruta(project_root: Path, valor: str) -> Path:
    p = Path(valor).expanduser()
    return p if p.is_absolute() else project_root / p


def cargar_eval_set_m3(ruta: Path) -> list[dict]:
    """Carga M3/ejecucion/eval_set_casos.json con el mismo esquema que el gold de M2 (caso_01..caso_15)."""
    jsonl_candidato = ruta.parent.parent / "M4/eval_set/eval_set_casos.jsonl"
    if jsonl_candidato.exists():
        with open(jsonl_candidato, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if not ruta.exists():
        raise FileNotFoundError(f"Eval set de M3 no encontrado: {ruta}")
    casos = cargar_json(ruta)
    if not isinstance(casos, list) or not casos:
        raise ValueError(f"El eval set de M3 esta vacio o no es una lista: {ruta}")
    registros = []
    for i, c in enumerate(casos):
        texto = (c.get("input") or "").strip()
        entidad = (c.get("entidad") or "").strip()
        entities = c.get("entities_gold") or ([entidad] if entidad else [])
        if isinstance(entities, str):
            entities = [entities]
        registros.append({
            "doc_id": str(c.get("id") or f"caso_{i + 1:02d}"),
            "text": texto,
            "entities_gold": entities,
            "criterio": "",
        })
    return registros


def _cargar_gold(manifest: dict, rutas: dict) -> dict[str, dict]:
    if manifest.get("fuente") == "m3":
        rel = manifest.get("gold_set") or rutas.get("m3_eval_set", "M3/ejecucion/eval_set_casos.json")
        return {r["doc_id"]: r for r in cargar_eval_set_m3(REPO_ROOT / rel)}
    return {r["doc_id"]: r for r in cargar_gold_set(REPO_ROOT / rutas["gold_set"])}


def _ner_por_documento(cache: dict) -> tuple[dict, bool]:
    """Acepta NER por documento o por pagina (<doc_id>__p<k>); en ese caso une por documento."""
    por_pagina = any("__p" in k for k in cache)
    if not por_pagina:
        return {k: list(v) for k, v in cache.items()}, False
    unido = defaultdict(set)
    for k, v in cache.items():
        unido[k.split("__p")[0]].update(v)
    return {k: sorted(v) for k, v in unido.items()}, True


# --------------------------------------------------------------------------- #
# Evaluacion de una entrada
# --------------------------------------------------------------------------- #

def evaluar_entrada(entrada: dict, manifest: dict, docs_manifest: dict, gold: dict, ner_texto: dict | None,
                    cfg: dict, project_root: Path, salida: Path) -> dict:
    ruta_t = _ruta(project_root, entrada["transcripciones"])
    nivel, unido, avisos = unir_paginas(cargar_json(ruta_t), docs_manifest, entrada.get("nivel"))
    if nivel not in manifest["niveles"]:
        raise ValueError(f"El nivel {nivel!r} no esta en el manifiesto")
    resumen_nivel = manifest["niveles"][nivel]
    backend = next((d["backend"] for d in unido.values() if d.get("backend")), "desconocido")
    nombre = entrada.get("nombre") or backend

    ner, ner_por_pagina = None, False
    if entrada.get("ner"):
        ruta_ner = _ruta(project_root, entrada["ner"])
        if ruta_ner.exists():
            ner, ner_por_pagina = _ner_por_documento(cargar_json(ruta_ner))
        else:
            log(f"Aviso: cache NER no encontrado ({ruta_ner}); se evalua solo transcripcion.")

    ruta_unido = salida / "transcripciones_por_documento" / f"{nombre}__{nivel}.json"
    ruta_unido.parent.mkdir(parents=True, exist_ok=True)
    ruta_unido.write_text(json.dumps(unido, ensure_ascii=False, indent=1), encoding="utf-8")

    norm, aud, ref_cfg = cfg["normalizacion"], cfg["auditoria"], cfg["referencia"]
    filas_doc, filas_ent, candidatos = [], [], []
    conteos = {"visible": [], "completo": []}
    lectura_total = {"legibles": 0, "exactas": 0, "aproximadas": 0, "perdidas": 0}
    categorias = {c: 0 for c in CATEGORIAS}
    fp_compartidos = 0
    caracteres_extra = 0

    for doc_id, trans in unido.items():
        info = docs_manifest[doc_id]["niveles"][nivel]
        hip = "" if trans["valvula"] else trans["texto"]
        c_vis = comparar("\n".join(info["texto_visible_por_pagina"]), hip, norm)
        c_comp = comparar("\n".join(info["texto_por_pagina"]), hip, norm)
        conteos["visible"].append(c_vis)
        conteos["completo"].append(c_comp)

        extra = " ".join(x["texto"] for pg in info["paginas"] for x in pg["texto_extra"])
        caracteres_extra += len(normalizar_texto(extra))
        estados = {x["entidad"]: x["estado"] for x in info["entidades"]["no_visibles"]}
        excluir = {"oculta", "no_localizada"} | ({"parcial"} if ref_cfg.get("excluir_parciales") else set())
        gold_doc = list(dict.fromkeys(gold[doc_id]["entities_gold"]))
        legibles = list(dict.fromkeys(normalizar_entidad(e) for e in gold_doc if estados.get(e) not in excluir))
        hip_n = normalizar_texto(hip)
        lect = lectura_entidades(legibles, hip_n, aud)
        for k in lectura_total:
            lectura_total[k] += lect[k]

        cat_doc = {c: 0 for c in CATEGORIAS}
        if ner is not None:
            filas = clasificar_predichas(
                ner.get(doc_id, []), {normalizar_entidad(e) for e in gold_doc},
                normalizar_texto(gold[doc_id]["text"]), normalizar_texto(extra), hip_n,
                {normalizar_entidad(e) for e in ner_texto.get(doc_id, [])} if ner_texto is not None else None, aud)
            for f in filas:
                cat_doc[f["categoria"]] += 1
                categorias[f["categoria"]] += 1
                if f["categoria"] == "fp_ner" and f["en_sistema_texto"]:
                    fp_compartidos += 1
                filas_ent.append({"sistema": nombre, "nivel": nivel, "doc_id": doc_id, **f})
                if f["categoria"] in ("mal_leida", "inventada"):
                    candidatos.append({"sistema": nombre, "nivel": nivel, "doc_id": doc_id, "entidad": f["entidad"],
                                       "categoria": f["categoria"], "parecido": f["parecido"],
                                       "fragmento_original": f["fragmento_original"],
                                       "afirmacion": f"La nota menciona '{f['entidad']}'.",
                                       "imagenes": trans["imagenes"]})

        tv, tc = tasas(c_vis), tasas(c_comp)
        estilo = info.get("estilo_manuscrito") or {}
        filas_doc.append({
            "sistema": nombre, "backend": backend, "nivel": nivel, "familia": resumen_nivel.get("familia"),
            "severidad": resumen_nivel.get("severidad"), "doc_id": doc_id,
            "caligrafia": estilo.get("fuente"), "dificultad_caligrafia": estilo.get("dificultad_fuente"),
            "paginas": trans["n_paginas"], "paginas_valvula": trans["paginas_valvula"],
            "paginas_error": trans["paginas_error"], "paginas_faltantes": trans["paginas_faltantes"],
            "segundos": trans["segundos"],
            "cer_visible": tv["cer"], "wer_visible": tv["wer"], "cer_sin_diacriticos_visible": tv["cer_sin_diacriticos"],
            "cer_completo": tc["cer"], "wer_completo": tc["wer"],
            "caracteres_ref_visible": c_vis["caracteres_ref"], "caracteres_hip": c_vis["caracteres_hip"],
            "errores_caracter_visible": c_vis["errores_caracter"],
            "inserciones_caracter": c_vis["ops_caracter"]["inserciones"],
            "borrados_caracter": c_vis["ops_caracter"]["borrados"],
            "sustituciones_caracter": c_vis["ops_caracter"]["sustituciones"],
            "caracteres_extra_en_imagen": len(normalizar_texto(extra)),
            "gold_legibles": lect["legibles"], "gold_leidas_exactas": lect["exactas"],
            "gold_leidas_aproximadas": lect["aproximadas"], "gold_perdidas": lect["perdidas"],
            **({f"ent_{c}": cat_doc[c] for c in CATEGORIAS} if ner is not None else {}),
        })

    esperados = [d["doc_id"] for d in manifest["documentos"] if nivel in d["niveles"]]
    tot_v, tot_c = sumar(conteos["visible"]), sumar(conteos["completo"])
    n_pred = sum(categorias.values())

    def tasa(a, b):
        return round(a / b, 4) if b else None

    resultado = {
        "sistema": nombre, "backend": backend, "nivel": nivel,
        "familia": resumen_nivel.get("familia"), "severidad": resumen_nivel.get("severidad"),
        "transcripciones": str(ruta_t), "ner": entrada.get("ner"),
        "cache_por_documento": str(ruta_unido),
        "documentos": len(unido), "documentos_esperados": len(esperados),
        "cobertura": tasa(len(unido), len(esperados)),
        "paginas": sum(d["n_paginas"] for d in unido.values()),
        "paginas_valvula": sum(d["paginas_valvula"] for d in unido.values()),
        "paginas_error": sum(d["paginas_error"] for d in unido.values()),
        "avisos": {**{k: len(v) for k, v in avisos.items()}, "ner_por_pagina": ner_por_pagina,
                   "ejemplos": {k: v[:5] for k, v in avisos.items() if v}},
        "visible": {**tasas(tot_v), "conteos": tot_v},
        "completo": {**tasas(tot_c), "conteos": tot_c},
        "macro_visible": {
            "cer": _media([f["cer_visible"] for f in filas_doc]),
            "wer": _media([f["wer_visible"] for f in filas_doc]),
        },
        # Cota de cuanto pueden inflar el CER los sellos, notas y tachaduras si el VLM los copia.
        "texto_extra_pct_de_referencia": tasa(caracteres_extra, tot_v["caracteres_ref"]),
        "lectura_entidades": {**lectura_total,
                              "tasa_exacta": tasa(lectura_total["exactas"], lectura_total["legibles"]),
                              "tasa_exacta_o_aproximada": tasa(lectura_total["exactas"] + lectura_total["aproximadas"],
                                                               lectura_total["legibles"])},
        "por_dificultad_caligrafia": _por_grupo(filas_doc, conteos["visible"], "dificultad_caligrafia"),
        "_docs": {f["doc_id"]: c for f, c in zip(filas_doc, conteos["visible"])},
    }
    if ner is not None:
        resultado["auditoria_entidades"] = {
            "predichas": n_pred, **categorias,
            "tasa_inventadas": tasa(categorias["inventada"], n_pred),
            "tasa_mal_leidas": tasa(categorias["mal_leida"], n_pred),
            "tasa_fp_ner": tasa(categorias["fp_ner"], n_pred),
            "fp_ner_tambien_en_sistema_texto": fp_compartidos if ner_texto is not None else None,
        }
    return {"resultado": resultado, "filas_doc": filas_doc, "filas_ent": filas_ent, "candidatos": candidatos}


def _media(valores):
    vals = [v for v in valores if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _por_grupo(filas: list[dict], conteos: list[dict], campo: str) -> dict:
    grupos = defaultdict(list)
    for f, c in zip(filas, conteos):
        if f.get(campo):
            grupos[f[campo]].append(c)
    return {g: {"documentos": len(cs), **tasas(sumar(cs))} for g, cs in sorted(grupos.items())}


def _techo_mismos_docs(resultados: list[dict]) -> None:
    """CER de 'limpio' del mismo sistema restringido a los documentos de cada nivel (comparacion justa)."""
    limpios = {r["sistema"]: r for r in resultados if r["nivel"] == "limpio"}
    for r in resultados:
        base = limpios.get(r["sistema"])
        if base is None or r["nivel"] == "limpio":
            continue
        comunes = [d for d in r["_docs"] if d in base["_docs"]]
        if comunes:
            t = tasas(sumar([base["_docs"][d] for d in comunes]))
            r["limpio_mismos_docs"] = {"documentos": len(comunes), "cer": t["cer"], "wer": t["wer"]}


# --------------------------------------------------------------------------- #
# Reportes
# --------------------------------------------------------------------------- #

def _escribir_csv(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def _pct(x):
    return "--" if x is None else f"{100 * x:.1f} %"


def _num(x):
    return "--" if x is None else f"{x:.3f}"


def _markdown(resumen: dict) -> str:
    lineas = []
    if resumen["simulado"]:
        lineas += ["> **SIMULADO.** Estas cifras salen de transcripciones simuladas con errores inyectados.",
                   "> Sirven para validar el modulo, no son resultados del sistema.", ""]
    lineas += [f"# Metricas de transcripcion (M4, tarea 5a)", "",
               f"Generado: {resumen['fecha']} · referencia principal: texto **{resumen['referencia_principal']}** · "
               f"manifiesto: `{resumen['manifest']}`", "",
               "CER y WER micro (errores totales / tamano total de la referencia). `--` = no aplica.", "",
               "| Sistema | Nivel | Docs | CER visible | WER visible | CER completo | CER sin tildes | "
               "CER limpio (mismos docs) | Entidades leidas | Inventadas | Mal leidas |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in resumen["resultados"]:
        a = r.get("auditoria_entidades") or {}
        techo = (r.get("limpio_mismos_docs") or {}).get("cer")
        lineas.append(
            f"| {r['sistema']} | {r['nivel']} | {r['documentos']}/{r['documentos_esperados']} | "
            f"{_num(r['visible']['cer'])} | {_num(r['visible']['wer'])} | {_num(r['completo']['cer'])} | "
            f"{_num(r['visible']['cer_sin_diacriticos'])} | {_num(techo)} | "
            f"{_pct(r['lectura_entidades']['tasa_exacta'])} | {_pct(a.get('tasa_inventadas'))} | "
            f"{_pct(a.get('tasa_mal_leidas'))} |")
    lineas += ["", "**Entidades leidas:** entidades del gold visibles en la imagen que aparecen tal cual en la "
               "transcripcion (no depende del NER). **Inventadas / mal leidas:** fraccion de las entidades predichas "
               "por el NER que no estan en el texto original (ver auditoria.py).", ""]

    con_aud = [r for r in resumen["resultados"] if r.get("auditoria_entidades")]
    if con_aud:
        lineas += ["## Origen de las entidades predichas", "",
                   "| Sistema | Nivel | Predichas | Acierto | FP del NER | Texto extra | Mal leida | Inventada | Fuera de transcripcion |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for r in con_aud:
            a = r["auditoria_entidades"]
            lineas.append(f"| {r['sistema']} | {r['nivel']} | {a['predichas']} | {a['acierto']} | {a['fp_ner']} | "
                          f"{a['texto_extra']} | {a['mal_leida']} | {a['inventada']} | {a['fuera_de_transcripcion']} |")
        lineas.append("")

    dif = [(r, g, v) for r in resumen["resultados"] for g, v in r["por_dificultad_caligrafia"].items()]
    if dif:
        lineas += ["## CER por dificultad de la caligrafia (niveles manuscritos)", "",
                   "| Sistema | Nivel | Dificultad | Docs | CER | WER |", "|---|---|---|---|---|---|"]
        for r, g, v in dif:
            lineas.append(f"| {r['sistema']} | {r['nivel']} | {g} | {v['documentos']} | {_num(v['cer'])} | {_num(v['wer'])} |")
        lineas += ["", "La dificultad de cada caligrafia se etiqueto a ojo (ver M4/eval_visual/config.yaml).", ""]

    avisos = [r for r in resumen["resultados"] if any(v for k, v in r["avisos"].items() if k not in ("ejemplos",))]
    if avisos:
        lineas += ["## Avisos", ""]
        for r in avisos:
            lineas.append(f"- {r['sistema']} / {r['nivel']}: {json.dumps({k: v for k, v in r['avisos'].items() if k != 'ejemplos'}, ensure_ascii=False)}")
        lineas.append("")
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #

def run(cfg: dict, project_root: Path, entradas_extra: list[dict] | None = None, simular: bool = False) -> dict:
    t0 = time.time()
    rutas = cfg["rutas"]
    salida = _ruta(project_root, rutas["salida_dir"]) / ("simulado" if simular else "")
    salida.mkdir(parents=True, exist_ok=True)

    ruta_manifest = _ruta(project_root, rutas["manifest"])
    if not ruta_manifest.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {ruta_manifest}. Corre antes M4/eval_visual/generar_set.py")
    manifest = cargar_json(ruta_manifest)
    docs_manifest = {d["doc_id"]: d for d in manifest["documentos"]}
    gold = _cargar_gold(manifest, rutas)

    ner_texto = None
    candidatos_ner_texto = [
        rutas.get("ner_texto"),
        "M2/predictions/texto_techo_m4.json",
        "M2/predictions/encoder_solo.json",
    ]
    for c_ner in candidatos_ner_texto:
        if c_ner:
            r_c = _ruta(project_root, c_ner)
            if r_c.exists():
                cargado, _ = _ner_por_documento(cargar_json(r_c))
                if any(d in cargado for d in docs_manifest):
                    ner_texto = cargado
                    log(f"Cache NER del sistema sobre texto: {c_ner}")
                    break

    if simular:
        import simular as sim
        entradas = sim.generar(cfg, manifest, gold, salida / "caches")
        log(f"Simulacion: {len(entradas)} niveles en {salida / 'caches'}")
    else:
        entradas = list(entradas_extra) if entradas_extra else list(cfg.get("entradas") or [])
    if not entradas:
        raise ValueError("No hay entradas. Agregalas en config.yaml o con --transcripciones (o usa --simular).")

    evaluaciones = []
    for e in entradas:
        log(f"Evaluando {e['transcripciones']}")
        evaluaciones.append(evaluar_entrada(e, manifest, docs_manifest, gold, ner_texto, cfg, project_root, salida))
    resultados = [ev["resultado"] for ev in evaluaciones]
    _techo_mismos_docs(resultados)
    for r in resultados:
        r.pop("_docs")

    resumen = {
        "modulo": "metricas_transcripcion", "responsable": "luis", "tarea": "5a",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "simulado": simular, "manifest": str(ruta_manifest),
        "referencia_principal": cfg["referencia"]["principal"],
        "normalizacion": cfg["normalizacion"], "auditoria": cfg["auditoria"],
        "resultados": resultados,
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "resumen.md").write_text(_markdown(resumen), encoding="utf-8")
    _escribir_csv(salida / "por_documento.csv", [f for ev in evaluaciones for f in ev["filas_doc"]])
    _escribir_csv(salida / "entidades.csv", [f for ev in evaluaciones for f in ev["filas_ent"]])
    with open(salida / "candidatos_juez.jsonl", "w", encoding="utf-8") as f:
        for c in (c for ev in evaluaciones for c in ev["candidatos"]):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if simular:
        import validar_simulacion
        resumen["validacion"] = validar_simulacion.validar(salida, resultados)

    log(f"Listo en {time.time() - t0:.1f} s: {salida}")
    return resumen


def main():
    parser = argparse.ArgumentParser(description="Metricas de transcripcion (M4, tarea 5a)")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--transcripciones", nargs="+", default=None, help="Caches de M4/vision (uno por nivel)")
    parser.add_argument("--ner", nargs="+", default=None, help="Caches NER, en el mismo orden que --transcripciones")
    parser.add_argument("--nivel", default=None, help="Nivel, si no se puede deducir de las rutas de imagen")
    parser.add_argument("--nombre", default=None, help="Nombre del sistema en los reportes")
    parser.add_argument("--simular", action="store_true", help="Probar con transcripciones simuladas")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    extra = []
    if args.transcripciones:
        if args.ner and len(args.ner) != len(args.transcripciones):
            parser.error("--ner debe tener tantas rutas como --transcripciones")
        for i, t in enumerate(args.transcripciones):
            e = {"transcripciones": t}
            if args.ner:
                e["ner"] = args.ner[i]
            if args.nivel:
                e["nivel"] = args.nivel
            if args.nombre:
                e["nombre"] = args.nombre
            extra.append(e)
    resumen = run(cfg, resolver_project_root(), entradas_extra=extra, simular=args.simular)
    for r in resumen["resultados"]:
        a = r.get("auditoria_entidades") or {}
        print(f"{r['sistema']:12s} {r['nivel']:28s} CER_vis={r['visible']['cer']} WER_vis={r['visible']['wer']} "
              f"lectura={r['lectura_entidades']['tasa_exacta']} inventadas={a.get('tasa_inventadas')}")
    if resumen.get("validacion"):
        print(json.dumps(resumen["validacion"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
