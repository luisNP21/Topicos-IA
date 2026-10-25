"""Entry point de la tarea 3a: genera el set sintetico de imagenes.

Uso:
    python generar_set.py                              # todos los niveles de config.yaml
    python generar_set.py --limite 3                   # primeros 3 documentos (revision rapida)
    python generar_set.py --niveles limpio foto_realista

Lee el gold set, renderiza cada documento (impreso o a mano), aplica los efectos de
cada nivel y escribe:
- las imagenes:            <PROJECT_ROOT>/M4/data/imagenes/<nivel>/<doc_id>_p<k>.<ext>
- el manifiesto:           <PROJECT_ROOT>/M4/data/imagenes/manifest.json
- un CSV de casos por nivel (formato de M4/vision): <PROJECT_ROOT>/M4/data/imagenes/casos/<nivel>.csv

El gold sigue siendo el texto original. Cada imagen conserva el doc_id que asigna
M2/harness/gold_loader.cargar_gold_set, el mismo que usa el harness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

for extra in (REPO_ROOT / "M2", REPO_ROOT / "M2" / "harness", SCRIPT_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import log  # noqa: E402  (M2/common.py)
from gold_loader import cargar_gold_set  # noqa: E402  (M2/harness/gold_loader.py)

from degradaciones import EFECTOS  # noqa: E402
from manuscrito import elegir_estilo, renderizar_manuscrito, tiene_glifo  # noqa: E402
from render import cargar_fuente, renderizar_impreso, resolver_fuente, texto_de_lineas  # noqa: E402
from utilidades import Pagina, puntos_de_cajas, redondear, semilla  # noqa: E402
from visibilidad import estado_entidades, fraccion_visible_por_token  # noqa: E402

LETRAS_ESPANOL = "áéíóúÁÉÍÓÚñÑüÜ"


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


def cargar_eval_set_m3(ruta: Path) -> list[dict]:
    """Carga el eval set de M3 (M3/ejecucion/eval_set_casos.json) con el formato del gold de M2.

    Cada caso aporta el texto (`input`) que se renderiza como imagen. El `id` de M3
    (caso_01...) pasa a ser el doc_id de las imagenes. La entidad del caso es la unica
    entidad de referencia: sirve para saber si quedo visible en la imagen. Si la cadena
    no aparece tal cual en el texto, queda como `no_localizada`.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Eval set de M3 no encontrado: {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        casos = json.load(f)
    if not isinstance(casos, list) or not casos:
        raise ValueError(f"El eval set de M3 esta vacio o no es una lista: {ruta}")
    registros = []
    for i, c in enumerate(casos):
        texto = (c.get("input") or "").strip()
        if not texto:
            raise ValueError(f"Caso {c.get('id', i)} de M3 sin campo 'input'")
        entidad = (c.get("entidad") or "").strip()
        registros.append({
            "doc_id": str(c.get("id") or f"caso_{i + 1:02d}"),
            "text": texto,
            "entities_gold": [entidad] if entidad else [],
            "criterio": "",
        })
    ids = [r["doc_id"] for r in registros]
    if len(set(ids)) != len(ids):
        raise ValueError("ids repetidos en el eval set de M3")
    log(f"Eval set de M3 cargado: {len(registros)} casos ({ruta.name})")
    return registros


def _cargar_documentos(cfg: dict) -> list[dict]:
    entrada = cfg["entrada"]
    fuente = entrada.get("fuente", "gold_m2")
    if fuente == "m3":
        return cargar_eval_set_m3(REPO_ROOT / entrada["m3_eval_set"])
    if fuente == "gold_m2":
        return cargar_gold_set(REPO_ROOT / entrada["gold_set"])
    raise ValueError(f"entrada.fuente debe ser 'm3' o 'gold_m2', no {fuente!r}")


def _validar(cfg: dict, seleccion: list[str]) -> None:
    niveles = cfg["niveles"]
    desconocidos = [n for n in seleccion if n not in niveles]
    if desconocidos:
        raise ValueError(f"Niveles no definidos: {desconocidos}. Disponibles: {list(niveles)}")
    for n in seleccion:
        ncfg = niveles[n]
        if ncfg.get("render", "impreso") not in ("impreso", "manuscrito"):
            raise ValueError(f"{n}: render debe ser impreso o manuscrito")
        if ncfg.get("documentos", "todos") not in ("todos", *cfg.get("subconjuntos", {})):
            raise ValueError(f"{n}: documentos debe ser 'todos' o un subconjunto definido")
        for e in ncfg.get("efectos", []):
            if e["tipo"] not in EFECTOS:
                raise ValueError(f"{n}: efecto desconocido {e['tipo']!r}. Disponibles: {sorted(EFECTOS)}")


def _fuentes_manuscritas(cfg: dict) -> list[dict]:
    fuentes = []
    for f in cfg.get("fuentes_manuscritas", []):
        ruta = Path(f["ruta"])
        ruta = ruta if ruta.is_absolute() else REPO_ROOT / ruta
        if not ruta.exists():
            raise FileNotFoundError(f"Fuente manuscrita no encontrada: {ruta}")
        faltan = [c for c in LETRAS_ESPANOL if not tiene_glifo(str(ruta), c)]
        if faltan:
            raise ValueError(f"La fuente {f['nombre']} no tiene {''.join(faltan)}")
        fuentes.append({**f, "ruta": str(ruta)})
    return fuentes


def _subconjuntos(cfg: dict, doc_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Mismo subconjunto para todos los niveles que lo usan (comparables entre severidades)."""
    orden = sorted(doc_ids, key=lambda d: semilla(seed, "subconjunto", d))
    subs = {"todos": list(doc_ids)}
    for nombre, n in (cfg.get("subconjuntos") or {}).items():
        elegidos = set(orden[:int(n)])
        subs[nombre] = [d for d in doc_ids if d in elegidos]
    return subs


def _aplicar_efectos(pag: Pagina, efectos: list[dict], seed: int, doc_id: str, k: int, ctx: dict) -> list[dict]:
    registro = []
    for i, efecto in enumerate(efectos):
        tipo = efecto["tipo"]
        params = {c: v for c, v in efecto.items() if c not in ("tipo", "prob")}
        # La semilla no depende del nombre del nivel: el mismo documento recibe los mismos
        # sorteos en leve/moderado/severo y solo cambia la intensidad.
        s = semilla(seed, "efecto", doc_id, k, i, tipo)
        rng = np.random.default_rng(s)
        if "prob" in efecto and rng.random() >= float(efecto["prob"]):
            registro.append({"tipo": tipo, "aplicado": False, "semilla": s})
            continue
        usados = EFECTOS[tipo](pag, rng, params, ctx)
        registro.append({"tipo": tipo, "aplicado": True, "semilla": s, "params_usados": redondear(usados)})
    return registro


def _guardar(pag: Pagina, ruta: Path, ncfg: dict, rng) -> dict:
    img = pag.a_pil()
    if ncfg.get("formato", "png") == "jpg":
        valor = ncfg.get("calidad_jpeg", 90)
        calidad = int(round(rng.uniform(*valor))) if isinstance(valor, list) else int(valor)
        img.save(ruta, format="JPEG", quality=calidad)
        return {"formato": "jpg", "calidad_jpeg": calidad}
    img.save(ruta, format="PNG")
    return {"formato": "png"}


_TRABAJO: dict = {}   # contexto de cada proceso (se llena en _iniciar_proceso)


def _iniciar_proceso(trabajo: dict) -> None:
    _TRABAJO.clear()
    _TRABAJO.update(trabajo)
    _TRABAJO["fuente"] = cargar_fuente(trabajo["ctx"]["fuente_impresa"], trabajo["cfg"]["render"]["tamano_fuente"])


def _procesar_documento(rec: dict) -> dict:
    """Genera todas las imagenes de un documento. Solo depende de semillas por documento,
    asi que el resultado es el mismo en serie o en paralelo."""
    cfg, ctx, subs = _TRABAJO["cfg"], _TRABAJO["ctx"], _TRABAJO["subs"]
    seleccion, imagenes_dir, umbral = _TRABAJO["seleccion"], _TRABAJO["imagenes_dir"], _TRABAJO["umbral"]
    seed, niveles_cfg = cfg["proyecto"]["seed_global"], cfg["niveles"]
    doc_id, texto, gold = rec["doc_id"], rec["text"], rec["entities_gold"]
    entrada = {"doc_id": doc_id, "n_caracteres": len(texto),
               "sha256_texto": hashlib.sha256(texto.encode("utf-8")).hexdigest(), "niveles": {}}
    casos, tiempos = {}, {}
    impreso = None

    for nivel in seleccion:
        ncfg = niveles_cfg[nivel]
        if doc_id not in subs[ncfg.get("documentos", "todos")]:
            continue
        t0 = time.time()
        estilo = None
        if ncfg.get("render", "impreso") == "manuscrito":
            estilo_doc = elegir_estilo(ncfg["manuscrito"], ctx["fuentes_manuscritas"],
                                       np.random.default_rng(semilla(seed, "estilo_manuscrito", doc_id)))
            base = renderizar_manuscrito(texto, ncfg["manuscrito"], estilo_doc, ctx["fuente_impresa"],
                                         np.random.default_rng(semilla(seed, "manuscrito", doc_id)))
            estilo = base["estilo"]
        else:
            impreso = impreso or renderizar_impreso(texto, _TRABAJO["fuente"], cfg["render"])
            base = impreso
        tokens = base["tokens"]

        registros, pt_tokens, vivos, filas = [], [], [], []
        for k, pr in enumerate(base["paginas"], start=1):
            pts, pt_tok = pr["pts"] if "pts" in pr else puntos_de_cajas(pr["cajas"])
            pag = Pagina(img=pr["imagen"].copy(), pts=pts.copy(), pt_token=pt_tok,
                         vivo=np.ones(len(pt_tok), bool), alto_linea=pr["alto_linea"],
                         color_papel=pr["color_papel"], extra=list(pr["extra"]))
            efectos = _aplicar_efectos(pag, ncfg.get("efectos", []), seed, doc_id, k, ctx)
            ext = "jpg" if ncfg.get("formato", "png") == "jpg" else "png"
            nombre = f"{doc_id}_p{k}.{ext}"
            guardado = _guardar(pag, imagenes_dir / nivel / nombre, ncfg,
                                np.random.default_rng(semilla(seed, "guardado", nivel, doc_id, k)))
            pt_tokens.append(pag.pt_token)
            vivos.append(pag.vivo)
            registros.append({"pagina": k, "archivo": f"{nivel}/{nombre}", "tamano_px": [pag.ancho, pag.alto],
                              **guardado, "efectos": efectos, "texto_extra": pag.extra})
            filas.append({"id": f"{doc_id}__p{k}", "doc_id": doc_id, "pagina": k,
                          "n_paginas": len(base["paginas"]), "imagen": f"{nivel}/{nombre}",
                          "nivel": nivel, "familia": ncfg.get("familia", ""),
                          "severidad": ncfg.get("severidad", ""), "tipo": "sintetico"})

        frac = fraccion_visible_por_token(len(tokens), pt_tokens, vivos)
        visibles = frac >= umbral
        conjunto = set(np.where(visibles)[0].tolist())
        entrada["niveles"][nivel] = {
            "n_paginas": len(base["paginas"]),
            "texto_por_pagina": [texto_de_lineas(tokens, p["lineas"]) for p in base["paginas"]],
            "texto_visible_por_pagina": [texto_de_lineas(tokens, p["lineas"], conjunto) for p in base["paginas"]],
            "palabras": {"total": len(tokens), "ocultas": int((~visibles).sum())},
            "entidades": estado_entidades(texto, gold, tokens, visibles),
            **({"estilo_manuscrito": estilo} if estilo else {}),
            "paginas": registros,
        }
        casos[nivel] = filas
        tiempos[nivel] = time.time() - t0
    return {"entrada": entrada, "casos": casos, "tiempos": tiempos}


def run(cfg: dict, project_root: Path, limite: int | None = None, niveles: list[str] | None = None,
        procesos: int = 1) -> dict:
    seed = cfg["proyecto"]["seed_global"]
    niveles_cfg = cfg["niveles"]
    seleccion = niveles or list(niveles_cfg)
    _validar(cfg, seleccion)
    umbral = float(cfg.get("visibilidad", {}).get("umbral_token", 0.5))

    eval_set = _cargar_documentos(cfg)
    if limite is not None:
        eval_set = eval_set[:limite]
    subs = _subconjuntos(cfg, [r["doc_id"] for r in eval_set], seed)

    imagenes_dir = project_root / cfg["salida"]["imagenes_dir"]
    manifest_path = project_root / cfg["salida"]["manifest_path"]
    casos_dir = project_root / cfg["salida"].get("casos_dir", f"{cfg['salida']['imagenes_dir']}/casos")
    for carpeta in (manifest_path.parent, casos_dir, *(imagenes_dir / n for n in seleccion)):
        carpeta.mkdir(parents=True, exist_ok=True)

    fuente_ruta = resolver_fuente(cfg["render"], REPO_ROOT)
    fuentes_m = _fuentes_manuscritas(cfg)
    ctx = {"fuente_impresa": fuente_ruta, "fuentes_manuscritas": fuentes_m}
    log(f"Fuente impresa: {fuente_ruta} | caligrafias: {len(fuentes_m)}")
    log(f"Documentos: {len(eval_set)} | niveles: {len(seleccion)} | barrido: {len(subs.get('barrido', []))} docs"
        f" | procesos: {procesos}")

    trabajo = {"cfg": cfg, "ctx": ctx, "subs": subs, "seleccion": seleccion,
               "imagenes_dir": imagenes_dir, "umbral": umbral}
    documentos, casos = [], {n: [] for n in seleccion}
    resumen = {n: {"documentos": 0, "paginas": 0, "palabras": 0, "palabras_ocultas": 0,
                   "entidades": {"total": 0, "visible": 0, "parcial": 0, "oculta": 0, "no_localizada": 0},
                   "segundos": 0.0} for n in seleccion}
    n_imagenes = 0
    t_inicio = time.time()

    if procesos > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(procesos, initializer=_iniciar_proceso, initargs=(trabajo,))
        resultados = pool.imap(_procesar_documento, eval_set)   # imap conserva el orden de los documentos
    else:
        pool = None
        _iniciar_proceso(trabajo)
        resultados = map(_procesar_documento, eval_set)

    try:
        for i, res in enumerate(resultados, start=1):
            entrada = res["entrada"]
            documentos.append(entrada)
            for nivel, info in entrada["niveles"].items():
                casos[nivel].extend(res["casos"][nivel])
                r = resumen[nivel]
                r["documentos"] += 1
                r["paginas"] += info["n_paginas"]
                r["palabras"] += info["palabras"]["total"]
                r["palabras_ocultas"] += info["palabras"]["ocultas"]
                for c in ("total", "visible", "parcial", "oculta", "no_localizada"):
                    r["entidades"][c] += info["entidades"][c]
                r["segundos"] += res["tiempos"][nivel]
                n_imagenes += info["n_paginas"]
            if i % 5 == 0 or i == len(eval_set):
                log(f"  {i}/{len(eval_set)} documentos | {n_imagenes} imagenes | {time.time() - t_inicio:.0f} s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    for nivel, filas in casos.items():
        with open(casos_dir / f"{nivel}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "doc_id", "pagina", "n_paginas", "imagen", "nivel",
                                              "familia", "severidad", "tipo"])
            w.writeheader()
            w.writerows(filas)

    for nivel, r in resumen.items():
        ncfg = niveles_cfg[nivel]
        r.update(descripcion=ncfg.get("descripcion", ""), familia=ncfg.get("familia", ""),
                 severidad=ncfg.get("severidad", ""), subconjunto=ncfg.get("documentos", "todos"),
                 pct_palabras_ocultas=round(100 * r["palabras_ocultas"] / max(1, r["palabras"]), 2),
                 segundos=round(r["segundos"], 1))

    manifest = {
        "descripcion": (
            "Set sintetico de imagenes derivado de los documentos del gold set. El gold es el texto "
            "original; las imagenes son sinteticas y no equivalen a escaneos o fotos reales."
        ),
        "fuente": cfg["entrada"].get("fuente", "gold_m2"),
        "gold_set": cfg["entrada"].get("m3_eval_set" if cfg["entrada"].get("fuente") == "m3" else "gold_set"),
        "seed_global": seed,
        "fuentes": {"impresa": Path(fuente_ruta).name, "manuscritas": [f["nombre"] for f in fuentes_m]},
        "versiones": {"python": sys.version.split()[0], "Pillow": PIL.__version__, "numpy": np.__version__},
        "visibilidad": {"umbral_token": umbral, "puntos_por_palabra": 15},
        "subconjuntos": subs,
        "niveles": resumen,
        "niveles_config": {n: niveles_cfg[n] for n in seleccion},
        "render": cfg["render"],
        "n_documentos": len(documentos),
        "n_imagenes": n_imagenes,
        "documentos": documentos,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(redondear(manifest), f, ensure_ascii=False, indent=1)

    log(f"Listo: {n_imagenes} imagenes en {imagenes_dir} ({time.time() - t_inicio:.0f} s)")
    log(f"Manifiesto: {manifest_path}")
    log(f"Casos por nivel: {casos_dir}")
    return {"n_documentos": len(documentos), "n_imagenes": n_imagenes, "imagenes_dir": str(imagenes_dir),
            "manifest_path": str(manifest_path), "casos_dir": str(casos_dir)}


def main():
    parser = argparse.ArgumentParser(description="Genera el set sintetico de imagenes (M4, 3a)")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--limite", type=int, default=None, help="Solo los primeros N documentos")
    parser.add_argument("--niveles", nargs="+", default=None, help="Subconjunto de niveles")
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Documentos en paralelo (por defecto, nucleos - 1). El resultado no cambia.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stats = run(cfg, resolver_project_root(), limite=args.limite, niveles=args.niveles, procesos=args.procesos)
    print(stats)


if __name__ == "__main__":
    main()
