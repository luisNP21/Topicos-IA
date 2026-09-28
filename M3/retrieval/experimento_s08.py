"""
Experimento controlado de retrieval (protocolo del laboratorio S08, ampliado).

Todos los sistemas buscan en el mismo índice y reciben las mismas consultas; solo cambia la
técnica de búsqueda, así que cualquier diferencia se atribuye a la técnica. Se mide:

  * si llega la guía correcta (acierto en el primer lugar, en el top-k, MRR, precisión);
  * si llega la sección de tratamiento, cuando la consulta tiene chunks etiquetados;
  * el rechazo de consultas cuya enfermedad no está en el corpus;
  * la latencia por consulta.

El sistema completo (normalización más compuerta de evidencia) necesita umbrales. Para no
calibrarlos y evaluarlos sobre las mismas consultas se usa validación cruzada estratificada
por tipo de consulta: cada consulta se evalúa con umbrales calibrados en los otros pliegues.
Los umbrales finales, para usar en producción, se calibran con todas las consultas.

Uso, desde la raíz del repositorio (rutas y parámetros en config_retrieval.yaml):
    python M3/retrieval/experimento_s08.py
    python M3/retrieval/experimento_s08.py --corpus real
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "generacion"))

from orquestacion import es_sigla, resolver_query  # noqa: E402
from retrieval import TIPO_SCORE, IndiceRAG, info_indice  # noqa: E402

SISTEMAS = [
    ("ingenuo", "Denso (RAG ingenuo)", "densa"),
    ("hibrido", "Híbrido BM25 + denso", "hibrida"),
    ("reranker", "Híbrido + reranker", "rerank"),
    ("reranker_pregunta", "Híbrido + reranker con pregunta de tratamiento", "rerank_pregunta"),
    ("intencion_sin_reranker", "Consultas de tratamiento, sin reranker", "intencion_sin_rerank"),
    ("intencion", "Consultas de tratamiento + reranker", "intencion"),
    ("intencion_seccion", "Consultas de tratamiento + reranker + filtro de sección", "intencion_seccion"),
]
NOMBRE = {clave: nombre for clave, nombre, _ in SISTEMAS}
MODO = {clave: modo for clave, _, modo in SISTEMAS}
NOMBRE["completo"] = "Sistema completo (normalización + compuerta)"
NOMBRE["completo_siglas"] = "Sistema completo + regla de siglas"

TIPOS_CONSULTA = {"exacta": "Nombre exacto", "sigla": "Sigla", "sinonimo": "Sinónimo",
                  "errata": "Errata o sin tildes", "hiper_especifica": "Hiperespecífica",
                  "confusion": "Guías parecidas", "vaga": "Entidad vaga",
                  "fuera_de_corpus": "Fuera del corpus"}


def etiquetas_resumen(k: int) -> dict[str, str]:
    return {"consultas": "Consultas con guía en el corpus",
            "hit1": "Guía correcta en 1.er lugar", "hitk": f"Guía correcta en el top-{k}",
            "rr": "MRR de la guía", "recall": f"Guías relevantes en el top-{k}",
            "precision": f"Precisión de guía en el top-{k}",
            "hit1_sec": "Tratamiento en 1.er lugar", "hitk_sec": f"Tratamiento en el top-{k}",
            "rr_sec": "MRR del tratamiento", "precision_sec": f"Precisión de tratamiento en el top-{k}",
            "rechazo_fuera": "Rechazo correcto fuera del corpus",
            "rechazo_erroneo": "Rechazo erróneo con guía disponible",
            "tool": "Consultas que usan la normalización", "ms": "Latencia (ms por consulta)",
            "delta_rr": "Cambio en MRR de la guía frente al ingenuo",
            "delta_rr_sec": "Cambio en MRR del tratamiento frente al ingenuo"}


ETIQUETAS_DETALLE = {
    "clave": "Clave del sistema", "sistema": "Sistema", "consulta": "Consulta", "tipo": "Tipo de consulta",
    "fuera_de_corpus": "Fuera del corpus", "pliegue": "Pliegue de validación",
    "hit1": "Guía correcta en 1.er lugar", "hitk": "Guía correcta en el top-k", "rr": "Rango recíproco de la guía",
    "recall": "Guías relevantes en el top-k", "precision": "Precisión de guía en el top-k",
    "puesto": "Puesto de la guía correcta", "hit1_sec": "Tratamiento en 1.er lugar",
    "hitk_sec": "Tratamiento en el top-k", "rr_sec": "Rango recíproco del tratamiento",
    "precision_sec": "Precisión de tratamiento en el top-k", "score_top": "Score del primer fragmento",
    "score_relevante": "Score de la guía correcta", "score_relevante_sec": "Score del tratamiento correcto",
    "top_docs": "Guías recuperadas", "top_chunks": "Chunks recuperados", "seg": "Latencia (s)",
    "rechazo": "Sin fragmentos entregados", "tool": "Usó la normalización", "query_final": "Consulta final",
    "tool_reason": "Motivo de la decisión", "umbral": "Umbral usado", "umbral_evidencia": "Umbral de evidencia usado",
}


# ---------------------------------------------------------------------------
# Consultas y métricas
# ---------------------------------------------------------------------------
def cargar_consultas(ruta: str | Path) -> list[dict]:
    """JSONL con: consulta, tipo, relevantes (doc_id de las guías que responden),
    chunks_relevantes (opcional: chunks de tratamiento) y fuera_de_corpus.
    Las consultas con relevantes = null aún no están etiquetadas y se omiten."""
    consultas, omitidas = [], 0
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            c = json.loads(linea)
            if c.get("fuera_de_corpus"):
                c["relevantes"] = []
            if c.get("relevantes") is None:
                omitidas += 1
                continue
            consultas.append(c)
    if omitidas:
        print(f"Aviso: se omiten {omitidas} consultas sin etiquetar.")
    return consultas


def asignar_pliegues(consultas: list[dict], n: int, semilla: int = 42) -> list[int]:
    """Reparte las consultas en n pliegues, balanceando cada tipo de consulta."""
    rng, pliegues, siguiente = random.Random(semilla), [0] * len(consultas), 0
    por_tipo: dict[str, list[int]] = {}
    for i, c in enumerate(consultas):
        por_tipo.setdefault(c.get("tipo", ""), []).append(i)
    for tipo in sorted(por_tipo):
        indices = por_tipo[tipo]
        rng.shuffle(indices)
        for i in indices:
            pliegues[i] = siguiente % n
            siguiente += 1
    return pliegues


def _medir(fragments: list[dict], relevantes: list[str], k: int,
           chunks_relevantes: list[str] | None = None) -> dict:
    docs = [f["doc_id"] for f in fragments[:k]]
    puesto = next((i for i, d in enumerate(docs, 1) if d in relevantes), None)
    m = {"hit1": int(puesto == 1), "hitk": int(puesto is not None), "rr": 1.0 / puesto if puesto else 0.0,
         "recall": len(set(docs) & set(relevantes)) / len(relevantes) if relevantes else 0.0,
         "precision": sum(d in relevantes for d in docs) / len(docs) if docs and relevantes else None,
         "puesto": puesto, "hit1_sec": None, "hitk_sec": None, "rr_sec": None, "precision_sec": None}
    if chunks_relevantes:
        ids = [f["chunk_id"] for f in fragments[:k]]
        p = next((i for i, c in enumerate(ids, 1) if c in chunks_relevantes), None)
        m.update({"hit1_sec": int(p == 1), "hitk_sec": int(p is not None), "rr_sec": 1.0 / p if p else 0.0,
                  "precision_sec": sum(c in chunks_relevantes for c in ids) / len(ids) if ids else None})
    return m


# ---------------------------------------------------------------------------
# Calibración de umbrales
# ---------------------------------------------------------------------------
def youden(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """Umbral que maximiza TPR - FPR. Ante empates se elige el menor."""
    if not scores_pos or not scores_neg:
        return None
    mejor_t, mejor_j = None, -1.0
    for t in sorted(set(scores_pos + scores_neg)):
        j = sum(s >= t for s in scores_pos) / len(scores_pos) - sum(s >= t for s in scores_neg) / len(scores_neg)
        if j > mejor_j:
            mejor_t, mejor_j = t, j
    return mejor_t


def umbral_por_tolerancia(scores_pos: list[float], scores_neg: list[float],
                          tolerancia: float = 0.0) -> float | None:
    """Menor umbral que deja pasar como máximo una fracción `tolerancia` de consultas fuera
    del corpus. Con tolerancia 0 queda justo por encima del score más alto que obtuvo una
    consulta fuera del corpus: se prioriza no responder con una guía equivocada."""
    if not scores_pos or not scores_neg:
        return None
    candidatos = sorted(set(scores_pos + scores_neg) | {round(max(scores_neg) + 1e-6, 6)})
    for t in candidatos:
        if sum(s >= t for s in scores_neg) / len(scores_neg) <= tolerancia:
            return t
    return None


def calibrar(filas: list[dict], excluir_siglas: bool = True, tolerancia_fuera: float = 0.0) -> dict:
    """Calibra los dos umbrales con las corridas del sistema base.

    umbral: separa las consultas cuyo primer fragmento ya es correcto (la sección de
    tratamiento si está etiquetada; si no, la guía) de las demás, con el índice de Youden.
    umbral_evidencia: separa los fragmentos correctos de los scores de consultas fuera del
    corpus, con la regla de tolerancia. Las siglas se excluyen porque su score no es confiable
    y la regla de siglas las resuelve sin mirarlo."""
    if excluir_siglas:
        filas = [f for f in filas if not es_sigla(f["consulta"])]
    correcto = [f["hit1_sec"] if f["hit1_sec"] is not None else f["hit1"] for f in filas]
    top_ok = [f["score_top"] for f, ok in zip(filas, correcto) if ok == 1 and f["score_top"] is not None]
    top_mal = [f["score_top"] for f, ok in zip(filas, correcto) if ok != 1 and f["score_top"] is not None]
    pos_ev = [f["score_relevante_sec"] if f["score_relevante_sec"] is not None else f["score_relevante"]
              for f in filas if not f["fuera_de_corpus"]]
    pos_ev = [s for s in pos_ev if s is not None]
    neg_ev = [f["score_top"] for f in filas if f["fuera_de_corpus"] and f["score_top"] is not None]

    u = youden(top_ok, top_mal)
    ue = umbral_por_tolerancia(pos_ev, neg_ev, tolerancia_fuera)
    zona_dudosa = u is not None and ue is not None and ue < u
    if u is not None and ue is not None and ue > u:
        ue = u
    return {"umbral": u, "umbral_evidencia": ue, "hay_zona_dudosa": zona_dudosa,
            "evidencia_conservada": (round(sum(s >= ue for s in pos_ev) / len(pos_ev), 3)
                                     if ue is not None and pos_ev else None),
            "n": {"primer_fragmento_correcto": len(top_ok), "primer_fragmento_incorrecto": len(top_mal),
                  "con_evidencia_correcta": len(pos_ev), "fuera_de_corpus": len(neg_ev)}}


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def correr_experimento(indice: IndiceRAG, consultas: list[dict], k: int = 5, normalizar_fn=None,
                       sistema_base: str = "intencion", n_pliegues: int = 3, umbral: float | None = None,
                       umbral_evidencia: float | None = None, variantes_sigla: tuple[bool, ...] = (False, True),
                       claves: tuple[str, ...] | None = None, semilla: int = 42,
                       tolerancia_fuera: float = 0.0) -> tuple[list[dict], list[dict], dict]:
    """Devuelve (detalle por consulta, resumen por sistema, calibración).

    sistema_base es el sistema sobre el que corre la normalización; debe usar reranker.
    Con umbral y umbral_evidencia fijos no se calibra; con n_pliegues=1 se calibra y evalúa
    sobre las mismas consultas."""
    if TIPO_SCORE.get(MODO[sistema_base], "reranker_prob") != "reranker_prob":
        raise ValueError("El sistema base debe usar reranker para que sus scores admitan umbrales.")
    pliegues = asignar_pliegues(consultas, n_pliegues, semilla) if n_pliegues > 1 else [0] * len(consultas)
    detalle: list[dict] = []

    def registrar(clave, c, pliegue, frags, seg, extra=None):
        fila = {"clave": clave, "sistema": NOMBRE[clave], "consulta": c["consulta"], "tipo": c.get("tipo", ""),
                "fuera_de_corpus": bool(c.get("fuera_de_corpus")), "pliegue": pliegue,
                **_medir(frags, c["relevantes"], k, c.get("chunks_relevantes")),
                "score_top": frags[0]["score"] if frags else None,
                "score_relevante": max((f["score"] for f in frags[:k] if f["doc_id"] in c["relevantes"]), default=None),
                "score_relevante_sec": max((f["score"] for f in frags[:k]
                                            if f["chunk_id"] in (c.get("chunks_relevantes") or [])), default=None),
                "top_docs": "|".join(f["doc_id"] for f in frags[:k]),
                "top_chunks": "|".join(f["chunk_id"] for f in frags[:k]), "seg": seg, "rechazo": int(not frags)}
        fila.update(extra or {})
        detalle.append(fila)

    for clave, _, modo in SISTEMAS:
        if claves is not None and clave not in claves and clave != sistema_base:
            continue
        for c, p in zip(consultas, pliegues):
            t0 = time.perf_counter()
            frags = indice.buscar(c["consulta"], k, modo)
            registrar(clave, c, p, frags, time.perf_counter() - t0)

    base = [d for d in detalle if d["clave"] == sistema_base]
    calibracion = {"sistema_base": NOMBRE[sistema_base], "final": calibrar(base, tolerancia_fuera=tolerancia_fuera),
                   "validacion": (f"cruzada, {n_pliegues} pliegues estratificados por tipo de consulta"
                                  if n_pliegues > 1 else "sin separar: calibración y evaluación con las mismas consultas"),
                   "pliegues": []}
    por_pliegue = {}
    if n_pliegues > 1:
        for p in range(n_pliegues):
            cal = calibrar([d for d in base if d["pliegue"] != p], tolerancia_fuera=tolerancia_fuera)
            if cal["umbral"] is None:
                cal = {**calibracion["final"], "nota": "sin datos suficientes en el pliegue; se usa la calibración final"}
            por_pliegue[p] = cal
            calibracion["pliegues"].append({"pliegue": p, **cal})

    if normalizar_fn is not None:
        for forzar in variantes_sigla:
            clave = "completo_siglas" if forzar else "completo"
            for c, p in zip(consultas, pliegues):
                if umbral is not None:
                    u, ue = umbral, umbral_evidencia
                else:
                    cal = por_pliegue.get(p, calibracion["final"])
                    u, ue = cal["umbral"], cal["umbral_evidencia"]
                if u is None:
                    raise RuntimeError("No hay datos suficientes para calibrar los umbrales.")
                t0 = time.perf_counter()
                q = resolver_query(c["consulta"], retrieve_fn=lambda t, kk: indice.buscar(t, kk, MODO[sistema_base]),
                                   normalizar_fn=normalizar_fn, umbral=u, k=k, umbral_evidencia=ue,
                                   forzar_por_sigla=forzar)
                registrar(clave, c, p, q["fragments"], time.perf_counter() - t0,
                          {"tool": int(q["tool_invoked"]), "query_final": q["query_final"],
                           "tool_reason": q["tool_reason"], "umbral": u, "umbral_evidencia": ue})

    return detalle, resumir(detalle, k), calibracion


def resumir(detalle: list[dict], k: int) -> list[dict]:
    et = etiquetas_resumen(k)

    def prom(xs, campo):
        vals = [x[campo] for x in xs if x.get(campo) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    filas = []
    for clave in dict.fromkeys(d["clave"] for d in detalle):
        dentro = [d for d in detalle if d["clave"] == clave and not d["fuera_de_corpus"]]
        fuera = [d for d in detalle if d["clave"] == clave and d["fuera_de_corpus"]]
        todas = dentro + fuera
        fila = {"Sistema": NOMBRE[clave], et["consultas"]: len(dentro)}
        for campo in ("hit1", "hitk", "rr", "recall", "precision", "hit1_sec", "hitk_sec", "rr_sec", "precision_sec"):
            fila[et[campo]] = prom(dentro, campo)
        fila[et["rechazo_fuera"]] = prom(fuera, "rechazo")
        fila[et["rechazo_erroneo"]] = prom(dentro, "rechazo")
        fila[et["tool"]] = prom(todas, "tool")
        fila[et["ms"]] = round(1000 * sum(d["seg"] for d in todas) / max(1, len(todas)), 1)
        fila["Clave"] = clave
        filas.append(fila)
    base = next((f for f in filas if f["Clave"] == "ingenuo"), None)
    for f in filas:
        for campo, delta in (("rr", "delta_rr"), ("rr_sec", "delta_rr_sec")):
            a, b = (base or {}).get(et[campo]), f[et[campo]]
            f[et[delta]] = round(b - a, 3) if a is not None and b is not None else None
    return filas


def deltas_contrato(resumen: list[dict], k: int, naive: str = "ingenuo",
                    advanced: str = "intencion") -> list[dict]:
    """Formato Delta del proyecto: [{metric_name, naive, advanced, delta}]."""
    fila = {f["Clave"]: f for f in resumen}
    if naive not in fila or advanced not in fila:
        return []
    et = etiquetas_resumen(k)
    salida = []
    for campo in ("hit1", "hitk", "rr", "precision", "hit1_sec", "rr_sec", "precision_sec", "ms"):
        a, b = fila[naive][et[campo]], fila[advanced][et[campo]]
        salida.append({"metric_name": et[campo], "naive": a, "advanced": b,
                       "delta": None if a is None or b is None else round(b - a, 3)})
    return salida


def detalle_legible(detalle: list[dict]) -> list[dict]:
    filas = []
    for d in detalle:
        d = dict(d, tipo=TIPOS_CONSULTA.get(d["tipo"], d["tipo"]))
        filas.append({ETIQUETAS_DETALLE.get(c, c): v for c, v in d.items()})
    return filas


def guardar(salida: Path, detalle: list[dict], resumen: list[dict], calibracion: dict, k: int = 5,
            indice: IndiceRAG | None = None, advanced: str = "intencion") -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, filas in (("deltas_s08.csv", resumen), ("detalle_por_consulta.csv", detalle_legible(detalle))):
        columnas = list(dict.fromkeys(c for f in filas for c in f))
        with open(salida / nombre, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columnas)
            w.writeheader()
            w.writerows(filas)
    archivos = {"umbrales.json": calibracion,
                "delta_naive_vs_advanced.json": deltas_contrato(resumen, k, advanced=advanced)}
    if indice is not None:
        archivos["indice_info.json"] = info_indice(indice)
    for nombre, contenido in archivos.items():
        with open(salida / nombre, "w", encoding="utf-8") as fh:
            json.dump(contenido, fh, indent=2, ensure_ascii=False)


def correr_desde_config(cfg: dict, indice: IndiceRAG | None = None, normalizar_fn=None,
                        guardar_resultados: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Ejecuta el experimento con los parámetros del YAML. Si no se pasa un normalizador, usa el
    del YAML: el mapa simulado del corpus de prueba o la herramienta de normalización real."""
    from config_retrieval import indice_desde_config, normalizador_desde_config, rutas

    r = rutas(cfg)
    indice = indice or indice_desde_config(cfg)
    normalizar_fn = normalizar_fn or normalizador_desde_config(cfg)
    o, cal, k = cfg["orquestacion"], cfg["calibracion"], cfg["retrieval"]["k"]
    detalle, resumen, calibracion = correr_experimento(
        indice, cargar_consultas(r["consultas"]), k, normalizar_fn, o["sistema_base"], cal["pliegues"],
        o["umbral"], o["umbral_evidencia"], semilla=cal["semilla"], tolerancia_fuera=cal["tolerancia_fuera_de_corpus"])
    if guardar_resultados:
        guardar(Path(r["salida"]), detalle, resumen, calibracion, k, indice, cfg["experimento"]["sistema_advanced"])
    return detalle, resumen, calibracion


def main() -> None:
    from config_retrieval import cargar_config, rutas

    p = argparse.ArgumentParser(description="Experimento de retrieval (S08)")
    p.add_argument("--config", default=None, help="ruta del YAML (por defecto config_retrieval.yaml)")
    p.add_argument("--corpus", choices=["mock", "real"], default=None, help="reemplaza el corpus del YAML")
    a = p.parse_args()

    cfg = cargar_config(a.config)
    if a.corpus:
        cfg["corpus"] = a.corpus
    _, resumen, calibracion = correr_desde_config(cfg)
    for f in resumen:
        print(" | ".join("-" if v is None else str(v) for v in f.values()))
    print("Umbrales finales:", {x: calibracion["final"][x] for x in ("umbral", "umbral_evidencia")})
    print("Resultados en", rutas(cfg)["salida"])


if __name__ == "__main__":
    main()
