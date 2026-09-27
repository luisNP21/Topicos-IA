"""
Orquestación de la herramienta de normalización terminológica (SNOMED CT / UMLS).

La decisión es determinista y usa como señal el score del primer fragmento recuperado
(probabilidad del reranker, en [0, 1]). Dos umbrales dividen ese score en tres zonas:

    score >= umbral                      confiable: no se invoca la herramienta
    umbral_evidencia <= score < umbral   dudoso: se normaliza la entidad y se vuelve a buscar
    score < umbral_evidencia             sin evidencia: no se entregan fragmentos y la
                                         generación responde que no tiene la información

Tras normalizar se conserva el intento con mejor score, salvo cuando la entidad es una sigla
y forzar_por_sigla está activo: en ese caso se invoca la herramienta aunque el score sea alto
y se usa el término normalizado, porque ante siglas el reranker asigna scores altos a guías
que no corresponden. Los umbrales se calibran con experimento_s08.py.
"""

from __future__ import annotations

import re

from contratos import Fragment, NormalizationResult, NormalizeFn, QueryResuelta, RetrieveFn

PATRON_SIGLA = re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\-]{1,7}$")


def es_sigla(entidad: str) -> bool:
    return bool(PATRON_SIGLA.match(entidad.strip()))


def _retrieve_seguro(retrieve_fn: RetrieveFn, consulta: str, k: int) -> list[Fragment]:
    try:
        return list(retrieve_fn(consulta, k) or [])
    except Exception as e:
        print(f"    [orquestacion] falló el retrieval ('{consulta}'): {e}")
        return []


def _normalizar_seguro(normalizar_fn: NormalizeFn, entidad: str) -> NormalizationResult:
    try:
        r = normalizar_fn(entidad)
        if not isinstance(r, dict):
            raise TypeError("normalizar_entidad no devolvió un dict")
        return r
    except Exception as e:
        print(f"    [orquestacion] falló la normalización ('{entidad}'): {e}")
        return {"entidad_original": entidad, "entidad_normalizada": entidad,
                "source_terminology": None, "normalization_failed": True}


def _score_top(fragments: list[Fragment]) -> float | None:
    if not fragments:
        return None
    if fragments[0].get("score_tipo") == "rrf":
        raise ValueError("El score RRF no es comparable con un umbral fijo; use un modo con reranker.")
    s = fragments[0].get("score")
    return float(s) if s is not None else None


def _fmt(s: float | None) -> str:
    return "None" if s is None else f"{s:.3f}"


def _compuerta(fragments: list[Fragment], umbral_evidencia: float | None) -> list[Fragment]:
    if umbral_evidencia is None:
        return fragments
    return [f for f in fragments if f.get("score") is not None and float(f["score"]) >= umbral_evidencia]


def resolver_query(entidad: str, *, retrieve_fn: RetrieveFn, normalizar_fn: NormalizeFn,
                   umbral: float, k: int, umbral_evidencia: float | None = None,
                   forzar_por_sigla: bool = False) -> QueryResuelta:
    """Devuelve la consulta final, si se invocó la herramienta y por qué, y los fragmentos que
    superan la compuerta de evidencia. Con umbral_evidencia=None no se aplica la compuerta."""
    if umbral_evidencia is not None and umbral_evidencia > umbral:
        raise ValueError("umbral_evidencia debe ser menor o igual que umbral")

    fragments = _retrieve_seguro(retrieve_fn, entidad, k)
    score_crudo = _score_top(fragments)
    sigla = forzar_por_sigla and es_sigla(entidad)

    if score_crudo is not None and score_crudo >= umbral and not sigla:
        return {"query_final": entidad, "tool_invoked": False, "tool_reason": "",
                "fragments": _compuerta(fragments, umbral_evidencia)}

    if sigla:
        motivo = f"entidad con forma de sigla (score={_fmt(score_crudo)})"
    elif score_crudo is None:
        motivo = "sin resultados para la entidad original"
    else:
        motivo = f"score={_fmt(score_crudo)} < umbral={umbral:.3f}"

    norm = _normalizar_seguro(normalizar_fn, entidad)
    query_final, elegidos = entidad, fragments
    detalle = "normalización SIN match: se mantiene la entidad original"
    termino = norm.get("entidad_normalizada")
    if not norm.get("normalization_failed") and termino and termino.strip().lower() != entidad.strip().lower():
        frag_norm = _retrieve_seguro(retrieve_fn, termino, k)
        score_norm = _score_top(frag_norm)
        if score_norm is not None and (sigla or score_crudo is None or score_norm > score_crudo):
            query_final, elegidos = termino, frag_norm
            detalle = (f"normalizada vía {norm.get('source_terminology')} "
                       f"(score {_fmt(score_crudo)} -> {_fmt(score_norm)})")
        else:
            detalle = (f"normalizada vía {norm.get('source_terminology')}, sin mejora "
                       f"(score {_fmt(score_crudo)} vs {_fmt(score_norm)}): se mantiene la entidad original")

    finales = _compuerta(elegidos, umbral_evidencia)
    if elegidos and not finales:
        detalle += f"; ningún fragmento supera umbral_evidencia={umbral_evidencia:.3f}: sin evidencia"
    return {"query_final": query_final, "tool_invoked": True,
            "tool_reason": f"{motivo}; {detalle}", "fragments": finales}
