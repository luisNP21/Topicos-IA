"""
Orquestacion determinista de la tool de normalizacion  (Agustin / M3).

La rubrica pide "criterio explicito de cuando el sistema invoca la tool y que
hace si falla". Eso se resuelve con reglas medibles en Python, NO con la
decision libre de un LLM: la S10 lo respalda ("no todo necesita un agente") y
asi la trazabilidad es total — se puede reproducir por que se invoco la tool.

Encaja ademas con el extractor de M1/M2: un encoder NO razona, solo clasifica
tokens; por eso la decision de invocar la tool vive en esta capa, entre la
salida del encoder y las tools.

Entradas inyectadas (desarrollo en paralelo):
  retrieve_fn   -> pieza de Pau (retrieval avanzado)
  normalizar_fn -> tool de Luis (SNOMED CT / UMLS)

Regla (criterio explicito, trazable en `tool_reason`):
  1. Se busca con la entidad cruda tal como la extrajo el encoder.
  2. Si hay fragmentos y el score del top-1 >= umbral -> NO se invoca la tool.
  3. Si no -> se invoca `normalizar_entidad`:
       - con match  -> se reintenta el retrieval con la entidad normalizada;
       - sin match  -> se mantiene la entidad cruda y se marca el fallo.
  4. Nunca falla en silencio: siempre devuelve `query_final` y `tool_reason`.
"""

from __future__ import annotations

from contratos import Fragment, NormalizationResult, NormalizeFn, QueryResuelta, RetrieveFn


def _retrieve_seguro(retrieve_fn: RetrieveFn, consulta: str, k: int) -> list[Fragment]:
    try:
        return list(retrieve_fn(consulta, k) or [])
    except Exception as e:  # el retrieval es una dependencia externa: no romper el pipeline
        print(f"    [orquestacion] fallo el retrieval ('{consulta}'): {e}")
        return []


def _normalizar_seguro(normalizar_fn: NormalizeFn, entidad: str) -> NormalizationResult:
    try:
        r = normalizar_fn(entidad)
        if not isinstance(r, dict):
            raise TypeError("normalizar_entidad no devolvio un dict")
        return r
    except Exception as e:
        print(f"    [orquestacion] fallo la tool de normalizacion ('{entidad}'): {e}")
        return {"entidad_original": entidad, "entidad_normalizada": entidad,
                "source_terminology": None, "normalization_failed": True}


def _score_top(fragments: list[Fragment]) -> float | None:
    if not fragments:
        return None
    s = fragments[0].get("score")
    return float(s) if s is not None else None


def resolver_query(entidad: str, *, retrieve_fn: RetrieveFn, normalizar_fn: NormalizeFn,
                   umbral: float, k: int) -> QueryResuelta:
    """
    Devuelve la query final, si se invoco la tool y por que, MAS los fragmentos
    ya recuperados (evita un segundo retrieval aguas abajo).

    `umbral` depende de la escala de `Fragment.score` (ver contratos.py): la
    orquestacion solo asume "mayor = mejor", declarado por Pau con `score_tipo`.
    """
    # 1) intento con la entidad cruda (salida del encoder de M1/M2)
    fragments = _retrieve_seguro(retrieve_fn, entidad, k)
    score_top = _score_top(fragments)

    # 2) hay evidencia suficiente -> la tool no aporta, no se invoca
    if fragments and score_top is not None and score_top >= umbral:
        return {
            "query_final": entidad, "tool_invoked": False, "tool_reason": "",
            "fragments": fragments,
        }

    # 3) score bajo o sin resultados -> se invoca la tool (criterio explicito)
    motivo = (f"score_top={score_top:.3f} < umbral={umbral}" if score_top is not None
              else "sin resultados para la entidad cruda")
    norm = _normalizar_seguro(normalizar_fn, entidad)

    # 3a) la tool no encontro match -> se mantiene la entidad cruda (fallo explicito)
    if norm.get("normalization_failed") or not norm.get("entidad_normalizada"):
        return {
            "query_final": entidad, "tool_invoked": True,
            "tool_reason": f"{motivo}; tool invocada SIN match -> se mantiene la entidad cruda",
            "fragments": fragments,
        }

    # 3b) la tool encontro match -> se reintenta el retrieval con la entidad normalizada
    fragments_norm = _retrieve_seguro(retrieve_fn, norm["entidad_normalizada"], k)
    if fragments_norm:
        return {
            "query_final": norm["entidad_normalizada"], "tool_invoked": True,
            "tool_reason": f"{motivo}; normalizada vía {norm.get('source_terminology')}",
            "fragments": fragments_norm,
        }

    # 3c) se normalizo pero el retrieval normalizado no devolvio nada -> crudo
    return {
        "query_final": entidad, "tool_invoked": True,
        "tool_reason": (f"{motivo}; normalizada pero el retrieval con la entidad "
                        f"normalizada no devolvió resultados -> se mantiene la cruda"),
        "fragments": fragments,
    }
