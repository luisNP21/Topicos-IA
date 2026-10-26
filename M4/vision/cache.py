"""Cache de la pieza de visión (M4 · Agustín).

Dos artefactos, con propósitos distintos:

1. `transcripciones_<backend>.json`: fuente de verdad de la visión
   (`doc_id -> Transcripcion`). Lo consume el análisis de errores de
   transcripción (CER/WER, dueño: Luis) y sirve para reproducir.
2. `vision_<backend>.json` en `M2/predictions/`: cache de entidades NER en el
   MISMO formato que ya lee el harness de M2 (`doc_id -> list[str]`). Este es
   el contrato con la Tarea 4 de Isabella; no se toca el harness desde aquí.

Los caches son locales y están en `.gitignore`: no se versionan.
"""

from __future__ import annotations

import json
from pathlib import Path

from contratos import Transcripcion


def guardar_json(obj, path: str | Path) -> Path:
    ruta = Path(path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return ruta


def cargar_json(path: str | Path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_cache_transcripciones(
    transcripciones: list[Transcripcion], path: str | Path
) -> Path:
    """Escribe `{doc_id: Transcripcion}` y valida que no haya ids repetidos."""
    cache: dict[str, dict] = {}
    for t in transcripciones:
        doc_id = str(t.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("Cada transcripción necesita un doc_id para el cache")
        if doc_id in cache:
            raise ValueError(f"doc_id repetido en el cache de transcripciones: {doc_id}")
        cache[doc_id] = dict(t)
    return guardar_json(cache, path)


def cargar_cache_transcripciones(path: str | Path) -> dict[str, dict]:
    cache = cargar_json(path)
    if not isinstance(cache, dict):
        raise ValueError(f"El cache de transcripciones debe ser un objeto JSON: {path}")
    return cache


def validar_cache_ner(cache: dict) -> None:
    """Valida el contrato que espera `M2/harness/cached_system.py`."""
    if not isinstance(cache, dict):
        raise ValueError("El cache NER debe ser un objeto JSON doc_id -> list[str]")
    for doc_id, entidades in cache.items():
        if not isinstance(doc_id, str):
            raise ValueError(f"doc_id no es string: {doc_id!r}")
        if not isinstance(entidades, list) or not all(
            isinstance(e, str) for e in entidades
        ):
            raise ValueError(
                f"Las predicciones de {doc_id} deben ser list[str], "
                f"vino {type(entidades).__name__}"
            )


def guardar_cache_ner(cache: dict, path: str | Path) -> Path:
    validar_cache_ner(cache)
    return guardar_json(
        {doc_id: sorted(set(entidades)) for doc_id, entidades in cache.items()},
        path,
    )


def leer_cache_ner(path: str | Path) -> dict:
    cache = cargar_json(path)
    validar_cache_ner(cache)
    return cache


def cache_desde_transcripciones(
    cache_transcripciones: dict[str, dict],
    entidades_por_doc: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Combina transcripciones y entidades conservando exactamente los doc_id.

    Falla si falta un doc_id: el harness de M2 exige que cache y eval_set
    coincidan, y un fallo silencioso aquí se vería como una caída de F1.
    """
    faltantes = sorted(set(cache_transcripciones) - set(entidades_por_doc))
    if faltantes:
        raise ValueError(
            "Faltan entidades NER para estos doc_id del cache de visión: "
            f"{faltantes[:10]}"
        )
    sobrantes = sorted(set(entidades_por_doc) - set(cache_transcripciones))
    if sobrantes:
        raise ValueError(
            "Hay entidades NER para doc_id que no están en la visión: "
            f"{sobrantes[:10]}"
        )
    return {doc_id: sorted(set(entidades_por_doc[doc_id])) for doc_id in cache_transcripciones}
