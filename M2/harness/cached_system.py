import json
from pathlib import Path

from harness import Sistema


def sistema_desde_cache(cache_path: Path, eval_set: list[dict]) -> Sistema:
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        raise ValueError(f"El cache debe ser un objeto JSON doc_id -> entidades: {cache_path}")

    texto_a_doc_id = {}
    for record in eval_set:
        doc_id = record["doc_id"]
        text = record["text"]
        if text in texto_a_doc_id and texto_a_doc_id[text] != doc_id:
            raise ValueError(
                "El sistema basado en texto no distingue textos duplicados: "
                f"{texto_a_doc_id[text]} y {doc_id}"
            )
        texto_a_doc_id[text] = doc_id

    eval_doc_ids = set(texto_a_doc_id.values())
    cache_doc_ids = set(cache)
    missing = sorted(eval_doc_ids - cache_doc_ids)
    unexpected = sorted(cache_doc_ids - eval_doc_ids)
    if missing or unexpected:
        raise ValueError(
            "Los doc_id de eval_set y cache no coinciden. "
            f"Sin prediccion: {missing[:5]}; fuera de eval_set: {unexpected[:5]}"
        )

    for doc_id, entities in cache.items():
        if not isinstance(entities, list) or not all(isinstance(entity, str) for entity in entities):
            raise ValueError(f"Las predicciones de {doc_id} deben ser list[str]")

    def sistema(text: str) -> set[str]:
        try:
            doc_id = texto_a_doc_id[text]
        except KeyError as exc:
            raise KeyError("El texto no pertenece al eval_set asociado al cache") from exc
        return set(cache[doc_id])

    return sistema
