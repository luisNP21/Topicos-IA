"""
Funciones puras de la Seccion 4, reutilizadas de M1/NB2.
No dependen del modelo ni del tokenizer -- por eso viven separadas
de inference.py, y son las que van a compartir metrics_exact.py,
metrics_semantic.py y metrics_judge.py sin duplicacion.
"""

import re
import string


def normalizar_entidad(e: str) -> str:
    """Quita puntuacion de borde que es artefacto de tokenizar con .split()."""
    return e.lower().strip().strip(string.punctuation + " ")


def strip_chunk_suffix(doc_id: str) -> str:
    return re.sub(r"_chunk\d+$", "", doc_id)


def bio_to_entity_set(tokens: list[str], tags: list[str]) -> set[str]:
    entities, current = [], []
    for tok, tag in zip(tokens, tags):
        if tag == "B-ENFERMEDAD":
            if current:
                entities.append(" ".join(current))
            current = [tok]
        elif tag == "I-ENFERMEDAD" and current:
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
            current = []
    if current:
        entities.append(" ".join(current))
    return {normalizar_entidad(e) for e in entities}


def aggregate_entities_by_original_doc(doc_ids: list[str], entity_sets: list[set]) -> dict:
    grouped = {}
    for doc_id, ents in zip(doc_ids, entity_sets):
        orig_id = strip_chunk_suffix(doc_id)
        grouped.setdefault(orig_id, set()).update(ents)
    return grouped


def micro_prf1_by_doc(true_by_doc: dict, pred_by_doc: dict) -> dict:
    tp = fp = fn = 0
    all_doc_ids = set(true_by_doc) | set(pred_by_doc)
    for doc_id in all_doc_ids:
        true_ents = true_by_doc.get(doc_id, set())
        pred_ents = pred_by_doc.get(doc_id, set())
        tp += len(true_ents & pred_ents)
        fp += len(pred_ents - true_ents)
        fn += len(true_ents - pred_ents)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def chunk_words(words: list[str], window_words: int, overlap_words: int) -> list[tuple]:
    """
    Parte una lista de palabras en ventanas solapadas.
    Devuelve lista de (chunk_idx, sublist_de_palabras).
    """
    if len(words) <= window_words:
        return [(0, words)]

    chunks = []
    step = window_words - overlap_words
    start = 0
    idx = 0
    while start < len(words):
        chunk = words[start: start + window_words]
        chunks.append((idx, chunk))
        if start + window_words >= len(words):
            break
        start += step
        idx += 1
    return chunks