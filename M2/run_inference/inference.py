import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import aggregate_entities_by_original_doc, bio_to_entity_set, chunk_words, log


def predict_word_level(tokens: list[str], model, tokenizer, id2label: dict) -> list[str]:
    import torch

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
    word_ids = inputs.word_ids(batch_index=0)

    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1)[0].tolist()

    word_preds, seen = [], set()
    for idx, wid in zip(pred_ids, word_ids):
        if wid is None or wid in seen:
            continue
        word_preds.append(id2label[idx])
        seen.add(wid)
    return word_preds


def predict_entities_chunked(
    text: str, window_words: int, overlap_words: int, model, tokenizer, id2label: dict
) -> list[tuple]:
    words = text.split()
    chunks = chunk_words(words, window_words, overlap_words)
    resultados = []

    for chunk_idx, chunk_words_list in chunks:
        pred_tags = predict_word_level(chunk_words_list, model, tokenizer, id2label)
        entity_set = bio_to_entity_set(chunk_words_list, pred_tags)
        suffix = f"_chunk{chunk_idx}" if len(chunks) > 1 else ""
        resultados.append((suffix, entity_set))

    return resultados


def run_inference(eval_set: list[dict], modelo, chunk_fn, output_path: Path) -> None:
    """
    Corre modelo sobre cada chunk de cada documento de eval_set,
    agrega por doc_id original y escribe JSON {doc_id: [entidades]}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_doc_ids, pred_entity_sets = [], []

    model, tokenizer, id2label = modelo
    for rec in eval_set:
        doc_id = rec["doc_id"]
        text = rec["text"]
        words = text.split()
        chunks = chunk_fn(words)

        for chunk_idx, chunk_words_list in chunks:
            pred_tags = predict_word_level(chunk_words_list, model, tokenizer, id2label)
            entity_set = bio_to_entity_set(chunk_words_list, pred_tags)
            pred_doc_ids.append(f"{doc_id}_chunk{chunk_idx}" if len(chunks) > 1 else doc_id)
            pred_entity_sets.append(entity_set)

    pred_by_doc = aggregate_entities_by_original_doc(pred_doc_ids, pred_entity_sets)
    result = {doc_id: sorted(entities) for doc_id, entities in pred_by_doc.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"Inferencia guardada en {output_path} para {len(result)} documentos")
