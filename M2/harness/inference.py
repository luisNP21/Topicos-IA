"""
Funciones de la Seccion 4 que requieren el modelo/tokenizer cargados.
Se pasan explicitamente como argumentos (en vez de depender de variables
globales, como en el notebook) para que estos modulos sean testeables
de forma aislada.
"""

import torch

from common import bio_to_entity_set, chunk_words


def predict_word_level(tokens: list[str], model, tokenizer, id2label: dict) -> list[str]:
    """
    Predice tag por subtoken y lo colapsa a nivel de palabra.
    Se usa por chunk, no por documento completo.
    """
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
    """
    Chunkea el texto si excede la ventana, predice por chunk, y devuelve
    una lista de (chunk_suffix, entity_set). chunk_suffix sigue el mismo
    esquema que NB2: "_chunk0", "_chunk1", ... para que
    aggregate_entities_by_original_doc los junte correctamente.
    """
    words = text.split()
    chunks = chunk_words(words, window_words, overlap_words)

    resultados = []
    for chunk_idx, chunk_words_list in chunks:
        pred_tags = predict_word_level(chunk_words_list, model, tokenizer, id2label)
        entity_set = bio_to_entity_set(chunk_words_list, pred_tags)
        suffix = f"_chunk{chunk_idx}" if len(chunks) > 1 else ""
        resultados.append((suffix, entity_set))
    return resultados