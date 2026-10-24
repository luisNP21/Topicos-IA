"""Puente visión -> NER de M1/M2 (M4 · Agustín).

La visión sola no produce entidades: produce texto. Este puente reutiliza el
modelo Clinical BERT + LoRA y el chunking YA existentes en `M2/run_inference`
para escribir un cache `{doc_id: [entidades]}` en el formato que el harness de
M2 consume.

No reimplementa el NER ni el harness. Solo lo orquesta sobre las
transcripciones de la visión.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cache import guardar_cache_ner


def _agregar_paths_m2(project_root: Path) -> None:
    for ruta in (project_root / "M2", project_root / "M2" / "run_inference"):
        if str(ruta) not in sys.path:
            # append (no insert) para no tapar módulos homónimos del repo.
            sys.path.append(str(ruta))


def generar_cache_ner(
    cache_transcripciones: dict[str, dict],
    cfg_ner: dict,
    project_root: str | Path,
    output_path: str | Path,
) -> dict:
    """Transcripciones -> predicciones NER cacheadas por documento.

    `cfg_ner` es la sección `ner:` de `M4/vision/config.yaml`:
        enabled, checkpoint, adapter_path, chunking.window_words/overlap_words
    """
    project_root = Path(project_root)
    _agregar_paths_m2(project_root)

    try:
        from model_loader import cargar_modelo
        from common import (
            aggregate_entities_by_original_doc,
            bio_to_entity_set,
            chunk_words,
        )
        from inference import predict_word_level
    except Exception as exc:  # pragma: no cover - depende de torch/peft
        raise RuntimeError(
            "No se pudo importar el NER de M2. Instalar sus dependencias "
            "(`pip install -r M2/run_inference/requirements.txt`) y verificar "
            f"que el adaptador exista bajo PROJECT_ROOT: {exc}"
        ) from exc

    modelo, tokenizer, id2label = cargar_modelo(cfg_ner, project_root)
    chunking = cfg_ner.get("chunking", {})
    window_words = int(chunking.get("window_words", 277))
    overlap_words = int(chunking.get("overlap_words", 50))

    pred_doc_ids: list[str] = []
    pred_entity_sets: list[set[str]] = []

    for doc_id, trans in cache_transcripciones.items():
        texto = str(trans.get("texto") or "")
        # Sin texto legible no se inventan entidades: se cachea lista vacía.
        if not texto or trans.get("valvula"):
            pred_doc_ids.append(doc_id)
            pred_entity_sets.append(set())
            continue

        palabras = texto.split()
        chunks = chunk_words(palabras, window_words, overlap_words)
        for chunk_idx, chunk in chunks:
            tags = predict_word_level(chunk, modelo, tokenizer, id2label)
            entidades = bio_to_entity_set(chunk, tags)
            sufijo = f"_chunk{chunk_idx}" if len(chunks) > 1 else ""
            pred_doc_ids.append(f"{doc_id}{sufijo}")
            pred_entity_sets.append(entidades)

    agrupado = aggregate_entities_by_original_doc(pred_doc_ids, pred_entity_sets)
    cache = {
        doc_id: sorted(agrupado.get(doc_id, set()))
        for doc_id in cache_transcripciones
    }
    ruta = guardar_cache_ner(cache, output_path)
    return {
        "n_documentos": len(cache),
        "n_con_entidades": sum(1 for ents in cache.values() if ents),
        "output_path": str(ruta),
    }
