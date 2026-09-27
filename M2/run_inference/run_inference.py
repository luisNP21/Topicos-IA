import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HARNESS_DIR = ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import yaml
from dotenv import load_dotenv

from common import aggregate_entities_by_original_doc, chunk_words, log
from gold_loader import cargar_gold_set
from model_loader import cargar_modelo, sanity_check
from inference import run_inference


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "PROJECT_ROOT no esta definido. Crear un .env con:\n"
            "  PROJECT_ROOT=/ruta/al/proyecto/en/Drive"
        )
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def run(cfg: dict, project_root: Path) -> dict:
    model_cfg = cfg["modelo"]
    model, tokenizer, id2label = cargar_modelo(model_cfg, project_root)
    sanity_check(model, tokenizer, id2label)

    eval_path = project_root / cfg["eval_set_path"]
    eval_set = cargar_gold_set(eval_path)

    output_path = project_root / cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def chunk_fn(words):
        return chunk_words(words, cfg["chunking"]["window_words"], cfg["chunking"]["overlap_words"])

    inference_records = []
    for rec in eval_set:
        doc_id = rec["doc_id"]
        text = rec["text"]
        words = text.split()
        chunks = chunk_fn(words)
        collected = []
        for chunk_idx, chunk_words_list in chunks:
            import torch
            from transformers import AutoTokenizer
            inputs = tokenizer(
                chunk_words_list,
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
            from common import bio_to_entity_set
            entity_set = bio_to_entity_set(chunk_words_list, word_preds)
            collected.append((f"{doc_id}_chunk{chunk_idx}" if len(chunks) > 1 else doc_id, entity_set))
        inference_records.extend(collected)

    pred_doc_ids = [doc_id for doc_id, _ in inference_records]
    pred_entity_sets = [ents for _, ents in inference_records]
    grouped = aggregate_entities_by_original_doc(pred_doc_ids, pred_entity_sets)
    serialized = {doc_id: sorted(ents) for doc_id, ents in grouped.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    n_docs = len(serialized)
    log(f"run_inference completado: {n_docs} documentos cacheados en {output_path}")
    return {"sistema": cfg["sistema"], "n_documentos": n_docs, "output_path": str(output_path)}


if __name__ == "__main__":
    with open(Path(__file__).with_name("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project_root = resolver_project_root()
    run(cfg, project_root)
