import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer


def cargar_modelo(cfg_modelo: dict, project_root: Path | None = None):
    base_checkpoint = cfg_modelo["checkpoint"]
    adapter_path = cfg_modelo["adapter_path"]
    if project_root is not None:
        adapter_path = project_root / adapter_path

    id2label = {0: "O", 1: "B-ENFERMEDAD", 2: "I-ENFERMEDAD"}
    label2id = {v: k for k, v in id2label.items()}

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    return model, tokenizer, id2label


def sanity_check(model, tokenizer, id2label: dict):
    frase_prueba = "El paciente presenta diabetes mellitus tipo 2 y antecedentes de hipertension arterial."
    tokens_prueba = frase_prueba.split()
    inputs_p = tokenizer(tokens_prueba, is_split_into_words=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits_p = model(**inputs_p).logits

    pred_ids_p = logits_p.argmax(dim=-1)[0].tolist()
    word_ids_p = inputs_p.word_ids(batch_index=0)

    seen = set()
    for idx, wid in zip(pred_ids_p, word_ids_p):
        if wid is None or wid in seen:
            continue
        print(f"  {tokens_prueba[wid]:<25} -> {id2label[idx]}")
        seen.add(wid)
