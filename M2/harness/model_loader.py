"""
Seccion 3 del harness: carga del modelo Clinical BERT + adaptador LoRA,
y sanity check con una frase de prueba.
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel


def cargar_modelo(base_checkpoint: str, model_dir: Path, label_list: list[str], device: str):
    """
    Carga el modelo base, inyecta el adaptador LoRA, y carga el tokenizer
    guardado junto al adaptador. Devuelve (model, tokenizer, id2label).
    """
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    print("id2label:", id2label)

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    print("Modelo base cargado. Parametros totales:",
          sum(p.numel() for p in base_model.parameters()))

    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()  # CRITICO: desactiva dropout para resultados deterministas

    if device == "cuda":
        model = model.to("cuda")

    print("Adaptador LoRA cargado")
    print("Dispositivo:", next(model.parameters()).device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print("Tokenizer cargado. Vocab size:", tokenizer.vocab_size)

    return model, tokenizer, id2label


def sanity_check(model, tokenizer, id2label: dict):
    """
    Verificacion rapida con una frase de juguete antes de evaluar el gold
    set completo. Esperado: diabetes->B-ENFERMEDAD, mellitus->I-ENFERMEDAD,
    hipertension->B-ENFERMEDAD.
    """
    frase_prueba = "El paciente presenta diabetes mellitus tipo 2 y antecedentes de hipertension arterial."
    tokens_prueba = frase_prueba.split()

    inputs_p = tokenizer(
        tokens_prueba, is_split_into_words=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        logits_p = model(**inputs_p).logits

    pred_ids_p = logits_p.argmax(dim=-1)[0].tolist()
    word_ids_p = inputs_p.word_ids(batch_index=0)

    print("Sanity check -- predicciones por palabra:")
    seen = set()
    for idx, wid in zip(pred_ids_p, word_ids_p):
        if wid is None or wid in seen:
            continue
        print(f"  {tokens_prueba[wid]:<25} -> {id2label[idx]}")
        seen.add(wid)