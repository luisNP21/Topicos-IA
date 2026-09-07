"""
Seccion 5 del harness: Dimension 1 -- exact-match con chunking.
Reutiliza environment.py, gold_loader.py, model_loader.py, common.py, inference.py.
"""

import json

from common import (
    normalizar_entidad,
    bio_to_entity_set,
    aggregate_entities_by_original_doc,
    micro_prf1_by_doc,
)
from environment import preparar_entorno
from gold_loader import resolver_rutas, cargar_gold_set
from model_loader import cargar_modelo, sanity_check
from inference import predict_entities_chunked


def dimension_metrica_clasica(gold_examples, window_words, overlap_words, model, tokenizer, id2label):
    """
    Metrica clasica de exact-match, con chunking para documentos que
    exceden la ventana del tokenizer. El gold (esperado) es a nivel de
    documento completo -- no se chunkea, porque no viene con offsets por
    chunk. Solo se chunkea el INPUT para prediccion, y las entidades
    predichas de todos los chunks de un mismo documento se re-agregan
    antes de comparar contra el gold.
    """
    true_by_doc = {}
    pred_doc_ids, pred_entity_sets = [], []

    for i, ex in enumerate(gold_examples):
        doc_id = f"ex_{i}"
        true_by_doc[doc_id] = {normalizar_entidad(e) for e in ex["esperado"]}

        resultados_chunks = predict_entities_chunked(
            ex["input"], window_words, overlap_words, model, tokenizer, id2label
        )
        for suffix, entity_set in resultados_chunks:
            pred_doc_ids.append(doc_id + suffix)
            pred_entity_sets.append(entity_set)

    pred_by_doc = aggregate_entities_by_original_doc(pred_doc_ids, pred_entity_sets)
    metrics = micro_prf1_by_doc(true_by_doc, pred_by_doc)
    return metrics, true_by_doc, pred_by_doc


def detectar_boundary_errors(pred_ents: set, true_ents: set) -> list[dict]:
    """Pares donde una entidad es substring de la otra -- evidencia de la limitacion del exact-match."""
    casos = []
    for p in pred_ents:
        for t in true_ents:
            if p != t and (p in t or t in p):
                casos.append({"predicho": p, "gold": t})
    return casos


def analizar_errores(true_by_doc: dict, pred_by_doc: dict) -> tuple[dict, list]:
    errores_por_doc = {}
    for doc_id in true_by_doc:
        fn = true_by_doc[doc_id] - pred_by_doc.get(doc_id, set())
        fp = pred_by_doc.get(doc_id, set()) - true_by_doc[doc_id]
        if fn or fp:
            errores_por_doc[doc_id] = {"fn": sorted(fn), "fp": sorted(fp)}

    ejemplos_boundary = []
    for doc_id in true_by_doc:
        casos = detectar_boundary_errors(pred_by_doc.get(doc_id, set()), true_by_doc[doc_id])
        for c in casos:
            ejemplos_boundary.append({"doc_id": doc_id, **c})

    return errores_por_doc, ejemplos_boundary


def run(cfg: dict, project_root) -> dict:
    entorno = preparar_entorno(cfg)
    rutas = resolver_rutas(cfg, project_root)
    gold_examples = cargar_gold_set(rutas["gold_set_path"])

    model, tokenizer, id2label = cargar_modelo(
        cfg["modelo"]["base_checkpoint"], rutas["model_dir"],
        cfg["modelo"]["label_list"], entorno["device"],
    )
    sanity_check(model, tokenizer, id2label)

    window_words = cfg["chunking"]["window_words"]
    overlap_words = cfg["chunking"]["overlap_words"]

    print(f"Evaluando {len(gold_examples)} ejemplos del gold set (con chunking, window={window_words})...")
    metrics_exact, true_by_doc, pred_by_doc = dimension_metrica_clasica(
        gold_examples, window_words, overlap_words, model, tokenizer, id2label
    )
    print("=== Dimension 1 -- Exact-match (micro-PRF1) ===")
    print(json.dumps(metrics_exact, indent=2))

    errores_por_doc, ejemplos_boundary = analizar_errores(true_by_doc, pred_by_doc)
    print(f"Ejemplos con al menos un error: {len(errores_por_doc)} / {len(true_by_doc)}")
    print(f"Casos de boundary/truncamiento: {len(ejemplos_boundary)}")

    resultado_dimension1 = {
        "dimension": "metrica_clasica_exact_match",
        "rol": "luis",
        "modelo": cfg["modelo"]["base_checkpoint"],
        "adaptador_lora": str(rutas["model_dir"]),
        "gold_set": str(rutas["gold_set_path"]),
        "n_ejemplos_evaluados": len(true_by_doc),
        "metrics": metrics_exact,
        "true_by_doc": {k: sorted(v) for k, v in true_by_doc.items()},
        "pred_by_doc": {k: sorted(v) for k, v in pred_by_doc.items()},
        "casos_boundary": ejemplos_boundary,
    }

    with open(rutas["dim1_path"], "w", encoding="utf-8") as f:
        json.dump(resultado_dimension1, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {rutas['dim1_path']}")
    print(f"  Precision : {metrics_exact['precision']:.4f}")
    print(f"  Recall    : {metrics_exact['recall']:.4f}")
    print(f"  F1        : {metrics_exact['f1']:.4f}")

    return resultado_dimension1