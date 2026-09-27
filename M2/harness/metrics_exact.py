"""
Seccion 5 del harness: Dimension 1 -- exact-match con chunking.
"""

import json

from common import (
    normalizar_entidad,
    micro_prf1_by_doc,
)
from gold_loader import resolver_rutas


def dimension_metrica_clasica(eval_set, sistema):
    """Compara el gold con las entidades que devuelve el sistema inyectado."""
    true_by_doc = {}
    pred_by_doc = {}
    for example in eval_set:
        doc_id = example["doc_id"]
        if doc_id in true_by_doc:
            raise ValueError(f"doc_id duplicado en el gold set: {doc_id}")
        true_by_doc[doc_id] = {
            normalizar_entidad(entity) for entity in example["entities_gold"]
        }
        pred_by_doc[doc_id] = {
            normalizar_entidad(entity) for entity in sistema(example["text"])
        }

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


def run(cfg: dict, project_root, eval_set: list[dict], sistema) -> dict:
    rutas = resolver_rutas(cfg, project_root)
    print(f"Evaluando {len(eval_set)} ejemplos con el sistema recibido...")
    metrics_exact, true_by_doc, pred_by_doc = dimension_metrica_clasica(
        eval_set, sistema
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
        "predicciones_origen": str(rutas["predictions_path"]),
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