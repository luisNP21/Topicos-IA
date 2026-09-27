"""
cruce_harness.py
Cruza las metricas RAGAS con el F1 de extraccion de M2 

Lee resultado_dimension1.json de M2, calcula F1 por documento a partir
de true_by_doc y pred_by_doc, y lo cruza con las metricas RAGAS por
documento para diagnosticar en que etapa del pipeline falla el sistema.

"""

import json
from pathlib import Path



# Calcular F1 por documento desde true_by_doc y pred_by_doc


def _f1_por_doc(true_by_doc: dict, pred_by_doc: dict) -> dict:
    """
    El resultado_dimension1.json tiene el F1 global pero no por documento.
    Lo recalculamos aqui a partir de true_by_doc y pred_by_doc.
    """
    resultados = {}
    for doc_id, gold in true_by_doc.items():
        gold_set = set(gold)
        pred_set = set(pred_by_doc.get(doc_id, []))
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)
        resultados[doc_id] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return resultados


# Reglas de diagnostico


def diagnosticar(
    f1_extraccion: float,
    context_recall: float,
    faithfulness: float,
    umbrales: dict,
) -> str:
    """
    Misma logica de diagnosticar_debilidad() de M2/harness/scorecard.py.
    Cada metrica apunta a una etapa distinta del pipeline:

    - f1_extraccion bajo  -> M1/M2 no detecto bien las entidades
    - context_recall bajo -> corpus/retrieval (Pau/Isa) no trajo lo necesario
    - faithfulness bajo   -> generacion (Agustin) alucino cosas fuera del contexto
    """
    f1_bajo         = umbrales.get("f1_bajo", 0.4)
    f1_alto         = umbrales.get("f1_alto", 0.7)
    recall_bajo     = umbrales.get("recall_bajo", 0.4)
    faith_bajo      = umbrales.get("faithfulness_bajo", 0.4)

    if f1_extraccion < f1_bajo and context_recall >= (1 - recall_bajo) and faithfulness >= (1 - faith_bajo):
        return "problema_extraccion"        # M1/M2 no detecto bien; el resto funciona
    if f1_extraccion >= f1_alto and context_recall < recall_bajo:
        return "problema_corpus_retrieval"  # buena extraccion, pero no hay guia o no se recupero
    if faithfulness < faith_bajo:
        return "alucinacion_generacion"     # Agustin invento cosas fuera del contexto
    if f1_extraccion >= f1_alto and context_recall >= (1 - recall_bajo) and faithfulness >= (1 - faith_bajo):
        return "funcionamiento_correcto"
    return "caso_mixto"



# Cruce principal


def cruzar(
    resultado_dim1_path: str,
    metricas_ragas: dict,
    umbrales: dict,
) -> dict:
    """
    resultado_dim1_path : ruta al resultado_dimension1.json de M2
    metricas_ragas      : dict con faithfulness, context_precision,
                          context_recall, answer_relevancy (globales o por doc)
    umbrales            : umbrales de diagnostico del config.yaml

    Devuelve un dict con:
    - f1_global_m2      : F1 global de M2
    - f1_por_doc        : F1 por documento calculado desde true/pred_by_doc
    - diagnosticos      : diagnostico por documento
    - resumen           : conteo de cada tipo de diagnostico
    - tabla_markdown    : tabla lista para copiar en el informe
    """
    path = Path(resultado_dim1_path)
    if not path.exists():
        print(f"[cruce_harness] AVISO: {path} no existe. Usando datos simulados.")
        return _cruce_simulado(metricas_ragas, umbrales)

    with open(path, "r", encoding="utf-8") as f:
        dim1 = json.load(f)

    f1_global = dim1["metrics"]["f1"]
    true_by_doc = dim1.get("true_by_doc", {})
    pred_by_doc = dim1.get("pred_by_doc", {})
    f1_doc = _f1_por_doc(true_by_doc, pred_by_doc)

    # RAGAS llega como metricas globales; las aplicamos a todos los docs
    # (cuando Pau/Agustin entreguen datos por doc, se puede extender)
    ctx_recall  = metricas_ragas.get("context_recall", 0.0)
    faith       = metricas_ragas.get("faithfulness", 0.0)

    diagnosticos = {
        doc_id: diagnosticar(vals["f1"], ctx_recall, faith, umbrales)
        for doc_id, vals in f1_doc.items()
    }

    resumen = {}
    for d in diagnosticos.values():
        resumen[d] = resumen.get(d, 0) + 1

    tabla = _tabla_markdown(f1_doc, metricas_ragas, diagnosticos)

    return {
        "f1_global_m2": f1_global,
        "f1_por_doc": f1_doc,
        "diagnosticos": diagnosticos,
        "resumen": resumen,
        "tabla_markdown": tabla,
    }


def _tabla_markdown(f1_doc: dict, ragas: dict, diagnosticos: dict) -> str:
    ctx_rec  = ragas.get("context_recall", 0.0)
    faith    = ragas.get("faithfulness", 0.0)
    ctx_prec = ragas.get("context_precision", 0.0)
    a_rel    = ragas.get("answer_relevancy", 0.0)

    lineas = [
        "| doc_id | F1 extraccion (M2) | Context recall | Faithfulness | Diagnostico |",
        "|---|---|---|---|---|",
    ]
    for doc_id, vals in sorted(f1_doc.items()):
        lineas.append(
            f"| {doc_id} | {vals['f1']:.3f} | {ctx_rec:.3f} | {faith:.3f} "
            f"| {diagnosticos.get(doc_id, '?')} |"
        )
    lineas.append("")
    lineas.append(
        f"**RAGAS global:** faithfulness={faith:.3f} | context_precision={ctx_prec:.3f} | "
        f"context_recall={ctx_rec:.3f} | answer_relevancy={a_rel:.3f}"
    )
    return "\n".join(lineas)


def _cruce_simulado(metricas_ragas: dict, umbrales: dict) -> dict:
    """Fallback cuando no hay resultado_dimension1.json disponible todavia."""
    docs_simulados = {f"ex_{i}": {"f1": round(0.3 + i * 0.07, 3)} for i in range(5)}
    ctx_recall = metricas_ragas.get("context_recall", 0.5)
    faith      = metricas_ragas.get("faithfulness", 0.5)
    diagnosticos = {
        doc_id: diagnosticar(vals["f1"], ctx_recall, faith, umbrales)
        for doc_id, vals in docs_simulados.items()
    }
    resumen = {}
    for d in diagnosticos.values():
        resumen[d] = resumen.get(d, 0) + 1
    return {
        "f1_global_m2": None,
        "f1_por_doc": docs_simulados,
        "diagnosticos": diagnosticos,
        "resumen": resumen,
        "tabla_markdown": "(datos simulados -- resultado_dimension1.json no disponible)",
    }


def run(cfg: dict, project_root: str, metricas_ragas: dict) -> dict:
    """Punto de entrada estandar. Recibe las metricas RAGAS ya calculadas."""
    cruce_cfg = cfg.get("cruce_harness", {})
    rel_path  = cruce_cfg.get(
        "resultado_dimension1_path",
        "M2/ejecucion/outputs_gold/resultado_dimension1.json",
    )
    umbrales  = {
        "f1_bajo":           cruce_cfg.get("umbral_f1_bajo", 0.4),
        "f1_alto":           cruce_cfg.get("umbral_f1_alto", 0.7),
        "recall_bajo":       cruce_cfg.get("umbral_recall_bajo", 0.4),
        "faithfulness_bajo": cruce_cfg.get("umbral_faithfulness_bajo", 0.4),
    }
    ruta_absoluta = str(Path(project_root) / rel_path) if project_root else rel_path
    return cruzar(ruta_absoluta, metricas_ragas, umbrales)

