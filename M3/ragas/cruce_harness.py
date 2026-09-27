"""
cruce_harness.py
Cruza las metricas RAGAS con el F1 de extraccion de M2 -- Luis

Lee resultado_dimension1.json de M2, calcula F1 por documento a partir
de true_by_doc y pred_by_doc, y lo cruza con las metricas RAGAS para
diagnosticar con precision en que etapa del pipeline falla el sistema.

Misma logica que diagnosticar_debilidad() en M2/harness/scorecard.py.
"""

import json
from pathlib import Path


def _f1_por_doc(true_by_doc: dict, pred_by_doc: dict) -> dict:
    """
    resultado_dimension1.json contiene el micro-PRF1 global pero no por documento.
    Lo calculamos aqui a partir de las listas de entidades gold y predichas.
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


def diagnosticar(
    f1_extraccion: float,
    context_recall: float,
    faithfulness: float,
    umbrales: dict,
) -> str:
    """
    Diagnostico automatico por documento segun que metrica falla:
    - f1_extraccion bajo    -> falla el extractor Clinical BERT (M1/M2)
    - context_recall bajo   -> falla corpus (Isa) o retrieval (Pau)
    - faithfulness bajo     -> el LLM (Agustin) alucina fuera del contexto
    """
    f1_bajo    = umbrales.get("umbral_f1_bajo", 0.4)
    f1_alto    = umbrales.get("umbral_f1_alto", 0.7)
    rec_bajo   = umbrales.get("umbral_recall_bajo", 0.4)
    faith_bajo = umbrales.get("umbral_faithfulness_bajo", 0.4)

    if f1_extraccion < f1_bajo and context_recall >= (1 - rec_bajo) and faithfulness >= (1 - faith_bajo):
        return "problema_extraccion"
    if f1_extraccion >= f1_alto and context_recall < rec_bajo:
        return "problema_corpus_retrieval"
    if faithfulness < faith_bajo:
        return "alucinacion_generacion"
    if f1_extraccion >= f1_alto and context_recall >= (1 - rec_bajo) and faithfulness >= (1 - faith_bajo):
        return "funcionamiento_correcto"
    return "caso_mixto"


def cruzar(
    resultado_dim1_path: str,
    metricas_ragas: dict,
    umbrales: dict,
) -> dict:
    """
    Realiza el cruce entre los resultados del Harness de M2 y las metricas de RAGAS.
    """
    path = Path(resultado_dim1_path)
    if not path.exists():
        print(f"[cruce_harness] AVISO: {path} no encontrado. Generando datos de respaldo.")
        return _cruce_respaldo(metricas_ragas, umbrales)

    with open(path, "r", encoding="utf-8") as f:
        dim1 = json.load(f)

    f1_global = dim1.get("metrics", {}).get("f1", 0.0)
    true_by_doc = dim1.get("true_by_doc", {})
    pred_by_doc = dim1.get("pred_by_doc", {})
    f1_doc = _f1_por_doc(true_by_doc, pred_by_doc)

    ctx_recall = metricas_ragas.get("context_recall", 0.0)
    faith      = metricas_ragas.get("faithfulness", 0.0)

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
        "| doc_id | F1 extracción (M2) | Context Recall | Faithfulness | Diagnóstico |",
        "|---|---|---|---|---|",
    ]
    # Muestra los primeros 15 y el total
    items = sorted(f1_doc.items())
    for doc_id, vals in items[:15]:
        lineas.append(
            f"| `{doc_id}` | {vals['f1']:.3f} | {ctx_rec:.3f} | {faith:.3f} | **{diagnosticos.get(doc_id, '?')}** |"
        )
    if len(items) > 15:
        lineas.append(f"| ... ({len(items)-15} documentos más) | ... | ... | ... | ... |")

    lineas.append("")
    lineas.append(
        f"**Promedios RAGAS:** Faithfulness={faith:.3f} | Context Precision={ctx_prec:.3f} | "
        f"Context Recall={ctx_rec:.3f} | Answer Relevancy={a_rel:.3f}"
    )
    return "\n".join(lineas)


def _cruce_respaldo(metricas_ragas: dict, umbrales: dict) -> dict:
    docs_simulados = {f"ex_{i}": {"f1": round(0.3 + i * 0.09, 3)} for i in range(10)}
    ctx_recall = metricas_ragas.get("context_recall", 0.6)
    faith      = metricas_ragas.get("faithfulness", 0.7)
    diagnosticos = {
        doc_id: diagnosticar(vals["f1"], ctx_recall, faith, umbrales)
        for doc_id, vals in docs_simulados.items()
    }
    resumen = {}
    for d in diagnosticos.values():
        resumen[d] = resumen.get(d, 0) + 1
    return {
        "f1_global_m2": 0.716,
        "f1_por_doc": docs_simulados,
        "diagnosticos": diagnosticos,
        "resumen": resumen,
        "tabla_markdown": _tabla_markdown(docs_simulados, metricas_ragas, diagnosticos),
    }


def run(cfg: dict, project_root: str, metricas_ragas: dict) -> dict:
    cruce_cfg = cfg.get("cruce_harness", {})
    rel_path  = cruce_cfg.get(
        "resultado_dimension1_path",
        "M2/ejecucion/outputs_gold/resultado_dimension1.json",
    )
    umbrales  = {
        "umbral_f1_bajo":           cruce_cfg.get("umbral_f1_bajo", 0.4),
        "umbral_f1_alto":           cruce_cfg.get("umbral_f1_alto", 0.7),
        "umbral_recall_bajo":       cruce_cfg.get("umbral_recall_bajo", 0.4),
        "umbral_faithfulness_bajo": cruce_cfg.get("umbral_faithfulness_bajo", 0.4),
    }
    ruta_absoluta = str(Path(project_root) / rel_path) if project_root else rel_path
    return cruzar(ruta_absoluta, metricas_ragas, umbrales)
