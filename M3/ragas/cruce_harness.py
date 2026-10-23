"""
cruce_harness.py
Presenta las metricas RAGAS y el F1 de extraccion de M2 

Lee resultado_dimension1.json de M2, calcula F1 por documento a partir
de true_by_doc y pred_by_doc. Presenta por separado los agregados de RAGAS:
no hay correspondencia verificable entre las muestras de ambas corridas.

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
    f1_bajo         = umbrales.get("umbral_f1_bajo", umbrales.get("f1_bajo", 0.4))
    f1_alto         = umbrales.get("umbral_f1_alto", umbrales.get("f1_alto", 0.7))
    recall_bajo     = umbrales.get("umbral_recall_bajo", umbrales.get("recall_bajo", 0.4))
    faith_bajo      = umbrales.get("umbral_faithfulness_bajo", umbrales.get("faithfulness_bajo", 0.4))

    if f1_extraccion < f1_bajo and context_recall >= (1 - recall_bajo) and faithfulness >= (1 - faith_bajo):
        return "problema_extraccion"        # M1/M2 no detecto bien; el resto funciona
    if f1_extraccion >= f1_bajo and context_recall < recall_bajo:
        return "problema_corpus_retrieval"  # extraccion aceptable/alta, pero no hay guia o no se recupero
    if faithfulness < faith_bajo:
        return "alucinacion_generacion"     # generacion invento cosas fuera del contexto
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
                          context_recall, answer_relevancy (globales; pueden ser None)
    umbrales            : umbrales de diagnostico del config.yaml

    Devuelve un dict con:
    - f1_global_m2      : F1 global de M2
    - f1_por_doc        : F1 por documento calculado desde true/pred_by_doc
    - diagnosticos      : vacío; no se infieren diagnósticos por documento
    - resumen           : vacío; no hay casos pareados
    - alcance           : agregados separados
    - tabla_markdown    : tabla lista para copiar en el informe
    """
    path = Path(resultado_dim1_path)
    if not path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        candidatos = [
            repo_root / "M2/outputs/resultado_dimension1.json",
            repo_root / "M2/ejecucion/outputs_gold/resultado_dimension1.json",
        ]
        for cand in candidatos:
            if cand.exists():
                path = cand
                break
    if not path.exists():
        return {"f1_global_m2": None, "f1_por_doc": {}, "diagnosticos": {}, "resumen": {},
                "alcance": "sin_resultados_m2", "n_documentos_m2": 0,
                "tabla_markdown": "No hay resultados de M2 disponibles; no se inventan documentos."}

    with open(path, "r", encoding="utf-8") as f:
        dim1 = json.load(f)

    f1_global = dim1["metrics"]["f1"]
    true_by_doc = dim1.get("true_by_doc", {})
    pred_by_doc = dim1.get("pred_by_doc", {})
    f1_doc = _f1_por_doc(true_by_doc, pred_by_doc)

    # Las reglas diagnosticar() requieren medidas del mismo caso. Aquí no se invocan.
    tabla = _tabla_markdown(f1_doc, metricas_ragas)
    return {"f1_global_m2": f1_global, "f1_por_doc": f1_doc,
            "diagnosticos": {}, "resumen": {}, "tabla_markdown": tabla,
            "alcance": "agregados_no_pareados", "n_documentos_m2": len(f1_doc),
            "fuente_m2": str(path)}


def _tabla_markdown(f1_doc: dict, ragas: dict) -> str:
    lineas = ["### M2: extracción por documento", "| doc_id | F1 extracción |", "|---|---|"]
    for doc_id, vals in sorted(f1_doc.items()):
        lineas.append(f"| {doc_id} | {vals['f1']:.3f} |")
    lineas.extend(["", "### M3: métricas agregadas de la corrida de evaluación",
                   "| Métrica | Valor |", "|---|---|"])
    for nombre in ("faithfulness", "context_precision", "context_recall", "answer_relevancy"):
        valor = ragas.get(nombre)
        texto = "N/D" if valor is None else f"{valor:.3f}"
        lineas.append(f"| {nombre} | {texto} |")
    lineas.extend(["", "**Sin diagnóstico por documento:** no se ha establecido correspondencia "
                   "entre las muestras de M2 y M3. Un promedio de M3 no es la evaluación de cada documento de M2."])
    return "\n".join(lineas)


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
    ruta_absoluta = rel_path
    if project_root:
        nueva_salida = Path(project_root) / "M2/outputs/resultado_dimension1.json"
        salida_config = Path(project_root) / rel_path
        if nueva_salida.exists():
            ruta_absoluta = str(nueva_salida)
        else:
            ruta_absoluta = str(salida_config)
    return cruzar(ruta_absoluta, metricas_ragas, umbrales)

