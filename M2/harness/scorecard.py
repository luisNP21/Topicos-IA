"""
Seccion 8 del harness: scorecard integrador (Isa).
Junta las 3 dimensiones en un scorecard legible con diagnostico de debilidad.
"""

import json
from collections import Counter

from gold_loader import resolver_rutas


def f1_por_doc_semantico(dim1b_data: dict) -> dict:
    detalle = dim1b_data["detalle_emparejamientos"]
    no_emp = dim1b_data["no_emparejados"]
    resultado = {}
    for doc_id in set(detalle) | set(no_emp):
        tp = len(detalle.get(doc_id, []))
        fp = len(no_emp.get(doc_id, {}).get("fp", []))
        fn = len(no_emp.get(doc_id, {}).get("fn", []))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        resultado[doc_id] = {"precision": precision, "recall": recall, "f1": f1}
    return resultado


def diagnosticar_debilidad(f1_exacto, f1_semantico, score_juez, umbrales: dict) -> str:
    if f1_exacto is None or f1_semantico is None or score_juez is None:
        return "datos incompletos"

    f_bajo, f_alto = umbrales["f1_bajo"], umbrales["f1_alto"]
    s_bajo, s_alto = umbrales["score_juez_bajo"], umbrales["score_juez_alto"]

    if f1_exacto < f_bajo and f1_semantico >= f_alto and score_juez >= s_alto:
        return "boundary/formato -- el modelo entiende, falla el limite del span"
    if f1_exacto >= f_alto and f1_semantico >= f_alto and score_juez <= s_bajo:
        return "calidad clinica -- coincide el texto pero el juez lo objeta"
    if f1_exacto < f_bajo and f1_semantico < 0.4 and score_juez <= 2.0:
        return "fallo real -- ninguna metrica reconoce acierto"
    if f1_exacto >= f_alto and f1_semantico >= f_alto and score_juez >= s_alto:
        return "funcionamiento correcto"
    return "caso mixto"


def build(cfg: dict, project_root) -> dict:
    rutas = resolver_rutas(cfg, project_root)

    for path, nombre in [(rutas["dim1_path"], "exact-match"),
                          (rutas["dim1b_path"], "similitud semantica"),
                          (rutas["dim3_path"], "LLM-as-judge")]:
        assert path.exists(), f"No se encontro el resultado de {nombre} en {path}."

    with open(rutas["dim1_path"], "r", encoding="utf-8") as f:
        dim1 = json.load(f)
    with open(rutas["dim1b_path"], "r", encoding="utf-8") as f:
        dim1b = json.load(f)
    with open(rutas["dim3_path"], "r", encoding="utf-8") as f:
        dim3 = json.load(f)

    print("Las 3 dimensiones cargadas correctamente.")
    if not (dim1["n_ejemplos_evaluados"] == dim1b["n_ejemplos_evaluados"] == dim3["n_ejemplos_evaluados"]):
        print("ADVERTENCIA: las 3 dimensiones no evaluaron el mismo numero de ejemplos.")

    f1_sem_por_doc = f1_por_doc_semantico(dim1b)
    umbrales = cfg["scorecard"]["umbrales_debilidad"]

    filas_scorecard = []
    for doc_id, info in dim3["resultados_por_doc"].items():
        f1_exacto = info.get("f1_exacto")
        f1_semantico = f1_sem_por_doc.get(doc_id, {}).get("f1")
        score_juez = info.get("score_juez")
        filas_scorecard.append({
            "doc_id": doc_id, "n_gold": len(info["gold"]),
            "f1_exacto": round(f1_exacto, 4) if f1_exacto is not None else None,
            "f1_semantico": round(f1_semantico, 4) if f1_semantico is not None else None,
            "score_juez": round(score_juez, 4) if score_juez is not None else None,
            "debilidad": diagnosticar_debilidad(f1_exacto, f1_semantico, score_juez, umbrales),
        })

    distribucion_debilidad = Counter(row["debilidad"] for row in filas_scorecard)

    scorecard = {
        "proyecto": cfg["proyecto"]["nombre"],
        "n_ejemplos": len(filas_scorecard),
        "metricas_globales": {
            "exact_match": dim1["metrics"],
            "similitud_semantica": {
                "precision": dim1b["metrics"]["precision"], "recall": dim1b["metrics"]["recall"],
                "f1": dim1b["metrics"]["f1"], "soft_f1": dim1b["metrics"]["soft_f1"],
                "umbral": dim1b["configuracion"]["umbral"],
            },
            "llm_judge": dim3["metrics"],
        },
        "mitigacion_sesgos_juez": {
            "posicion": dim3["sesgo_posicion"], "longitud": dim3["sesgo_longitud"],
            "auto_preferencia": dim3["sesgo_autopreferencia"],
        },
        "distribucion_debilidad": dict(distribucion_debilidad),
        "detalle_por_doc": filas_scorecard,
    }

    with open(rutas["scorecard_json_path"], "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)
    print(f"Scorecard (JSON) guardado en: {rutas['scorecard_json_path']}")

    _generar_markdown(scorecard, rutas["scorecard_md_path"])
    return scorecard


def _generar_markdown(scorecard: dict, path):
    me = scorecard["metricas_globales"]
    sp = scorecard["mitigacion_sesgos_juez"]["posicion"]
    sl = scorecard["mitigacion_sesgos_juez"]["longitud"]
    distribucion = scorecard["distribucion_debilidad"]
    top_debilidad = max(distribucion.items(), key=lambda x: x[1])

    lineas = [
        f"# Scorecard -- {scorecard['proyecto']}\n",
        f"**Ejemplos evaluados:** {scorecard['n_ejemplos']}\n",
        "## Metricas globales\n",
        "| Dimension | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| Exact-match | {me['exact_match']['precision']:.3f} | {me['exact_match']['recall']:.3f} | {me['exact_match']['f1']:.3f} |",
        f"| Similitud semantica | {me['similitud_semantica']['precision']:.3f} | {me['similitud_semantica']['recall']:.3f} | {me['similitud_semantica']['f1']:.3f} |",
        f"| LLM-as-judge (score 1-5) | -- | -- | {me['llm_judge']['score_juez_mean']:.3f} |",
        "\n## Mitigacion de sesgos del juez\n",
        f"- **Posicion:** delta medio = {sp['delta_mean']:.3f} (mitigado con {sp['mitigacion']})",
        f"- **Longitud:** el juez premio calidad en {sl['pares_calidad_gana']}/{sl['pares_evaluados']} pares ({sl['pct_calidad_gana']:.1%})",
        "- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)",
        "\n## Distribucion de debilidades detectadas\n",
        "| Debilidad | Documentos |", "|---|---|",
    ]
    for debilidad, n in sorted(distribucion.items(), key=lambda x: -x[1]):
        lineas.append(f"| {debilidad} | {n} |")

    lineas.append("\n## Lectura del baseline\n")
    lineas.append(
        f"La debilidad mas frecuente es **{top_debilidad[0]}** ({top_debilidad[1]}/{scorecard['n_ejemplos']} documentos). "
        f"Comparando F1 exact-match ({me['exact_match']['f1']:.3f}) vs. F1 semantico "
        f"({me['similitud_semantica']['f1']:.3f}), la brecha indica cuanto del error es de "
        f"boundary/formato vs. comprension real de la entidad clinica."
    )

    texto = "\n".join(lineas)

    with open(rutas["scorecard_json_path"], "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2, ensure_ascii=False)

    print(f"Scorecard (Markdown) guardado en: {path}")
    print(texto)