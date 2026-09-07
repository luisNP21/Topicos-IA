"""
Seccion 7 del harness: LLM-as-judge con mitigacion de sesgos de
posicion, longitud, y documentacion de auto-preferencia.
"""

import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from gold_loader import resolver_rutas


RUBRICA_SYSTEM = (
    "Eres un médico especialista en terminología médica y anotación de entidades clínicas. "
    "Tu tarea es evaluar la calidad de las entidades clínicas predichas por un sistema de NER, "
    "comparando sus predicciones con una lista gold. "
    "Evalúa SOLO el contenido. No sabes qué modelo generó las predicciones."
)

RUBRICA_TEMPLATE = """
# Evaluación de NER clínico — RUBRICA

## Referencia (gold standard)
Enfermedades correctas: {gold_str}

## Predicción del sistema
Enfermedades detectadas: {pred_str}

## Instrucciones
Evalúa en cada dimensión del 1 al 5:

1. **Completitud** (¿capturó la mayoría de las enfermedades gold?):
   5=todas/casi todas | 3=la mitad | 1=prácticamente nada

2. **Exactitud de boundary** (¿los nombres coinciden con el gold?):
   5=coincidencia exacta o diferencia mínima | 3=algunos coinciden, otros truncados | 1=no se parecen

3. **Relevancia clínica** (¿las predicciones son términos de enfermedades válidos?):
   5=todas válidas | 3=mezcla | 1=mayoría inválidas

4. **Ausencia de ruido** (¿evitó marcar términos que NO son enfermedades?):
   5=sin FP notables | 3=algunos FP | 1=demasiados FP

## Respuesta
Responde ÚNICAMENTE en JSON válido con el siguiente formato:
```json
{{
  "completitud": <1-5>,
  "exactitud_boundary": <1-5>,
  "relevancia_clinica": <1-5>,
  "ausencia_ruido": <1-5>,
  "justificacion": "<una oración breve>"
}}
```
"""


def build_judge_prompt(gold: list, pred: list, order: str = "normal") -> str:
    gold_str = ", ".join(gold) if gold else "(ninguna)"
    pred_str = ", ".join(pred) if pred else "(ninguna)"
    if order == "inverted":
        return RUBRICA_TEMPLATE.format(gold_str=pred_str, pred_str=gold_str)
    return RUBRICA_TEMPLATE.format(gold_str=gold_str, pred_str=pred_str)


def parse_judge_response(text: str) -> dict:
    for pattern in [r"```(?:json)?\s*({[\s\S]*?})\s*```", r"({[\s\S]*?})"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    result = {}
    for key in ["completitud", "exactitud_boundary", "relevancia_clinica", "ausencia_ruido"]:
        val_m = re.search(rf'"{key}"\s*:\s*([1-5])', text, re.IGNORECASE)
        result[key] = int(val_m.group(1)) if val_m else None
    just_m = re.search(r'"justificacion"\s*:\s*"([^"\n]+)', text, re.IGNORECASE)
    result["justificacion"] = just_m.group(1) if just_m else (text[-200:].strip() if text else "Sin texto")
    return result


def compute_score(d: dict) -> float | None:
    vals = [d.get(k) for k in ["completitud", "exactitud_boundary", "relevancia_clinica", "ausencia_ruido"]
            if d.get(k) is not None]
    return float(np.mean(vals)) if vals else None


class JudgeClient:
    """Encapsula el cliente Groq y el estado de deteccion de JSON mode -- sin variables globales."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int,
                 fallar_si_no_disponible: bool = True):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self._test_ping(model):
            raise RuntimeError(
                f"El modelo juez '{model}' no esta disponible en Groq (probablemente deprecado).\n"
                f"Revisar https://console.groq.com/docs/deprecations y actualizar el config.\n"
                f"No se aplica fallback automatico para preservar reproducibilidad entre ejecuciones."
            )
        print(f"[OK] Modelo juez '{model}' verificado.")

        self.supports_json_mode = self._detectar_json_mode()

    def _test_ping(self, model_name: str) -> bool:
        try:
            self.client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": "hola"}], max_tokens=10,
            )
            return True
        except Exception:
            return False

    def _detectar_json_mode(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Responde solo JSON."},
                    {"role": "user", "content": 'Genera {"ping": "pong"}'},
                ],
                max_tokens=25,
                response_format={"type": "json_object"},
            )
            print("  Soporte de JSON mode nativo: SI")
            return True
        except Exception:
            print("  Soporte de JSON mode nativo: NO (se usara parseo robusto por regex)")
            return False

    def call(self, gold: list, pred: list, order: str = "normal", retry: int = 5, sleep_s: float = 1.0) -> dict:
        prompt = build_judge_prompt(gold, pred, order=order)
        use_json_mode = self.supports_json_mode

        for attempt in range(retry):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": RUBRICA_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                time.sleep(sleep_s)
                content = response.choices[0].message.content
                parsed = parse_judge_response(content)

                if any(parsed.get(k) is not None for k in
                       ["completitud", "exactitud_boundary", "relevancia_clinica", "ausencia_ruido"]):
                    return parsed

                if attempt == 0 and use_json_mode:
                    use_json_mode = False
                    continue
                return parsed

            except Exception as e:
                err_str = str(e)
                if "response_format" in err_str or "json_object" in err_str:
                    use_json_mode = False
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                wait_time = (5 if is_rate_limit else 2) * (attempt + 1)
                print(f"    [Reintento {attempt + 1}/{retry}] {e} -> esperando {wait_time}s...")
                time.sleep(wait_time)

        return {"completitud": None, "exactitud_boundary": None, "relevancia_clinica": None,
                "ausencia_ruido": None, "justificacion": "JUDGE_CALL_FAILED"}


def cargar_rich_examples(dim1_path) -> list[dict]:
    """Reutiliza las predicciones de la Dimension 1 -- mismo gold set que exact-match y semantica."""
    with open(dim1_path, "r", encoding="utf-8") as f:
        dim1 = json.load(f)

    true_by_doc = dim1["true_by_doc"]
    pred_by_doc = dim1["pred_by_doc"]

    rich_examples = []
    for doc_id in sorted(true_by_doc):
        gold = sorted(true_by_doc[doc_id])
        pred = sorted(pred_by_doc.get(doc_id, []))
        rich_examples.append({"doc_id": doc_id, "n_gold": len(gold), "gold": gold, "pred": pred})

    print(f"Ejemplos cargados desde Dimension 1: {len(rich_examples)} (mismo gold set, sin asimetria)")
    return rich_examples


def medir_sesgo_posicion(rich_examples: list[dict], judge: JudgeClient, checkpoint_path=None) -> tuple[list, "pd.Series"]:
    print(f"Corriendo juez (normal + invertido) sobre {len(rich_examples)} ejemplos...")

    # Si ya existe un checkpoint parcial de una corrida anterior, retomar desde ahi
    # en vez de volver a pagar las llamadas ya hechas.
    position_results = []
    ya_procesados = set()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            position_results = json.load(f)["position_results"]
        ya_procesados = {r["doc_id"] for r in position_results}
        print(f"Checkpoint encontrado: {len(position_results)} ejemplos ya procesados, se retoma desde ahi.")

    for i, doc in enumerate(rich_examples):
        if doc["doc_id"] in ya_procesados:
            continue

        print(f"  [{i + 1}/{len(rich_examples)}] {doc['doc_id']}")
        result_normal = judge.call(doc["gold"], doc["pred"], order="normal")
        score_normal = compute_score(result_normal)
        result_inverted = judge.call(doc["gold"], doc["pred"], order="inverted")
        score_inverted = compute_score(result_inverted)

        delta = abs(score_normal - score_inverted) if (score_normal is not None and score_inverted is not None) else None
        valid_scores = [s for s in [score_normal, score_inverted] if s is not None]
        score_mitigado = float(np.mean(valid_scores)) if valid_scores else None

        position_results.append({
            "doc_id": doc["doc_id"], "n_gold": doc["n_gold"], "n_pred": len(doc["pred"]),
            "gold": doc["gold"], "pred": doc["pred"],
            "score_normal": score_normal, "score_inverted": score_inverted,
            "delta_posicion": delta, "score_mitigado": score_mitigado,
            "result_normal": result_normal,
        })

        # Escritura incremental -- si algo falla en el siguiente ejemplo,
        # todo lo hecho hasta aca ya esta a salvo en disco.
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"position_results": position_results}, f, indent=2, ensure_ascii=False)

    df_pos = pd.DataFrame(position_results)
    deltas = df_pos["delta_posicion"].dropna()
    print(f"Sesgo de posicion -- delta medio: {deltas.mean():.3f}, max: {deltas.max():.3f}")

    return position_results, deltas


def medir_sesgo_longitud(pares_longitud: list[dict], judge: JudgeClient) -> pd.DataFrame:
    print(f"Corriendo juez sobre {len(pares_longitud)} pares de longitud...")
    length_results = []

    for pair in pares_longitud:
        r_cor = judge.call(pair["gold"], pair["pred_correcta"], order="normal")
        r_inc = judge.call(pair["gold"], pair["pred_incorrecta"], order="normal")
        sc, si = compute_score(r_cor), compute_score(r_inc)
        length_results.append({
            "tipo": pair["tipo"], "descripcion": pair["descripcion"],
            "n_correcta": len(pair["pred_correcta"]), "n_incorrecta": len(pair["pred_incorrecta"]),
            "score_correcta": sc, "score_incorrecta": si,
            "juez_premia_calidad": (sc > si) if (sc is not None and si is not None) else None,
        })

    df_len = pd.DataFrame(length_results)
    n_ok = df_len["juez_premia_calidad"].sum()
    n_total = df_len["juez_premia_calidad"].notna().sum()
    print(f"Sesgo de longitud -- el juez premio calidad en {n_ok}/{n_total} pares")
    return df_len, int(n_ok), int(n_total)


def verificar_autopreferencia(judge_model: str) -> dict:
    sample_prompt = build_judge_prompt(["neumonía", "sepsis"], ["neumonía"])
    forbidden = ["roberta", "bert", "mt5", "llama", "gpt", "gemini", "plantl", "biomedical", "clinical"]
    found = [n for n in forbidden if n.lower() in sample_prompt.lower()]

    return {
        "output_anonimizado": len(found) == 0,
        "formato_estandarizado": True,
        "familia_cruzada": True,
        "test_cross_family_disponible": False,
        "nota": "No se puede cuantificar sin un segundo juez de otra familia distinta",
        "terminos_encontrados": found,
    }


def compute_exact_f1_doc(gold: list, pred: list) -> dict:
    g, p = set(gold), set(pred)
    tp, fp, fn = len(g & p), len(p - g), len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def classify_pattern(row: dict) -> str:
    s, f = row["score_juez"], row["f1_exacto"]
    if s is None or f is None:
        return "? — datos faltantes"
    if f < 0.1 and s >= 3.5:
        return "A — boundary error (comprensión OK, span malo)"
    if f > 0.3 and s <= 2.5:
        return "B — string OK, calidad clínica baja"
    if f < 0.1 and s <= 2.0:
        return "C — fallo completo"
    if f >= 0.7 and s >= 4.0:
        return "D — referencia positiva"
    return "E — caso mixto"


def run(cfg: dict, project_root) -> dict:
    load_dotenv()
    rutas = resolver_rutas(cfg, project_root)
    judge_cfg = cfg["dimension3_llm_judge"]

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no definida en .env")

    judge = JudgeClient(
        api_key=api_key, model=judge_cfg["judge_model"],
        temperature=judge_cfg["judge_temperature"], max_tokens=judge_cfg["judge_max_tokens"],
    )

    rich_examples = cargar_rich_examples(rutas["dim1_path"])

    checkpoint_path = rutas["output_dir"] / "checkpoint_sesgo_posicion.json"
    position_results, deltas = medir_sesgo_posicion(rich_examples, judge, checkpoint_path=checkpoint_path)

    scorecard_rows = []
    for row in position_results:
        exact = compute_exact_f1_doc(row["gold"], row["pred"])
        scorecard_rows.append({
            "doc_id": row["doc_id"], "n_gold_entities": row["n_gold"], "n_pred_entities": row["n_pred"],
            "score_juez": row["score_mitigado"], "delta_posicion": row["delta_posicion"],
            "f1_exacto": exact["f1"], "gold": row["gold"], "pred": row["pred"],
        })
    df_sc = pd.DataFrame(scorecard_rows)
    df_sc["patron"] = df_sc.apply(classify_pattern, axis=1)

    from pathlib import Path as _Path
    _HARNESS_DIR = _Path(__file__).resolve().parent  # carpeta donde vive metrics_judge.py

    pares_longitud_path = _HARNESS_DIR / judge_cfg["pares_longitud_path"]

    with open(pares_longitud_path, "r", encoding="utf-8") as f:
        pares_longitud = json.load(f)
    df_len, n_ok, n_total = medir_sesgo_longitud(pares_longitud, judge)

    autopreferencia = verificar_autopreferencia(judge_cfg["judge_model"])

    resultado_dimension3 = {
        "dimension": "llm_as_judge",
        "rol": "agustin",
        "modelo_evaluado": cfg["modelo"]["base_checkpoint"],
        "modelo_juez": judge_cfg["judge_model"],
        "judge_provider": judge_cfg["judge_provider"],
        "predicciones_origen": str(rutas["dim1_path"]),
        "n_ejemplos_evaluados": len(rich_examples),
        "seed": cfg["proyecto"]["seed_global"],
        "metrics": {
            "score_juez_mean": float(df_sc["score_juez"].mean()),
            "score_juez_std": float(df_sc["score_juez"].std()),
            "f1_exacto_mean": float(df_sc["f1_exacto"].mean()),
        },
        "sesgo_posicion": {
            "delta_mean": float(deltas.mean()), "delta_max": float(deltas.max()),
            "delta_median": float(deltas.median()), "pct_delta_mayor_a_1": float((deltas > 1).mean()),
            "mitigacion": "promedio de score_normal y score_inverted",
        },
        "sesgo_longitud": {
            "pares_evaluados": n_total, "pares_calidad_gana": n_ok, "pct_calidad_gana": n_ok / n_total,
            "mitigacion": "pares de validacion con asimetria de longitud controlada",
        },
        "sesgo_autopreferencia": autopreferencia,
        "distribucion_patrones": df_sc["patron"].value_counts().to_dict(),
        "resultados_por_doc": {
            row["doc_id"]: {
                "gold": sorted(row["gold"]), "pred": sorted(row["pred"]),
                "score_juez": row["score_juez"], "delta_posicion": row["delta_posicion"],
                "f1_exacto": row["f1_exacto"], "patron": row["patron"],
            }
            for row in df_sc.to_dict(orient="records")
        },
    }

    with open(rutas["dim3_path"], "w", encoding="utf-8") as f:
        json.dump(resultado_dimension3, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {rutas['dim3_path']}")
    return resultado_dimension3