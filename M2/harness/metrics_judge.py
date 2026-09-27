"""
Seccion 7 del harness: LLM-as-judge (M2).

Diseno alineado con lo ensenado en S06 (harness de 3 dimensiones). Corrige los
tres apartamientos que se identificaron en la entrega anterior:

  1. JUEZ POINTWISE + DIMENSION DE DOMINIO (S06, Lab A y "Dimension 3")
     El juez puntua 1-5 con rubrica anclada Y ademas emite un veredicto binario
     de dominio ("cumple_criterio": si/no). Para eso SI lee el campo `criterio`
     del gold set, que antes el harness ignoraba. Esa es la dimension "aciertos
     de dominio" que exige el curso (antes inexistente).

  2. SESGO DE POSICION (S06, Lab B)
     NO se intercambian gold y prediccion dentro de la rubrica: eso medía la
     asimetria de la rubrica, no el orden. Se comparan DOS respuestas anonimas
     A/B y se intercambian (protocolo pairwise robusto de S06). Solo se declara
     ganador si el veredicto coincide en ambos ordenes; si el juez se contradice
     -> empate = evidencia de sesgo de posicion. Ese test es un DIAGNOSTICO y NO
     se promedia con el score final (antes contaminaba el resultado).

  3. SESGO DE LONGITUD (S06, Lab B)
     Los pares de control usan una respuesta larga PARCIALMENTE correcta (algunos
     aciertos + ruido/FP), no ruido sin relacion con el gold.

  4. AUTO-PREFERENCIA: se documenta y se mantiene el output anonimizado.

Referencias: SI4006 - S06 - Lab A (juez), Lab B (sesgos) y Dimension 3 (dominio).
"""

import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from gold_loader import resolver_rutas, cargar_gold_set


# --------------------------------------------------------------------------
# Rubrica pointwise (S06: anclas 1-5 explicitas + veredicto binario de dominio)
# --------------------------------------------------------------------------

RUBRICA_SYSTEM = (
    "Eres un medico especialista en terminologia clinica y anotacion de entidades. "
    "Evalúas la calidad de una lista de entidades ENFERMEDAD detectadas por un "
    "sistema de NER sobre texto clinico en espanol. Evalua SOLO el contenido, con "
    "criterio estricto y objetivo. No sabes que modelo genero la prediccion."
)

RUBRICA_TEMPLATE = """# Evaluacion de NER clinico — RUBRICA (S06)

## Criterio de evaluacion de ESTE caso
{criterio}

## Referencia (gold standard)
Enfermedades correctas: {gold_str}

## Prediccion del sistema
Enfermedades detectadas: {pred_str}

## Instrucciones
Evalua la prediccion del 1 al 5 en cada dimension, con estas anclas explicitas:

1. **Completitud** (¿capturo la mayoria de las enfermedades gold?):
   5 = todas o casi todas | 3 = aproximadamente la mitad | 1 = practicamente ninguna
2. **Exactitud de boundary** (¿los nombres coinciden con el gold?):
   5 = coincidencia exacta o diferencia minima | 3 = algunos coinciden, otros truncados | 1 = no se parecen
3. **Relevancia clinica** (¿las predicciones son terminos de enfermedad validos?):
   5 = todas validas | 3 = mezcla | 1 = mayoria invalidas
4. **Ausencia de ruido** (¿evito marcar terminos que NO son enfermedades?):
   5 = sin FP notables | 3 = algunos FP | 1 = demasiados FP

Ademas responde la pregunta binaria de dominio, que es la que define el
"acierto de dominio" del harness (S06, Dimension 3):

- **cumple_criterio**: responde "si" cuando la prediccion es UTIL para el
  criterio de ESTE caso; "no" cuando no lo es. Regla objetiva para decidir:
    * "si" si la mayoria de las enfermedades gold estan capturadas (completitud
      >= 4) Y las que se detectan son clinicamente validas (relevancia >= 4) Y no
      hay ruido grave (ausencia_ruido >= 3). Los errores leves de boundary/formato
      NO impiden el "si": una lista util con un limite de span imperfecto sigue
      sirviendo a un clinico.
    * "no" si falta una parte sustancial de las enfermedades, o si domina el
      ruido (falsos positivos), o si hay entidades inventadas.
  Referencia cuantitativa (usala para no dudar): si la prediccion captura
  aproximadamente el 75% o mas de las enfermedades gold y no tiene ruido grave,
  la respuesta es "si" aunque falte alguna entidad. Ejemplo: gold de 4 entidades
  y la prediccion acierta 3 -> es "si" (cumple). gold de 4 y acierta 2 -> es "no".
  No respondas "si" por cortesia, pero tampoco exijas perfeccion: el "si" es
  "esto sirve para el criterio del caso", no "esto es identico al gold".

## Respuesta
Responde UNICAMENTE con JSON valido, sin texto adicional:
```json
{{
  "completitud": <1-5>,
  "exactitud_boundary": <1-5>,
  "relevancia_clinica": <1-5>,
  "ausencia_ruido": <1-5>,
  "cumple_criterio": "si" | "no",
  "justificacion": "<una oracion breve>"
}}
```"""


# --------------------------------------------------------------------------
# Rubrica pairwise (S06, Lab B: "¿cual respuesta es mejor? A o B")
# --------------------------------------------------------------------------

PAIRWISE_SYSTEM = (
    "Eres un evaluador estricto y objetivo de anotacion de entidades clinicas (NER). "
    "Comparas dos listas anonimas de entidades ENFERMEDAD y eliges la mejor. "
    "No conoces el origen de ninguna de las dos."
)

PAIRWISE_TEMPLATE = """# Comparacion ciega de dos anotaciones de NER clinico

## Criterio del caso
{criterio}

## Candidato A
{resp_a}

## Candidato B
{resp_b}

## Pregunta
¿Cual lista de entidades ENFERMEDAD es mejor: la A o la B?
Responde SOLO con la letra A o B (una sola letra, sin explicacion)."""


DIM_KEYS = ["completitud", "exactitud_boundary", "relevancia_clinica", "ausencia_ruido"]
CRITERIO_DEFECTO = (
    "La respuesta debe identificar todas las menciones de enfermedad presentes "
    "en el texto clinico, sin incluir terminos que no sean enfermedades ni "
    "menciones alucinadas."
)


def build_judge_prompt(gold: list, pred: list, criterio: str = "") -> str:
    """Prompt pointwise. El `order` desaparecio: ya NO se intercambian gold/pred."""
    gold_str = ", ".join(gold) if gold else "(ninguna)"
    pred_str = ", ".join(pred) if pred else "(ninguna)"
    criterio_str = (criterio or "").strip() or CRITERIO_DEFECTO
    return RUBRICA_TEMPLATE.format(criterio=criterio_str, gold_str=gold_str, pred_str=pred_str)


def build_pairwise_prompt(criterio: str, resp_a: list, resp_b: list) -> str:
    """Prompt pairwise de S06: dos respuestas anonimas A/B."""
    a = ", ".join(resp_a) if resp_a else "(ninguna)"
    b = ", ".join(resp_b) if resp_b else "(ninguna)"
    criterio_str = (criterio or "").strip() or CRITERIO_DEFECTO
    return PAIRWISE_TEMPLATE.format(criterio=criterio_str, resp_a=a, resp_b=b)


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
    for key in DIM_KEYS:
        val_m = re.search(rf'"{key}"\s*:\s*([1-5])', text, re.IGNORECASE)
        result[key] = int(val_m.group(1)) if val_m else None
    cc_m = re.search(r'"cumple_criterio"\s*:\s*"?\s*(si|sí|no|true|false)', text, re.IGNORECASE)
    result["cumple_criterio"] = cc_m.group(1) if cc_m else None
    just_m = re.search(r'"justificacion"\s*:\s*"([^"\n]+)', text, re.IGNORECASE)
    result["justificacion"] = just_m.group(1) if just_m else (text[-200:].strip() if text else "Sin texto")
    return result


def parse_pairwise_response(text: str) -> str:
    """Letra A/B del veredicto. Prioriza una letra independiente para no
    confundir la 'A' de 'LA'/'MEJOR' con un veredicto. '?' si no hay letra."""
    upper = text.upper()
    standalone = re.search(r"(?<![A-Z])([AB])(?![A-Z])", upper)
    if standalone:
        return standalone.group(1)
    m = re.search(r"[AB]", upper)
    return m.group() if m else "?"


def normalizar_cumple_criterio(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("si", "sí", "s", "true", "yes", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def compute_score(d: dict) -> float | None:
    vals = [d.get(k) for k in DIM_KEYS if d.get(k) is not None]
    return float(np.mean(vals)) if vals else None


class JudgeClient:
    """Encapsula el cliente Groq y el estado de deteccion de JSON mode.

    Soporta ROTACION DE CLAVES: Groq aplica un cupo diario por organizacion
    (200.000 tokens/dia en el tier gratuito). Cuando una clave agota su cupo se
    pasa automaticamente a la siguiente, de modo que varias cuentas gratuitas
    suman su cupo para completar la corrida en un solo dia.
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int,
                 fallar_si_no_disponible: bool = True, pausa_entre_llamadas: float = 1.0,
                 max_reintentos_rate_limit: int = 8, api_keys: list | None = None):
        # api_keys permite pasar varias; 'api_key' se mantiene por compatibilidad.
        claves = [k for k in (api_keys or [api_key]) if k]
        # Se eliminan duplicados conservando el orden.
        self.api_keys = list(dict.fromkeys(claves))
        if not self.api_keys:
            raise RuntimeError("No se recibio ninguna GROQ_API_KEY para el juez.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Groq aplica limites por minuto (TPM/OTPM/RPM). Sin pausa, una rafaga de
        # llamadas los agota y genera 429 en cadena. Esta pausa espacia las
        # peticiones para mantenerse por debajo del cupo del tier gratuito.
        self.pausa_entre_llamadas = pausa_entre_llamadas
        self.max_reintentos_rate_limit = max_reintentos_rate_limit

        self._key_idx = 0
        self.client = Groq(api_key=self.api_keys[0])
        print(f"[juez] {len(self.api_keys)} clave(s) de Groq cargada(s).")

        if not self._test_ping(model):
            raise RuntimeError(
                f"El modelo juez '{model}' no esta disponible en Groq (probablemente deprecado).\n"
                f"Revisar https://console.groq.com/docs/deprecations y actualizar el config.\n"
                f"No se aplica fallback automatico para preservar reproducibilidad entre ejecuciones."
            )
        print(f"[OK] Modelo juez '{model}' verificado.")

        self.supports_json_mode = self._detectar_json_mode()

    def _rotar_clave(self) -> bool:
        """Pasa a la siguiente clave con cupo. False si ya no quedan claves."""
        if self._key_idx + 1 >= len(self.api_keys):
            return False
        self._key_idx += 1
        self.client = Groq(api_key=self.api_keys[self._key_idx])
        print(f"    [claves] cupo agotado -> rotando a la clave "
              f"{self._key_idx + 1}/{len(self.api_keys)}")
        return True

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

    def _complete(self, system: str, user: str, retry: int = 5, sleep_s: float = 1.0,
                  use_json: bool = False) -> str:
        """Llamada base con reintentos y espaciado. Devuelve el texto crudo del juez."""
        use_json_mode = use_json and self.supports_json_mode
        rate_limit_seguidos = 0
        for attempt in range(retry):
            try:
                # Espaciado preventivo contra los limites por minuto de Groq.
                time.sleep(self.pausa_entre_llamadas)

                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                finish = getattr(response.choices[0], "finish_reason", None)

                # gpt-oss-120b (y otros modelos de razonamiento) consumen tokens
                # internos de reasoning. Si finish_reason == 'length', el presupuesto
                # de tokens se agoto antes de emitir el JSON: no sirve reintentar
                # igual, hay que subir el limite.
                if finish == "length" and not content.strip():
                    nuevo_limite = min(self.max_tokens * 2, 8000)
                    if nuevo_limite > kwargs["max_tokens"]:
                        print(f"    [Ajuste] finish_reason='length' sin contenido -> "
                              f"max_tokens {kwargs['max_tokens']} -> {nuevo_limite}")
                        self.max_tokens = nuevo_limite
                        continue

                time.sleep(sleep_s)
                return content

            except Exception as e:
                err_str = str(e)
                if "response_format" in err_str or "json_object" in err_str:
                    use_json_mode = False
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

                if is_rate_limit:
                    # Distingue cupo DIARIO agotado (TPD) de limite POR MINUTO.
                    # El diario no se resuelve esperando: se rota la clave.
                    cupo_diario = "tokens per day" in err_str or "(TPD)" in err_str
                    if cupo_diario and self._rotar_clave():
                        rate_limit_seguidos = 0
                        continue

                    rate_limit_seguidos += 1
                    # El servidor indica cuanto esperar ('try again in Xs'). Se
                    # respeta ese valor y se aumenta la pausa para las siguientes
                    # llamadas, evitando la cascada de 429.
                    espera = self.pausa_entre_llamadas
                    m = re.search(r"try again in ([0-9.]+)s", err_str)
                    if m:
                        espera = max(espera, float(m.group(1)) + 2)
                    else:
                        espera = max(espera, 10 * rate_limit_seguidos)
                    self.pausa_entre_llamadas = min(self.pausa_entre_llamadas * 1.5, 30.0)
                    print(f"    [429] limite de tasa; esperando {espera:.1f}s "
                          f"(pausa futura {self.pausa_entre_llamadas:.1f}s)...")
                    time.sleep(espera)

                    # Si el limite se repite, se amplia el numero de intentos.
                    if rate_limit_seguidos >= retry:
                        retry = min(retry + self.max_reintentos_rate_limit, 40)
                    continue

                print(f"    [Error del juez] {type(e).__name__}: {err_str[:300]}")
                wait_time = 2 * (attempt + 1)
                print(f"    [Reintento {attempt + 1}/{retry}] esperando {wait_time}s...")
                time.sleep(wait_time)
        return ""

    def call_pointwise(self, gold: list, pred: list, criterio: str = "") -> dict:
        """Juez pointwise: 4 dimensiones 1-5 + veredicto binario de dominio."""
        prompt = build_judge_prompt(gold, pred, criterio)

        # 1) Intento con JSON mode nativo.
        content = self._complete(RUBRICA_SYSTEM, prompt, use_json=True)
        parsed = parse_judge_response(content)
        if any(parsed.get(k) is not None for k in DIM_KEYS):
            return parsed

        # 2) Fallback sin JSON mode (parseo robusto por regex sobre texto libre).
        content = self._complete(RUBRICA_SYSTEM, prompt, use_json=False)
        parsed = parse_judge_response(content)
        if any(parsed.get(k) is not None for k in DIM_KEYS):
            return parsed

        return {**{k: None for k in DIM_KEYS}, "cumple_criterio": None,
                "justificacion": "JUDGE_CALL_FAILED"}

    def call_pairwise(self, criterio: str, resp_a: list, resp_b: list, retry: int = 5) -> str:
        """Comparacion ciega A/B (S06 Lab B). Devuelve 'A', 'B' o '?'."""
        prompt = build_pairwise_prompt(criterio, resp_a, resp_b)
        return parse_pairwise_response(self._complete(PAIRWISE_SYSTEM, prompt, retry=retry, use_json=False))


# --------------------------------------------------------------------------
# Carga de ejemplos (predicciones de la Dimension 1 + criterio del gold set)
# --------------------------------------------------------------------------

def cargar_rich_examples(dim1: dict, gold_examples: list[dict]) -> list[dict]:
    """
    Une las predicciones exactas con el criterio y texto del eval_set por doc_id.
    """
    true_by_doc = dim1["true_by_doc"]
    pred_by_doc = dim1["pred_by_doc"]

    gold_by_doc = {example["doc_id"]: example for example in gold_examples}
    rich_examples = []
    for doc_id in sorted(true_by_doc):
        gold = sorted(true_by_doc[doc_id])
        pred = sorted(pred_by_doc.get(doc_id, []))
        ex = gold_by_doc.get(doc_id, {})
        rich_examples.append({
            "doc_id": doc_id,
            "n_gold": len(gold),
            "gold": gold,
            "pred": pred,
            "criterio": ex.get("criterio", ""),
            "input": ex.get("text", ""),
        })

    print(f"Ejemplos cargados desde Dimension 1: {len(rich_examples)} "
          f"(con `criterio` del gold set, sin asimetria de subset)")
    return rich_examples


# --------------------------------------------------------------------------
# Evaluacion pointwise + sesgo de posicion (pairwise A/B)
# --------------------------------------------------------------------------

def evaluar_rich_examples(rich_examples: list[dict], judge: JudgeClient,
                          checkpoint_path=None) -> list[dict]:
    """
    Por cada documento:
      1. Score pointwise (score final del juez, SIN promediar con ningun test).
      2. Veredicto binario de dominio (cumple_criterio).
      3. Test de sesgo de posicion pairwise A/B con intercambio de orden.

    Escritura incremental para no perder llamadas ya hechas.
    """
    print(f"Corriendo juez pointwise + test de posicion sobre {len(rich_examples)} ejemplos...")

    resultados: list[dict] = []
    ya_procesados = set()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            # .get() tolera checkpoints del formato anterior (clave 'position_results')
            resultados = json.load(f).get("resultados", []) or []
        ya_procesados = {r["doc_id"] for r in resultados}
        print(f"Checkpoint encontrado: {len(resultados)} ejemplos ya procesados, se retoma desde ahi.")

    for i, doc in enumerate(rich_examples):
        if doc["doc_id"] in ya_procesados:
            continue

        print(f"  [{i + 1}/{len(rich_examples)}] {doc['doc_id']} | n_gold={doc['n_gold']} | "
              f"n_pred={len(doc['pred'])}")

        # 1 + 2: score final pointwise y veredicto de dominio (una sola llamada)
        result_pointwise = judge.call_pointwise(doc["gold"], doc["pred"], doc["criterio"])
        score_punto = compute_score(result_pointwise)
        cumple = normalizar_cumple_criterio(result_pointwise.get("cumple_criterio"))

        # 3: sesgo de posicion. Solo hay senal si pred != gold. Se presentan
        #    pred y gold como dos respuestas anonimas A/B y se intercambian.
        veredicto = None
        v_pred_a = v_gold_a = None
        if set(doc["pred"]) != set(doc["gold"]):
            v_pred_a = judge.call_pairwise(doc["criterio"], doc["pred"], doc["gold"])  # pred=A, gold=B
            v_gold_a = judge.call_pairwise(doc["criterio"], doc["gold"], doc["pred"])  # gold=A, pred=B
            if v_pred_a == "B" and v_gold_a == "A":
                veredicto = "gold"   # el juez prefiere consistentemente la referencia
            elif v_pred_a == "A" and v_gold_a == "B":
                veredicto = "pred"   # el juez prefiere consistentemente la prediccion
            else:
                veredicto = "empate"  # se contradijo con el orden -> sesgo de posicion

        s_punto_str = f"{score_punto:.2f}" if score_punto is not None else "FAIL"
        print(f"      score pointwise: {s_punto_str} | cumple_criterio: {cumple} | "
              f"posicion: {veredicto} (A/B={v_pred_a}/{v_gold_a})")

        resultados.append({
            "doc_id": doc["doc_id"], "n_gold": doc["n_gold"], "n_pred": len(doc["pred"]),
            "gold": doc["gold"], "pred": doc["pred"], "criterio": doc["criterio"],
            "score_juez": score_punto, "cumple_criterio": cumple,
            "veredicto_posicion": veredicto, "v_pred_primero": v_pred_a, "v_gold_primero": v_gold_a,
            "result_pointwise": result_pointwise,
        })

        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"resultados": resultados}, f, indent=2, ensure_ascii=False)

    return resultados


def resumir_sesgo_posicion(resultados: list[dict]) -> dict:
    """Agrega el test pairwise A/B. El score final NO se toca aqui."""
    veredictos = [r["veredicto_posicion"] for r in resultados]
    pares = [v for v in veredictos if v is not None]
    n_pares = len(pares)
    n_gold = sum(1 for v in pares if v == "gold")
    n_pred = sum(1 for v in pares if v == "pred")
    n_empates = sum(1 for v in pares if v == "empate")
    tasa = (n_empates / n_pares) if n_pares else 0.0

    if n_pares == 0:
        veredicto = "sin pares comparables (pred == gold en todos los documentos)"
    elif tasa <= 0.2:
        veredicto = "estable: el juez casi no se contradice con el orden"
    elif tasa <= 0.5:
        veredicto = "sesgo moderado de posicion: conviene reportar el empate como no confiable"
    else:
        veredicto = "sesgo fuerte de posicion: el veredicto depende del orden"

    print(f"Sesgo de posicion (pairwise A/B): {n_empates}/{n_pares} empates "
          f"(tasa={tasa:.3f}); gana gold={n_gold}, gana pred={n_pred}")

    return {
        "protocolo": "pairwise A/B con intercambio de orden (S06, Lab B)",
        "n_pares": n_pares,
        "n_gana_gold": n_gold,
        "n_gana_pred": n_pred,
        "n_empates": n_empates,
        "tasa_sesgo_posicion": float(tasa),
        "veredicto": veredicto,
        "mitigacion": (
            "veredicto robusto: solo se declara ganador si coincide en ambos ordenes; "
            "si no -> empate. El score final es pointwise y NO se promedia con este test."
        ),
        "nota": "pares omitidos cuando pred == gold (sin senal de orden)",
    }


# --------------------------------------------------------------------------
# Sesgo de longitud (respuesta larga PARCIALMENTE correcta)
# --------------------------------------------------------------------------

def medir_sesgo_longitud(pares_longitud: list[dict], judge: JudgeClient):
    print(f"Corriendo juez sobre {len(pares_longitud)} pares de longitud...")
    length_results = []

    for pair in pares_longitud:
        criterio = pair.get("criterio", CRITERIO_DEFECTO)
        r_cor = judge.call_pointwise(pair["gold"], pair["pred_correcta"], criterio)
        r_inc = judge.call_pointwise(pair["gold"], pair["pred_incorrecta"], criterio)
        sc, si = compute_score(r_cor), compute_score(r_inc)
        length_results.append({
            "tipo": pair["tipo"], "descripcion": pair["descripcion"],
            "n_correcta": len(pair["pred_correcta"]), "n_incorrecta": len(pair["pred_incorrecta"]),
            "score_correcta": sc, "score_incorrecta": si,
            "juez_premia_calidad": (sc > si) if (sc is not None and si is not None) else None,
        })

    df_len = pd.DataFrame(length_results)
    n_ok = int(df_len["juez_premia_calidad"].sum())
    n_total = int(df_len["juez_premia_calidad"].notna().sum())
    print(f"Sesgo de longitud -- el juez premio calidad en {n_ok}/{n_total} pares")
    return df_len, n_ok, n_total


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


# --------------------------------------------------------------------------
# Triangulacion con F1 exacto
# --------------------------------------------------------------------------

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


def run(
    cfg: dict,
    project_root,
    eval_set: list[dict] | None = None,
    dim1_result: dict | None = None,
) -> dict:
    load_dotenv()
    rutas = resolver_rutas(cfg, project_root)
    judge_cfg = cfg["dimension3_llm_judge"]

    if dim1_result is None:
        with open(rutas["dim1_path"], "r", encoding="utf-8") as f:
            dim1_result = json.load(f)
    if eval_set is None:
        eval_set = cargar_gold_set(rutas["gold_set_path"])

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no definida en .env")

    # Recolecta claves adicionales opcionales (GROQ_API_KEY_2, GROQ_API_KEY_3, ...)
    # para sumar cupo diario y completar la corrida en un solo dia.
    api_keys = [api_key]
    for sufijo in ["_2", "_3", "_4", "_5"]:
        extra = os.environ.get(f"GROQ_API_KEY{sufijo}", "")
        if extra:
            api_keys.append(extra)

    judge = JudgeClient(
        api_key=api_key, model=judge_cfg["judge_model"],
        temperature=judge_cfg["judge_temperature"], max_tokens=judge_cfg["judge_max_tokens"],
        pausa_entre_llamadas=judge_cfg.get("judge_pausa_entre_llamadas", 1.0),
        api_keys=api_keys,
    )

    rich_examples = cargar_rich_examples(dim1_result, eval_set)

    checkpoint_data = json.dumps(
        {"judge": judge_cfg, "examples": rich_examples},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    checkpoint_id = hashlib.sha256(checkpoint_data).hexdigest()[:12]
    checkpoint_path = rutas["output_dir"] / f"checkpoint_sesgo_posicion_{checkpoint_id}.json"
    resultados = evaluar_rich_examples(rich_examples, judge, checkpoint_path=checkpoint_path)
    sesgo_posicion = resumir_sesgo_posicion(resultados)

    # --- Score final del juez = pointwise (NO promedio con el test de posicion) ---
    scorecard_rows = []
    for row in resultados:
        exact = compute_exact_f1_doc(row["gold"], row["pred"])
        scorecard_rows.append({
            "doc_id": row["doc_id"], "n_gold_entities": row["n_gold"], "n_pred_entities": row["n_pred"],
            "score_juez": row["score_juez"], "cumple_criterio": row["cumple_criterio"],
            "veredicto_posicion": row["veredicto_posicion"],
            "f1_exacto": exact["f1"], "gold": row["gold"], "pred": row["pred"],
        })
    df_sc = pd.DataFrame(scorecard_rows)
    df_sc["patron"] = df_sc.apply(classify_pattern, axis=1)

    # --- Dimension de dominio (S06 Dimension 3): aciertos si/no ---
    validos = df_sc["cumple_criterio"].dropna()
    aciertos_dominio = int(validos.sum())
    total_dominio = int(validos.shape[0])
    tasa_dominio = (aciertos_dominio / total_dominio) if total_dominio else 0.0
    print(f"Dimension de dominio -- aciertos: {aciertos_dominio}/{total_dominio} ({tasa_dominio:.1%})")

    from pathlib import Path as _Path
    _HARNESS_DIR = _Path(__file__).resolve().parent
    pares_longitud_path = _HARNESS_DIR / judge_cfg["pares_longitud_path"]

    with open(pares_longitud_path, "r", encoding="utf-8") as f:
        pares_longitud = json.load(f)
    df_len, n_ok, n_total = medir_sesgo_longitud(pares_longitud, judge)

    autopreferencia = verificar_autopreferencia(judge_cfg["judge_model"])

    resultado_dimension3 = {
        "dimension": "llm_as_judge",
        "rol": "agustin",
        "modelo_evaluado": cfg.get("modelo", {}).get("base_checkpoint", "cache"),
        "modelo_juez": judge_cfg["judge_model"],
        "judge_provider": judge_cfg["judge_provider"],
        "predicciones_origen": str(rutas["dim1_path"]),
        "gold_set": str(rutas["gold_set_path"]),
        "n_ejemplos_evaluados": len(rich_examples),
        "seed": cfg["proyecto"]["seed_global"],
        "metrics": {
            "score_juez_mean": float(df_sc["score_juez"].mean()),
            "score_juez_std": float(df_sc["score_juez"].std()),
            "f1_exacto_mean": float(df_sc["f1_exacto"].mean()),
            "aciertos_dominio": aciertos_dominio,
            "total_dominio": total_dominio,
            "tasa_aciertos_dominio": float(tasa_dominio),
        },
        "dimension_dominio": {
            "definicion": "acierto = el juez responde 'si' al criterio del caso (S06, Dimension 3)",
            "aciertos": aciertos_dominio,
            "total": total_dominio,
            "tasa": float(tasa_dominio),
        },
        "sesgo_posicion": sesgo_posicion,
        "sesgo_longitud": {
            "pares_evaluados": n_total, "pares_calidad_gana": n_ok,
            "pct_calidad_gana": (n_ok / n_total) if n_total else 0.0,
            "mitigacion": "pares controlados de respuesta corta correcta vs. respuesta larga parcialmente correcta",
        },
        "sesgo_autopreferencia": autopreferencia,
        "distribucion_patrones": df_sc["patron"].value_counts().to_dict(),
        "resultados_por_doc": {
            row["doc_id"]: {
                "gold": sorted(row["gold"]), "pred": sorted(row["pred"]),
                "score_juez": row["score_juez"], "cumple_criterio": row["cumple_criterio"],
                "veredicto_posicion": row["veredicto_posicion"],
                "f1_exacto": row["f1_exacto"], "patron": row["patron"],
            }
            for row in df_sc.to_dict(orient="records")
        },
    }

    with open(rutas["dim3_path"], "w", encoding="utf-8") as f:
        json.dump(resultado_dimension3, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {rutas['dim3_path']}")
    return resultado_dimension3
