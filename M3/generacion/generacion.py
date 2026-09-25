"""
Generacion final de la respuesta RAG  (Agustin / M3).

Diseno tomado del material de la profesora:
  - S07 / S10: prompt aumentado en 4 partes
      (instruccion -> contexto con fuente -> valvula de escape -> pregunta)
    y la firma `generar(system, user) -> str`.
  - S10: la valvula de escape anti-alucinacion ("No tengo esa informacion en
    mis fuentes.") es la linea que RAGAS castiga en faithfulness si no se usa.

Se agrega el contrato de M3: la salida es un `RespuestaRAG` trazable y el
fallback es EXPLICITO (nunca silencioso), como exige la rubrica.

Nota de alineacion con M2: el acceso a Groq reutiliza el patron de
`M2/harness/metrics_judge.py` (rotacion de claves y espaciado entre llamadas),
porque el juez de M2 ya quedo en `qwen/qwen3.8-27b` sobre Groq.
"""

from __future__ import annotations

import os
import time

from contratos import Fragment, GenerateFn, RespuestaRAG


MENSAJE_SIN_EVIDENCIA = "No tengo esa información en mis fuentes."

# Copiado de S07 (RAG ingenuo) y S10 (RAG avanzado): es la valvula de escape
# del curso. Mantener el texto literal permite comparar con los labs.
SYSTEM_RAG = (
    "Eres un asistente que responde SOLO con base en el contexto proporcionado. "
    "Si la respuesta no está en el contexto, di claramente: "
    f'"{MENSAJE_SIN_EVIDENCIA}" No inventes datos. '
    "Cuando respondas, menciona la fuente del contexto que usaste. Sé breve y claro."
)


# ---------------------------------------------------------------------------
# Prompt aumentado (S07: instruccion + contexto con fuente + pregunta)
# ---------------------------------------------------------------------------
def construir_contexto(fragments: list[Fragment]) -> str:
    """Contexto con la fuente de cada fragmento, como en S07/S08."""
    partes = []
    for f in fragments:
        fuente = f.get("fuente") or f.get("doc_id") or f["chunk_id"]
        partes.append(f"[Fuente: {fuente} | {f['chunk_id']}]\n{f['texto']}")
    return "\n\n".join(partes)


def construir_prompt(fragments: list[Fragment], pregunta: str) -> str:
    contexto = construir_contexto(fragments) if fragments else "(sin contexto)"
    return f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"


def _es_sin_evidencia(answer: str) -> bool:
    return "no tengo esa informaci" in (answer or "").lower()


# ---------------------------------------------------------------------------
# Generacion
# ---------------------------------------------------------------------------
def generar_respuesta(entidad: str, fragments: list[Fragment], *, generar_fn: GenerateFn,
                      pregunta: str | None = None) -> RespuestaRAG:
    """
    Sintetiza la respuesta final a partir de la entidad resuelta y los fragmentos.

    Fallback explicito y determinista: si NO hay fragmentos no se llama al LLM;
    se devuelve la valvula de escape de S07 con `fallback_used=True`. Asi el
    sistema nunca inventa cuando no tiene evidencia (lo que faithfulness de
    RAGAS penaliza).

    `sources_used` son los chunk_id en los que se APOYA la respuesta (no lo que
    el LLM diga que cito); queda vacio si la respuesta es la valvula de escape.
    """
    if not fragments:
        return {"answer": MENSAJE_SIN_EVIDENCIA, "sources_used": [], "fallback_used": True}

    pregunta = pregunta or f"¿Qué dice la guía clínica sobre «{entidad}»?"
    answer = (generar_fn(SYSTEM_RAG, construir_prompt(fragments, pregunta)) or "").strip()
    sources = [f["chunk_id"] for f in fragments]

    if not answer:
        return {"answer": MENSAJE_SIN_EVIDENCIA, "sources_used": [], "fallback_used": True}

    fallback = _es_sin_evidencia(answer)
    # Si la respuesta es la valvula de escape, ningun fragmento la respalda:
    # `sources_used` debe quedar vacio (honestidad de la traza).
    return {"answer": answer, "sources_used": [] if fallback else sources, "fallback_used": fallback}


def contextos_para_ragas(fragments: list[Fragment]) -> list[str]:
    """
    Textos de los fragmentos recuperados, en el orden en que se aumentaron.

    Es lo que RAGAS espera en `contexts` (mide retrieval: context precision /
    recall). No confundir con `sources_used`: `contexts` es TODO lo recuperado,
    `sources_used` es aquello en lo que la respuesta se apoya.
    """
    return [f["texto"] for f in fragments]


# ---------------------------------------------------------------------------
# Generador via Groq (implementacion concreta de GenerateFn)
# ---------------------------------------------------------------------------
def recolectar_api_keys(api_key: str | None = None) -> list[str]:
    """GROQ_API_KEY, GROQ_API_KEY_2, ... (varias cuentas suman cupo diario)."""
    claves = []
    base = api_key or os.environ.get("GROQ_API_KEY", "")
    if base:
        claves.append(base)
    for sufijo in ["_2", "_3", "_4", "_5"]:
        extra = os.environ.get(f"GROQ_API_KEY{sufijo}", "")
        if extra:
            claves.append(extra)
    return list(dict.fromkeys(claves))


class GroqGenerator:
    """
    Generador via Groq. Es un `GenerateFn`: se invoca como `generar(system, user)`.

    Reutiliza el patron ya probado en `M2/harness/metrics_judge.py`:
      - rotacion de claves al agotar el cupo diario (TPD),
      - espaciado entre llamadas para no gatillar los limites por minuto.
    """

    def __init__(self, api_key: str | None = None, model: str = "qwen/qwen3.8-27b",
                 temperature: float = 0.0, max_tokens: int = 700,
                 pausa_entre_llamadas: float = 3.0, max_reintentos: int = 6):
        from groq import Groq  # import perezoso: permite usar mocks sin groq

        claves = recolectar_api_keys(api_key)
        if not claves:
            raise RuntimeError(
                "No hay GROQ_API_KEY disponible para la generacion. "
                "Definir GROQ_API_KEY (y opcional GROQ_API_KEY_2) en el entorno/.env."
            )
        self.api_keys = claves
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pausa_entre_llamadas = pausa_entre_llamadas
        self.max_reintentos = max_reintentos
        self._idx = 0
        self.client = Groq(api_key=self.api_keys[0])

    def _rotar(self) -> bool:
        if self._idx + 1 >= len(self.api_keys):
            return False
        from groq import Groq
        self._idx += 1
        self.client = Groq(api_key=self.api_keys[self._idx])
        print(f"    [generacion] cupo agotado -> rotando a la clave {self._idx + 1}/{len(self.api_keys)}")
        return True

    def __call__(self, system: str, user: str) -> str:
        for intento in range(self.max_reintentos):
            try:
                time.sleep(self.pausa_entre_llamadas)
                r = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    cupo_diario = "tokens per day" in err or "(TPD)" in err
                    if cupo_diario and self._rotar():
                        continue
                    espera = 10 * (intento + 1)
                    print(f"    [generacion 429] limite de tasa; esperando {espera}s...")
                    time.sleep(espera)
                    continue
                print(f"    [generacion error] {type(e).__name__}: {err[:200]}")
                time.sleep(2 * (intento + 1))
        raise RuntimeError("La generacion fallo tras varios reintentos.")
