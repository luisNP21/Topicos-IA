"""
evaluacion_ragas.py
Evaluacion RAGAS del sistema RAG de M3 -- Luis

Calcula faithfulness, context_precision, context_recall y answer_relevancy
sobre un eval_set de casos (question, contexts, answer, ground_truth).

Dos modos:
- mock: calculo manual con embeddings (paraphrase-multilingual-MiniLM-L12-v2),
        mismo enfoque que el Lab C de la Sesion 10 de la profe. Funciona sin
        GROQ_API_KEY y sin depender de datos reales de Pau/Agustin.
- real: usa la libreria ragas con el LLM de Groq como juez.

El eval_set simulado integrado permite correr el modulo de punta a punta
antes de que Pau y Agustin entreguen sus datos.
"""

import os
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ------------------------------------------------------------------
# Modelo de embeddings compartido (mismo que M2 Dim1b)
# ------------------------------------------------------------------
_EMB_MODEL = None


def _get_emb_model(model_name: str):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer(model_name)
    return _EMB_MODEL


def _sim(a: str, b: str, model) -> float:
    """Similitud coseno entre dos textos."""
    ea, eb = model.encode([a, b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))


# ------------------------------------------------------------------
# Eval set simulado para smoke-test sin datos reales
# ------------------------------------------------------------------
EVAL_SET_MOCK = [
    {
        "question": "diabetes mellitus tipo 2",
        "contexts": [
            "La diabetes mellitus tipo 2 es una enfermedad crónica caracterizada por "
            "resistencia a la insulina. Su manejo incluye metformina como primera línea.",
            "El control glucémico en diabetes tipo 2 reduce el riesgo de complicaciones "
            "microvasculares como retinopatía y nefropatía.",
        ],
        "answer": "La diabetes mellitus tipo 2 es una enfermedad crónica que requiere "
                  "control glucémico y puede tratarse con metformina.",
        "ground_truth": "Enfermedad crónica con resistencia a insulina, tratada con metformina.",
    },
    {
        "question": "hipertensión arterial",
        "contexts": [
            "La hipertensión arterial se define como presión sistólica >= 140 mmHg. "
            "Los inhibidores de ECA son de primera línea en hipertensos con diabetes.",
        ],
        "answer": "La hipertensión arterial es presión alta. Se trata con medicamentos.",
        "ground_truth": "Presión sistólica >= 140 mmHg. Inhibidores ECA primera línea con diabetes.",
    },
    {
        "question": "insuficiencia renal aguda",
        "contexts": [
            "La guía de oncología recomienda radioterapia fraccionada para tumores cerebrales.",
        ],
        "answer": "La insuficiencia renal aguda requiere hemodiálisis urgente en casos severos.",
        "ground_truth": "Deterioro brusco de la función renal, reversible en muchos casos.",
    },
]


# ------------------------------------------------------------------
# Calculo manual (modo mock) -- mismo enfoque del Lab C de la profe
# ------------------------------------------------------------------

def _si(texto: str) -> int:
    """Interpreta una respuesta sí/no del juez local."""
    return 1 if re.search(r"\bsí\b|\bsi\b|\byes\b|\b1\b", texto.lower()) else 0


def _faithfulness_mock(caso: dict, model) -> float:
    """
    ¿Las afirmaciones de la respuesta estan respaldadas por el contexto?
    Usa similitud de embeddings: si sim(afirmacion, contexto) >= 0.5 -> respaldada.
    """
    contexto = " ".join(caso["contexts"])
    afirmaciones = [s.strip() for s in re.split(r"[.\n]", caso["answer"]) if len(s.strip()) > 10]
    if not afirmaciones:
        return 1.0
    ok = sum(1 for a in afirmaciones if _sim(a, contexto, model) >= 0.5)
    return ok / len(afirmaciones)


def _context_precision_mock(caso: dict, model) -> float:
    """¿Cuantos de los chunks recuperados son relevantes para la pregunta?"""
    relevantes = sum(
        1 for ch in caso["contexts"] if _sim(caso["question"], ch, model) >= 0.4
    )
    return relevantes / max(1, len(caso["contexts"]))


def _context_recall_mock(caso: dict, model) -> float:
    """¿El contexto contiene lo necesario para llegar a la referencia?"""
    contexto = " ".join(caso["contexts"])
    return min(1.0, _sim(caso["ground_truth"], contexto, model) + 0.1)


def _answer_relevancy_mock(caso: dict, model) -> float:
    """¿La respuesta contesta lo que se pregunto?"""
    return _sim(caso["question"], caso["answer"], model)


def _calcular_mock(eval_set: list[dict], embedding_model: str) -> dict:
    model = _get_emb_model(embedding_model)
    metricas = {
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
        "answer_relevancy": [],
    }
    for caso in eval_set:
        metricas["faithfulness"].append(_faithfulness_mock(caso, model))
        metricas["context_precision"].append(_context_precision_mock(caso, model))
        metricas["context_recall"].append(_context_recall_mock(caso, model))
        metricas["answer_relevancy"].append(_answer_relevancy_mock(caso, model))

    return {k: float(np.mean(v)) for k, v in metricas.items()}


# ------------------------------------------------------------------
# Calculo real con la libreria ragas (modo real)
# ------------------------------------------------------------------

def _calcular_real(eval_set: list[dict], llm_model: str) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            try:
                from google.colab import userdata
                groq_key = userdata.get("GROQ_API_KEY")
            except Exception:
                pass
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY no encontrada en .env ni en Colab Secrets. "
                "Usa modo='mock' para correr sin API key."
            )

        llm = LangchainLLMWrapper(ChatGroq(model=llm_model, api_key=groq_key, temperature=0))

        ds = Dataset.from_dict(
            {
                "question":     [e["question"] for e in eval_set],
                "answer":       [e["answer"] for e in eval_set],
                "contexts":     [e["contexts"] for e in eval_set],
                "ground_truth": [e["ground_truth"] for e in eval_set],
            }
        )
        resultado = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
        )
        return {
            "faithfulness":       float(resultado["faithfulness"]),
            "context_precision":  float(resultado["context_precision"]),
            "context_recall":     float(resultado["context_recall"]),
            "answer_relevancy":   float(resultado["answer_relevancy"]),
        }

    except ImportError as e:
        raise ImportError(
            f"Faltan dependencias para modo real: {e}. "
            "Instala con: pip install ragas langchain-groq"
        )


# ------------------------------------------------------------------
# Punto de entrada estandar del proyecto
# ------------------------------------------------------------------

def run(cfg: dict, project_root) -> dict:
    """
    Mismo contrato run(cfg, project_root) -> dict que M2/harness y M3/tools.

    cfg debe contener la clave 'ragas' con:
        modo            : "mock" | "real"
        embedding_model : str  (usado en mock)
        llm_model       : str  (usado en real, via Groq)
    Y opcionalmente 'eval_set' si se quiere pasar uno externo.
    """
    ragas_cfg = cfg.get("ragas", {})
    modo = ragas_cfg.get("modo", "mock")
    embedding_model = ragas_cfg.get(
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    llm_model = ragas_cfg.get("llm_model", "openai/gpt-oss-120b")

    # Usar eval_set externo (de Pau/Agustin) o el simulado integrado
    eval_set = cfg.get("eval_set_externo") or EVAL_SET_MOCK

    print(f"[ragas] modo={modo} | n_casos={len(eval_set)}")

    if modo == "mock":
        metricas = _calcular_mock(eval_set, embedding_model)
    else:
        metricas = _calcular_real(eval_set, llm_model)

    print(f"[ragas] faithfulness      : {metricas['faithfulness']:.4f}")
    print(f"[ragas] context_precision : {metricas['context_precision']:.4f}")
    print(f"[ragas] context_recall    : {metricas['context_recall']:.4f}")
    print(f"[ragas] answer_relevancy  : {metricas['answer_relevancy']:.4f}")

    return {
        "modo": modo,
        "n_casos": len(eval_set),
        "metricas": metricas,
    }


if __name__ == "__main__":
    resultado = run({"ragas": {"modo": "mock"}}, project_root="")
    print(resultado)

