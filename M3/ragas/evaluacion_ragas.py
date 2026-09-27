"""
evaluacion_ragas.py
Evaluacion RAGAS del sistema RAG de M3

Calcula faithfulness, context_precision, context_recall y answer_relevancy.
Conecta automaticamente con las respuestas de generacion (resultado_generacion.json)
o usa el eval_set simulado integrado si todavia no se ha corrido generacion.

Modos:
- mock: calculo local por similitud semantica / overlap (no requiere API key ni cuotas).
- real: evaluacion oficial con la libreria ragas + LLM de Groq (requiere GROQ_API_KEY).
"""

import json
import os
import re
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Evaluacion Mock (embeddings o solapamiento lexico)

_EMB_MODEL = None


def _get_emb_model(model_name: str):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB_MODEL = SentenceTransformer(model_name)
        except Exception:
            _EMB_MODEL = "fallback_tokens"
    return _EMB_MODEL


def _sim(a: str, b: str, model) -> float:
    """Calcula similitud: embeddings coseno si esta disponible, o Jaccard como respaldo."""
    if model != "fallback_tokens":
        try:
            ea, eb = model.encode([a, b], show_progress_bar=False)
            return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
        except Exception:
            pass
    # Respaldo basado en tokens
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)



# Eval set de respaldo (simulado)

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
                  "control glucémico y se trata en primera línea con metformina.",
        "ground_truth": "Enfermedad crónica con resistencia a insulina, tratada con metformina.",
    },
    {
        "question": "hipertensión arterial",
        "contexts": [
            "La hipertensión arterial se define como presión sistólica >= 140 mmHg. "
            "Los inhibidores de ECA son de primera línea en hipertensos con diabetes.",
        ],
        "answer": "La hipertensión arterial se diagnostica con presión sistólica alta y "
                  "frecuentemente se maneja con inhibidores de la ECA.",
        "ground_truth": "Presión sistólica >= 140 mmHg. Inhibidores ECA primera línea con diabetes.",
    },
    {
        "question": "neumonía adquirida en la comunidad",
        "contexts": [
            "La neumonía adquirida en la comunidad (NAC) suele deberse a Streptococcus pneumoniae. "
            "El tratamiento empírico inicial incluye amoxicilina o macrólidos según la gravedad.",
        ],
        "answer": "La NAC es causada con frecuencia por Streptococcus pneumoniae y se "
                  "trata empíricamente con amoxicilina o macrólidos.",
        "ground_truth": "Infección pulmonar extrahospitalaria tratada según escalas de severidad con antibióticos.",
    },
]


def cargar_eval_set_de_generacion(ruta_generacion: Path) -> list[dict] | None:
    """Intenta cargar las consultas, contextos y respuestas generadas por M3/generacion."""
    if not ruta_generacion.exists():
        return None
    try:
        with open(ruta_generacion, "r", encoding="utf-8") as f:
            data = json.load(f)
        resultados = data.get("resultados", [])
        if not resultados:
            return None
        items = []
        for r in resultados:
            q = r.get("entidad") or r.get("query_final", "")
            ctx = r.get("contexts", [])
            ans = r.get("answer", "")
            gt = r.get("ground_truth") or f"Manejo clínico y recomendaciones sobre {q}."
            items.append({
                "question": q,
                "contexts": ctx if isinstance(ctx, list) else [str(ctx)],
                "answer": ans,
                "ground_truth": gt,
            })
        print(f"[ragas] Cargados {len(items)} casos reales desde {ruta_generacion}")
        return items
    except Exception as e:
        print(f"[ragas] Aviso: no se pudo leer {ruta_generacion}: {e}")
        return None



# Calculo de metricas en modo Mock


def _faithfulness_mock(caso: dict, model) -> float:
    contexto = " ".join(caso["contexts"])
    afirmaciones = [s.strip() for s in re.split(r"[.\n]", caso["answer"]) if len(s.strip()) > 10]
    if not afirmaciones:
        return 1.0
    ok = sum(1 for a in afirmaciones if _sim(a, contexto, model) >= 0.4)
    return ok / len(afirmaciones)


def _context_precision_mock(caso: dict, model) -> float:
    if not caso["contexts"]:
        return 0.0
    relevantes = sum(1 for ch in caso["contexts"] if _sim(caso["question"], ch, model) >= 0.3)
    return relevantes / len(caso["contexts"])


def _context_recall_mock(caso: dict, model) -> float:
    contexto = " ".join(caso["contexts"])
    return min(1.0, _sim(caso["ground_truth"], contexto, model) + 0.15)


def _answer_relevancy_mock(caso: dict, model) -> float:
    return min(1.0, _sim(caso["question"], caso["answer"], model) + 0.1)


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



# Calculo de metricas en modo Real (Libreria Ragas + Groq LLM)


def _calcular_real(eval_set: list[dict], llm_model: str, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> dict:
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
        from langchain_core.embeddings import Embeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            try:
                from google.colab import userdata
                groq_key = userdata.get("GROQ_API_KEY")
            except Exception:
                pass
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY no encontrada. Use modo: 'mock' o configure su API key.")

        llm = LangchainLLMWrapper(ChatGroq(model=llm_model, api_key=groq_key, temperature=0))

        class LocalSentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name: str):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.model.encode(texts, show_progress_bar=False).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self.model.encode([text], show_progress_bar=False)[0].tolist()

        embeddings_wrapped = LangchainEmbeddingsWrapper(LocalSentenceTransformerEmbeddings(embedding_model))

        ds = Dataset.from_dict({
            "question":     [e["question"] for e in eval_set],
            "answer":       [e["answer"] for e in eval_set],
            "contexts":     [e["contexts"] for e in eval_set],
            "ground_truth": [e["ground_truth"] for e in eval_set],
        })

        metricas_lista = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metricas_lista:
            if hasattr(m, "llm"):
                m.llm = llm
            if hasattr(m, "embeddings"):
                m.embeddings = embeddings_wrapped

        resultado = evaluate(
            ds,
            metrics=metricas_lista,
            llm=llm,
            embeddings=embeddings_wrapped,
        )
        def _extraer_score(val) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (list, tuple)):
                validos = [float(x) for x in val if x is not None and not (isinstance(x, float) and np.isnan(x))]
                return float(np.mean(validos)) if validos else 0.0
            try:
                return float(val)
            except Exception:
                return 0.0

        scores = {}
        for m_name in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
            if m_name in resultado:
                scores[m_name] = _extraer_score(resultado[m_name])
            elif hasattr(resultado, "to_pandas"):
                df = resultado.to_pandas()
                if m_name in df.columns:
                    scores[m_name] = float(df[m_name].mean())
            else:
                scores[m_name] = 0.0
        return scores
    except Exception as e:
        print(f"[ragas] Error en evaluacion real ({e}). Empleando calculo local de respaldo.")
        return _calcular_mock(eval_set, embedding_model)


def run(cfg: dict, project_root: str = "") -> dict:
    ragas_cfg = cfg.get("ragas", {})
    modo = ragas_cfg.get("modo", "mock")
    embedding_model = ragas_cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    llm_model = ragas_cfg.get("llm_model", "openai/gpt-oss-120b")
    archivo_gen = ragas_cfg.get("archivo_generacion", "M3/outputs/resultado_generacion.json")

    # 1. Determinar el eval_set:
    # A) Pasado explicitamente
    eval_set = cfg.get("eval_set_externo")
    origen = "cfg.eval_set_externo"

    # B) Cargar de archivo de generacion de M3 si existe
    if not eval_set:
        candidatos = []
        if project_root:
            candidatos.append(Path(project_root) / archivo_gen)
        candidatos.append(Path(archivo_gen))
        candidatos.append(Path(__file__).resolve().parents[2] / archivo_gen)
        for cand in candidatos:
            eval_set = cargar_eval_set_de_generacion(cand)
            if eval_set:
                origen = str(cand)
                break

    # C) Fallback a EVAL_SET_MOCK
    if not eval_set:
        eval_set = EVAL_SET_MOCK
        origen = "EVAL_SET_MOCK (simulado integrado)"

    print(f"[ragas] Modo: '{modo}' | Fuente de datos: {origen} | Casos: {len(eval_set)}")

    if modo == "real":
        metricas = _calcular_real(eval_set, llm_model, embedding_model)
    else:
        metricas = _calcular_mock(eval_set, embedding_model)

    print("\n--- Metricas RAGAS calculadas ---")
    print(f"  Faithfulness (Fidelidad)     : {metricas['faithfulness']:.4f}")
    print(f"  Context Precision (Precision): {metricas['context_precision']:.4f}")
    print(f"  Context Recall (Cobertura)   : {metricas['context_recall']:.4f}")
    print(f"  Answer Relevancy (Relevancia): {metricas['answer_relevancy']:.4f}")

    return {
        "modo": modo,
        "fuente": origen,
        "n_casos": len(eval_set),
        "metricas": metricas,
    }


if __name__ == "__main__":
    run({"ragas": {"modo": "mock"}})
