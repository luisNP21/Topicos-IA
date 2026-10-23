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
from math import isfinite
from numbers import Real
import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
        for i, r in enumerate(resultados):
            # Soporte completo para pregunta y esperado
            q = r.get("pregunta") or r.get("question") or r.get("consulta") or r.get("entidad") or r.get("query_final", "")
            ctx = r.get("contexts", [])
            ans = r.get("answer", "")
            gt = r.get("esperado") or r.get("ground_truth")
            items.append({
                "id": r.get("id", f"fila_{i:04d}"),
                "fallback_used": r.get("fallback_used"),
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


def _faithfulness_mock(caso: dict, model) -> float | None:
    contexto = " ".join(caso["contexts"])
    afirmaciones = [s.strip() for s in re.split(r"[.\n]", caso["answer"]) if len(s.strip()) > 10]
    if not afirmaciones or not contexto.strip():
        # Convención del proxy: sin base de comparación no asignar una nota.
        # N/D no significa respuesta segura; la abstención se registra aparte.
        return None
    ok = sum(1 for a in afirmaciones if _sim(a, contexto, model) >= 0.4)
    return ok / len(afirmaciones)


def _context_precision_mock(caso: dict, model) -> float | None:
    if not caso["contexts"] or not " ".join(caso["contexts"]).strip():
        return None
    relevantes = sum(1 for ch in caso["contexts"] if _sim(caso["question"], ch, model) >= 0.3)
    return relevantes / len(caso["contexts"])


def _context_recall_mock(caso: dict, model) -> float | None:
    contexto = " ".join(caso["contexts"])
    if not caso.get("ground_truth"):
        return None
    if not contexto:
        # No se recupero evidencia de guia: cobertura nula (problema de corpus/retrieval)
        return 0.0
    return min(1.0, _sim(caso["ground_truth"], contexto, model) + 0.15)


def _answer_relevancy_mock(caso: dict, model) -> float:
    return min(1.0, _sim(caso["question"], caso["answer"], model) + 0.20)


METRICAS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def _resumir_filas(filas: list[dict], eval_set: list[dict]) -> dict:
    """Conserva valores finitos y explicita el denominador de cada media.

    Null representa ausencia, resultado no finito o tipo no numérico: no cero.
    Cada estado queda registrado. No se deduce la causa de un NaN.
    Los índices vinculan el resultado al orden de ESTA corrida, no a M2.
    """
    if not isinstance(filas, list) or not filas or len(filas) != len(eval_set):
        raise ValueError("Se requieren resultados por muestra, alineados con esta corrida.")
    por_caso = []
    for i, (fila, caso) in enumerate(zip(filas, eval_set)):
        if not isinstance(fila, dict):
            raise TypeError("Cada resultado debe ser un diccionario de métricas.")
        valores, estados = {}, {}
        for nombre in METRICAS:
            valor = fila.get(nombre)
            if isinstance(valor, Real) and not isinstance(valor, bool) and isfinite(float(valor)):
                valores[nombre], estados[nombre] = float(valor), "calculado"
            else:
                valores[nombre] = None
                estados[nombre] = "ausente" if valor is None else "no_finito_o_invalido"
        # Es una declaración del generador, no una certificación de seguridad.
        flag = caso.get("fallback_used")
        por_caso.append({"indice_evaluacion": i, "id": caso.get("id", f"fila_{i:04d}"),
                         "abstencion_declarada": flag if isinstance(flag, bool) else None,
                         "metricas": valores, "estado_metricas": estados})
    metricas, cobertura = {}, {}
    for nombre in METRICAS:
        validos = [r["metricas"][nombre] for r in por_caso if r["metricas"][nombre] is not None]
        metricas[nombre] = sum(validos) / len(validos) if validos else None
        cobertura[nombre] = {"n_validos": len(validos), "n_total": len(filas),
                             "n_no_calculados": len(filas) - len(validos)}
    total_validos = sum(v["n_validos"] for v in cobertura.values())
    estado = ("sin_metricas_validas" if total_validos == 0 else
              "completo" if total_validos == len(filas) * len(METRICAS) else "parcial")
    return {"metricas": metricas, "cobertura": cobertura, "por_caso": por_caso, "estado": estado}


def _calcular_mock(eval_set: list[dict], embedding_model: str, *, detalle: dict | None = None) -> dict:
    model = _get_emb_model(embedding_model)
    filas = [{"faithfulness": _faithfulness_mock(caso, model),
              "context_precision": _context_precision_mock(caso, model),
              "context_recall": _context_recall_mock(caso, model),
              "answer_relevancy": _answer_relevancy_mock(caso, model)} for caso in eval_set]
    reporte = _resumir_filas(filas, eval_set)
    if detalle is not None:
        detalle.update(reporte)
    return reporte["metricas"]


# Calculo de metricas en modo Real (Libreria Ragas + Groq LLM)


def _calcular_real(eval_set: list[dict], llm_model: str, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", *, detalle: dict | None = None) -> dict:
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

        llm = LangchainLLMWrapper(ChatGroq(model=llm_model, api_key=groq_key, temperature=0, max_retries=3, request_timeout=60))

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

        eval_kwargs = {
            "dataset": ds,
            "metrics": metricas_lista,
            "llm": llm,
            "embeddings": embeddings_wrapped,
        }
        try:
            from ragas.run_config import RunConfig
            eval_kwargs["run_config"] = RunConfig(max_workers=2, timeout=60, max_wait=30, max_retries=3)
        except Exception:
            pass

        try:
            resultado = evaluate(**eval_kwargs)
        except TypeError:
            resultado = evaluate(ds, metrics=metricas_lista, llm=llm, embeddings=embeddings_wrapped)

        filas = getattr(resultado, "scores", None)
        if not isinstance(filas, list) and hasattr(resultado, "to_pandas"):
            filas = resultado.to_pandas().to_dict(orient="records")
        reporte = _resumir_filas(filas, eval_set)
        if detalle is not None:
            detalle.update(reporte)
        return reporte["metricas"]
    except Exception as e:
        # No etiquetar un proxy como RAGAS. La causa se conserva en el traceback.
        raise RuntimeError(
            "No se completó la evaluación RAGAS; no se sustituyeron las métricas. "
            "Para una aproximación local, ejecutar explícitamente --modo mock."
        ) from e


def run(cfg: dict, project_root: str = "") -> dict:
    ragas_cfg = cfg.get("ragas", {})
    modo = ragas_cfg.get("modo", "mock")
    if modo not in {"real", "mock"}:
        raise ValueError("modo debe ser real o mock")
    embedding_model = ragas_cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    llm_model = ragas_cfg.get("llm_model", "qwen/qwen3.8-27b")
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
        if modo == "real":
            raise FileNotFoundError("No hay salidas de generación; no se evaluará el set de ejemplo como entrega.")
        eval_set = EVAL_SET_MOCK
        origen = "EVAL_SET_MOCK (simulado integrado)"

    print(f"[ragas] Modo: '{modo}' | Fuente de datos: {origen} | Casos: {len(eval_set)}")

    if modo == "real":
        for i, caso in enumerate(eval_set):
            for clave in ("question", "answer", "ground_truth"):
                if not isinstance(caso.get(clave), str) or not caso[clave].strip():
                    raise ValueError(f"Caso {i}: falta {clave}; no se inventan referencias de evaluación.")
            if not isinstance(caso.get("contexts"), list) or not all(isinstance(c, str) for c in caso["contexts"]):
                raise ValueError(f"Caso {i}: contexts debe ser una lista de textos.")

    detalle = {}
    if modo == "real":
        metricas = _calcular_real(eval_set, llm_model, embedding_model, detalle=detalle)
        backend = "ragas"
    else:
        metricas = _calcular_mock(eval_set, embedding_model, detalle=detalle)
        # _sim mantiene su respaldo léxico: no garantizar embeddings en cada llamada.
        backend = "proxy_local_con_respaldo_lexico"

    print(f"\n--- Evaluación: {backend} ({detalle['estado']}) ---")
    for nombre, valor in metricas.items():
        texto = "N/D" if valor is None else f"{valor:.4f}"
        c = detalle["cobertura"][nombre]
        print(f"  {nombre}: {texto} ({c['n_validos']}/{c['n_total']} valores calculados)")

    return {"modo": modo, "backend": backend, "fuente": origen,
            "n_casos": len(eval_set), "modelo_juez": llm_model if modo == "real" else None,
            "modelo_embeddings_configurado": embedding_model, **detalle}


if __name__ == "__main__":
    run({"ragas": {"modo": "mock"}})
