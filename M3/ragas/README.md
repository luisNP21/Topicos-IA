# M3/ragas — Evaluación RAGAS + cruce con harness M2

Evalúa el sistema RAG de M3 con las cuatro métricas estándar de **RAGAS** y las
cruza con el **F1 de extracción de M2** para diagnosticar en qué etapa del pipeline
falla el sistema.

---

## Qué mide cada métrica y qué te dice si falla

| Métrica | Pregunta que responde | Si falla, el problema está en… |
|---|---|---|
| **Faithfulness** | ¿La respuesta se basa en el contexto recuperado? | Generación (alucina cosas fuera del contexto) |
| **Context precision** | ¿Los chunks recuperados son relevantes? | Retrieval ( trae ruido) |
| **Context recall** | ¿Se recuperó todo lo necesario? | Corpus  o retrieval (no lo encontró) |
| **Answer relevancy** | ¿La respuesta contesta lo que se preguntó? | Retrieval + generación combinados |

---

## Modos de ejecución

### Modo mock (sin GROQ_API_KEY, sin datos reales)
Calcula las métricas manualmente usando embeddings locales.

```bash
python run_ragas.py --config config.yaml
# o explícitamente:
python run_ragas.py --config config.yaml --modo mock
```

### Modo real (con GROQ_API_KEY y datos reales)
Usa la librería `ragas` con el mismo LLM juez (`qwen/qwen3.8-27b` via Groq)
que se usó en M2 Dimensión 3 y M3 generación:

```bash
python run_ragas.py --config config.yaml --modo real
```

Requiere `GROQ_API_KEY` en `.env` o en la ventana de Secretos de Colab.

---

## Instalación

```bash
pip install -r requirements.txt
```

Para modo mock solo se necesitan `sentence-transformers`, `numpy` y `pyyaml`.
`ragas` y `langchain-groq` solo son necesarios para modo real.

---

## Resultados (corrida real)

Reportes versionados en `M3/ragas/resultados/`. El `run_ragas.py` escribe un JSON nuevo por
corrida (no sobrescribe) con el nombre `evaluacion_m3_<fecha>Z.json`. La corrida de referencia es
la última: `evaluacion_m3_20261103T185126284245Z.json`.

RAGAS no calcula todas las métricas en los 15 casos: en abstenciones y respuestas sin afirmaciones
verificables la métrica queda sin valor. Por eso cada métrica se reporta **con su cobertura** y el
promedio es sobre los casos válidos, no sobre 15.

| Métrica | Valor | Cobertura (casos válidos de 15) |
|---|---|---|
| Faithfulness (Fidelidad) | 0.406 | 7 |
| Context Precision (Precisión) | 0.222 | 4 |
| Context Recall (Cobertura) | 0.333 | 8 |
| Answer Relevancy (Relevancia) | 0.295 | 7 |

El cruce con el harness (`cruce_harness.py`) corre sobre los 15 casos y presenta por separado el F1
de extracción de M2 y los agregados de RAGAS (no son muestras pareadas, así que no se infiere un
diagnóstico por documento). El reparto **no marca ningún caso como alucinación**: la limitante es
la cobertura del corpus de 3 guías, no la generación.

Lectura: los scores bajos reflejan la cobertura del corpus, no invención. En los casos respondidos
la relevancia es alta donde el juez pudo calcularla (caso_11 0.92). Una corrida previa
(`evaluacion_m3_20261026T041555261137Z.json`, faithfulness 0.371, recall 0.143) queda versionada
como evidencia de la variabilidad del juez LLM entre corridas.

---

## Cómo conectar con los datos reales de 

Cuando entregue los `contexts` y Agustín las `answers`, el `eval_set_externo`
se pasa en `cfg` reemplazando el simulado integrado:

```python
from evaluacion_ragas import run

cfg = {
    "ragas": {"modo": "real", ...},
    "eval_set_externo": [
        {
            "question":     "diabetes mellitus tipo 2",
            "contexts":     ["...fragmento recuperado por Pau..."],
            "answer":       "...respuesta generada por Agustín...",
            "ground_truth": "...respuesta esperada del gold set...",
        },
        ...
    ]
}
resultado = run(cfg=cfg, project_root="/ruta/al/proyecto")
```

---

## Estructura

```
M3/ragas/
├── config.yaml          # modo, modelo LLM, umbrales de diagnostico
├── config_utils.py      # cargar_config() — mismo patron que M3/corpus/
├── requirements.txt     # sentence-transformers, ragas, langchain-groq, etc.
├── run_ragas.py         # entrypoint CLI
├── evaluacion_ragas.py  # 4 metricas RAGAS (mock + real)
└── cruce_harness.py     # cruza RAGAS con F1 de M2 y genera diagnostico
```

---

## Cruce con el harness de M2 

El módulo `cruce_harness.py` lee `M2/ejecucion/outputs_gold/resultado_dimension1.json`,
calcula el F1 por documento desde `true_by_doc` y `pred_by_doc`, y genera una tabla
de diagnóstico automático:

| F1 extracción | Context recall | Faithfulness | Diagnóstico |
|---|---|---|---|
| Bajo | Alto | Alto | **problema_extraccion** — M1/M2 no detectó bien; el resto funciona |
| Alto | Bajo | — | **problema_corpus_retrieval** — no hay guía clínica o retrieval no la encontró |
| — | — | Bajo | **alucinacion_generacion** — el LLM inventó cosas fuera del contexto |
| Alto | Alto | Alto | **funcionamiento_correcto** |

La lógica de diagnóstico replica explícitamente `diagnosticar_debilidad()` de
`M2/harness/scorecard.py` para mantener consistencia de diseño a lo largo del proyecto.

