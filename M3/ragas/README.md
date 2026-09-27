# M3/ragas — Evaluación RAGAS + Cruce con Harness M2 (Luis)

Módulo de evaluación integral del sistema RAG clínico y diagnóstico de debilidades por etapas.
Cumple con el criterio de **Nivel 4 (5 puntos)** de la rúbrica de M3.

---

## 🎯 ¿Qué hace este módulo?

1. **Calcula las 4 métricas estándar de RAGAS:**
   * **`faithfulness`** (Fidelidad): ¿La respuesta del LLM se sustenta exclusivamente en los fragmentos clínicos recuperados o alucina?
   * **`context_precision`** (Precisión del contexto): ¿Los pasajes traídos por el retrieval son realmente pertinentes para la consulta?
   * **`context_recall`** (Cobertura del contexto): ¿El retrieval logró encontrar toda la evidencia necesaria para la recomendación?
   * **`answer_relevancy`** (Relevancia de respuesta): ¿La respuesta final atiende directamente la consulta formulada?

2. **Cruce multidimensional con el Harness de M2:**
   Toma el archivo `M2/ejecucion/outputs_gold/resultado_dimension1.json` (donde Clinical BERT obtuvo un F1 de 0.72) y calcula el F1 por documento clínico para cruzarlo con el comportamiento del RAG.

3. **Diagnóstico automático de fallas:**
   * `problema_extraccion`: El extractor falló (F1 < 0.4), pero el retrieval y generación habrían funcionado.
   * `problema_corpus_retrieval`: Extracción perfecta (F1 ≥ 0.7), pero no se encontró la guía en el corpus (Recall < 0.4).
   * `alucinacion_generacion`: El LLM inventó datos no respaldados por las fuentes (Faithfulness < 0.4).
   * `funcionamiento_correcto`: Todas las etapas superan el estándar de calidad.

---

## 🚀 Modos de Ejecución

### Modo Mock (Prueba rápida / Sin costo de API)
```bash
python run_ragas.py --config config.yaml --modo mock
```
Calcula las métricas con embeddings multilingües y el cruce con M2 usando los casos disponibles.

### Modo Real (Con LLM Groq como Juez)
```bash
python run_ragas.py --config config.yaml --modo real
```
Requiere la variable `GROQ_API_KEY` en `.env` o en los Secretos de Google Colab (🔑).

---

## 📦 Estructura del Módulo

```
M3/ragas/
├── config.yaml          # Configuración, rutas a outputs y umbrales
├── config_utils.py      # Helper de carga de YAML
├── cruce_harness.py     # Lógica de cruce con M2 y diagnóstico
├── evaluacion_ragas.py  # Cálculo de las 4 métricas RAGAS
├── requirements.txt     # Dependencias del módulo
├── run_ragas.py         # Entrypoint CLI
└── start_ragas.ipynb    # Cuaderno lanzador para Google Colab
```
