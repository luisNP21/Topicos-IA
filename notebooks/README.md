# Notebooks

Notebooks de referencia, organizados por etapa:

- `ner/`: preparación, chunking y fine-tuning del detector.
- `evaluation/`: exact-match, similitud semántica, LLM-as-judge y harness.
- `rag/`: corpus, retrieval, normalización, generación y RAGAS.
- `vision/`: set sintético, transcripción, métricas, harness visual y app.

> **Importante:** estos notebooks conservan las rutas y comandos de las corridas originales en Colab. La estructura funcional actual usa `src/clinical_rag`, `configs/` y `scripts/`. Para ejecutar el proyecto, usa los entry points de `scripts/` o `pip install -e .`. Ver `docs/MIGRACION_ESTRUCTURA.md`.
