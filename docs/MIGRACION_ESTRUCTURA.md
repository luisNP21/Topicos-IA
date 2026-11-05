# Migración de estructura

La rama `refactor/project-structure` organiza el proyecto por responsabilidades, no por módulos de clase.

## Mapa de rutas

| Antes | Ahora |
|---|---|
| `M1/*/` (adaptadores) | `artifacts/models/*/` |
| `M1/*.ipynb` | `notebooks/ner/` |
| `M2/common.py` | `src/clinical_rag/common.py` |
| `M2/run_inference/` | `src/clinical_rag/ner/` |
| `M2/harness/` | `src/clinical_rag/evaluation/` |
| `M2/examples/*.jsonl` | `data/gold/` |
| `M2/ejecucion/outputs_*` | `reports/ner/` |
| `M3/corpus/` | `src/clinical_rag/corpus/` + `configs/corpus.yaml` + `configs/sources.yaml` |
| `M3/retrieval/` | `src/clinical_rag/retrieval/` + `configs/retrieval.yaml` + `data/retrieval/` |
| `M3/generacion/` | `src/clinical_rag/generation/` + `configs/generation.yaml` |
| `M3/tools/` | `src/clinical_rag/tools/` + `configs/tools.yaml` |
| `M3/ragas/` | `src/clinical_rag/evaluation/ragas/` + `configs/ragas.yaml` |
| `M3/ejecucion/eval_set_casos.json` | `data/gold/eval_set_m3.json` |
| `M4/vision/` | `src/clinical_rag/vision/` + `configs/vision.yaml` |
| `M4/eval_visual/` | `src/clinical_rag/synthetic/` + `configs/synthetic.yaml` + `assets/fonts/` |
| `M4/metricas_transcripcion/` | `src/clinical_rag/evaluation/transcription/` + `configs/transcription.yaml` |
| `M4/app/` | `src/clinical_rag/app/` + `configs/app.yaml` |
| `M4/data/imagenes/` | `data/visual/` |
| `M4/vision/predictions/` | `artifacts/vision/predictions/` |
| `M4/vision/outputs/` | `artifacts/vision/outputs/` |

## Entry points

Los scripts de `scripts/` son los puntos de entrada recomendados:

```bash
python scripts/run_ner_inference.py --config configs/ner.yaml
python scripts/run_harness.py --config configs/harness.yaml
python scripts/run_corpus.py --config configs/corpus.yaml --fuentes configs/sources.yaml
python scripts/run_retrieval.py --config configs/retrieval.yaml
python scripts/run_generation.py --config configs/generation.yaml
python scripts/run_ragas.py --config configs/ragas.yaml
python scripts/run_vision.py --config configs/vision.yaml
python scripts/generate_visual_set.py --config configs/synthetic.yaml
python scripts/run_transcription_metrics.py --config configs/transcription.yaml
python scripts/run_app.py
```

También se pueden usar los entry points instalados:

```bash
pip install -e ".[dev]"
clinical-rag-app
```

## Notebooks

Los notebooks se conservan en `notebooks/` como referencia de las corridas originales. Fueron escritos para Colab y para la estructura anterior; los scripts y el paquete `clinical_rag` son la fuente funcional actual. Si se van a reutilizar, hay que ajustar sus rutas de `PROJECT_ROOT`, `sys.path` y configs a esta estructura.
