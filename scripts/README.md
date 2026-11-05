# Scripts

Wrappers de línea de comandos. Añaden `src/` al `PYTHONPATH` para funcionar sin instalación editable.

```bash
python scripts/run_ner_inference.py --config configs/ner.yaml
python scripts/run_harness.py --config configs/harness.yaml
python scripts/run_corpus.py --config configs/corpus.yaml --fuentes configs/sources.yaml
python scripts/run_retrieval.py --config configs/retrieval.yaml
python scripts/run_generation.py --config configs/generation.yaml
python scripts/run_ragas.py --config configs/ragas.yaml --modo real
python scripts/run_vision.py --config configs/vision.yaml
python scripts/generate_visual_set.py --config configs/synthetic.yaml
python scripts/run_transcription_metrics.py --config configs/transcription.yaml
python scripts/run_app.py
```
