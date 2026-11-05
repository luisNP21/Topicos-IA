# Clinical RAG

Sistema clínico que detecta enfermedades en notas clínicas y recupera evidencia de guías para responder.

```text
nota clínica (texto o imagen)
  → detección de enfermedades (NER + LoRA)
  → búsqueda en guías (retrieval + reranker)
  → compuerta de evidencia
  → generación con LLM y fuentes
```

## Estructura

```text
src/clinical_rag/     código del paquete
configs/              configuración YAML
data/                 gold sets y datos pequeños versionados
artifacts/            modelos, predicciones y salidas (no versionar)
notebooks/            notebooks de experimentación y ejecución
scripts/              entry points de línea de comandos
tests/                pruebas
docs/                 documentación y diagramas
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # completar PROJECT_ROOT y GROQ_API_KEY
```

Para usar corpus, RAGAS o visión:

```bash
pip install -e ".[corpus,ragas,vision]"
```

## Configuración

- `PROJECT_ROOT`: raíz de datos/artefactos. Si se omite, se usa la raíz del repo.
- `GROQ_API_KEY`: generación y LLM-as-judge.
- `BIOPORTAL_API_KEY`: normalización SNOMED CT real (opcional).
- `TESSERACT_CMD`, `TESSDATA_PREFIX`: OCR clásico (opcional).

## Uso rápido

```bash
python scripts/run_ner_inference.py --config configs/ner.yaml
python scripts/run_harness.py --config configs/harness.yaml
python scripts/run_app.py
```

Ver `docs/` para la documentación técnica, los diagramas y las guías de exposición. El mapa de la estructura anterior a la actual está en `docs/MIGRACION_ESTRUCTURA.md`.
