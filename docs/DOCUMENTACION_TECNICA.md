# Documentación técnica integrada — Sistema clínico NER + RAG

> **Repositorio:** `luisNP21/Topicos-IA`
> **Rama documentada:** `main`
> **Commit:** `2958935` — *Merge pull request #32 from luisNP21/dev*
> **Fecha del commit:** 2026-10-26
> **Alcance:** estado real del código en `main`, no una propuesta futura. Cuando el README de un módulo y el código no coinciden, este documento prioriza el código y lo señala explícitamente.

---

## 0. Resumen ejecutivo

El proyecto construye un asistente clínico para **documentar un caso real y decidir qué sección de una guía revisar primero**. El usuario es un médico o residente que tiene al paciente enfrente; la decisión que cambia es pasar de “leer toda la guía” a “recibir el fragmento relevante, ligado a lo que escribí en la nota”.

El flujo completo es:

```text
Nota clínica (texto o imagen)
        │
        ▼
[M4] Visión: imagen → texto transcrito
        │
        ▼
[M1/M2] NER clínico: texto → entidades de enfermedad
        │
        ▼
[M3] Retrieval sobre guías clínicas
        ├── normalización terminológica (SNOMED CT / BioPortal) si hace falta
        ├── búsqueda híbrida + reranking
        └── compuerta de evidencia
        │
        ▼
[M3] Generación con LLM y válvula de escape
        │
        ▼
[M3] Evaluación RAGAS + cruce con el harness de M2
        │
        ▼
[M4] App Gradio: resumen para el médico, fuentes y detalles técnicos
```

| Módulo | Responsabilidad principal | Resultado medido más citado |
|---|---|---|
| **M1** | Fine-tuning de NER de enfermedades sobre DisTEMIST | F1 0.708 con RoBERTa biomedical-clinical + LoRA |
| **M2** | Harness de evaluación del NER en tres dimensiones | F1 exacto 0.716; semántico 0.894; juez 3.869/5 (59 documentos) |
| **M3** | Corpus, retrieval, tool de normalización, generación y RAGAS | Tratamiento en 1.er lugar: 0.045 (ingenuo) → 0.773 (sistema completo) |
| **M4** | Entrada por imagen, evaluación visual y app de punto de entrada | Delta cero frente a texto en el harness: F1 0.617, semántico 0.660, juez 4.417 |

La idea central del diseño es **no reescribir el pipeline por módulo**: M4 solo cambia la fuente del texto; M3 consume las entidades de M1/M2; M2 evalúa el NER sin cargarlo; y la app reutiliza las mismas piezas.

---

## 1. Arquitectura y contratos entre módulos

### 1.1 Principios de diseño

1. **Separar inferencia de evaluación.** M2 no carga el modelo NER: lee un cache `{doc_id: [entidades]}` producido por `run_inference.py`. Así se puede evaluar el mismo cache con o sin visión, con o sin normalización, sin tocar el harness.
2. **Un contrato `run(cfg, project_root) -> dict`.** M2, M3 y M4 siguen el mismo patrón: configuración YAML, raíz de datos `PROJECT_ROOT`, salida JSON/Markdown.
3. **Fallos explícitos antes que silenciosos.** Si falta un `doc_id`, si el índice no coincide con los chunks, si el score es RRF y no se puede comparar con un umbral, o si un backend de visión falla, el código falla o lo registra; no inventa un resultado.
4. **Determinismo donde se pueda.** Semillas fijas, `temperature=0`, decisión de normalización por regla y umbral, no por function-calling del LLM.
5. **Datos fuera de git.** El corpus, las imágenes, los adaptadores y las salidas viven en `PROJECT_ROOT` (Drive/Colab o una copia local). El repositorio versiona código, configuración, gold sets pequeños y resultados de referencia.

### 1.2 Contratos de datos

```python
# Documento de entrada al NER
Documento = {"doc_id": str, "text": str}

# Salida del NER / cache que consume el harness
CacheNER = dict[str, list[str]]          # doc_id -> entidades predichas

# Chunk del corpus de M3
Chunk = {
    "chunk_id": str,        # ej. "gpc_diabetes_mellitus_tipo_2_chunk7"
    "doc_id": str,          # FK al manifiesto del corpus
    "texto": str,
    "seccion": str,         # título crudo de la sección
    "categoria_seccion": str,  # tratamiento | diagnostico | epidemiologia | otro
}

# Fragmento que devuelve el retrieval
Fragment = {
    "chunk_id": str, "doc_id": str, "texto": str,
    "score": float, "score_tipo": str,  # cosine | rrf | reranker_prob
    "fuente": str, "seccion": str, "seccion_tipo": str,
}

# Resultado de la tool de normalización
NormalizationResult = {
    "entidad_original": str,
    "entidad_normalizada": str,
    "source_terminology": str | None,
    "normalization_failed": bool,
}

# Salida de la generación
RespuestaRAG = {
    "answer": str,
    "sources_used": list[str],   # chunk_id en los que se apoya
    "fallback_used": bool,       # True si no había evidencia
}
```

### 1.3 Mapa del repositorio

```text
proyecto/
├── README.md                     # definición del proyecto (dominio, usuario, métricas)
├── M1/                           # fine-tuning NER
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_dataset_chunking_and_splits_base.ipynb
│   ├── 03_finetuning_base.ipynb
│   ├── BERT/ clinical_BERT/ mT5/ # ejecuciones por modelo + adaptadores LoRA
│   └── saved_models/             # adaptador seleccionado para producción
├── M2/                           # evaluación del NER
│   ├── run_inference/            # NER por chunks + cache
│   ├── harness/                  # exact-match, semántica, juez, scorecard
│   ├── eval_harness/ examples/   # gold sets
│   ├── ejecucion/                # notebooks y resultados versionados
│   └── outputs/                  # salidas locales (gitignored)
├── M3/                           # RAG clínico
│   ├── corpus/                   # PDFs → chunks → embeddings → Chroma
│   ├── retrieval/                # denso, BM25, RRF, reranker, intención
│   ├── tools/                    # normalización SNOMED CT / BioPortal
│   ├── generacion/               # orquestación + LLM + válvula de escape
│   ├── ragas/                    # evaluación RAGAS + cruce con M2
│   ├── ejecucion/                # pipeline integrador
│   └── outputs/                  # resultados (gitignored)
├── M4/                           # componente visual
│   ├── vision/                   # VLM/OCR, prompts, cache, puente NER
│   ├── eval_visual/              # generación de imágenes sintéticas
│   ├── metricas_transcripcion/   # CER/WER y auditoría de entidades
│   ├── app/                      # app Gradio
│   ├── ejecucion/                # notebooks de la tanda
│   └── eval_set/                 # muestra versionada
└── .gitignore
```

---

## 2. Módulo 1 — Fine-tuning del NER clínico

### 2.1 Para qué sirve

M1 resuelve **Named Entity Recognition (NER) de nombres de enfermedad** en texto clínico en español. Es la primera etapa del pipeline: sus entidades son la consulta que M3 usa para buscar en las guías. El módulo compara tres arquitecturas y elige la mejor:

| Clave | Modelo base | Arquitectura | Carpeta de salida |
|---|---|---|---|
| `bert` | `dccuchile/bert-base-spanish-wwm-cased` | Encoder, token classification | `M1/BERT/bert-distemist-lora/` |
| `clinical_bert` | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | Encoder, token classification | `M1/clinical_BERT/clinical_bert-distemist-lora/` |
| `mt5` | `google/mt5-small` | Encoder-decoder, seq2seq | `M1/mT5/mt5-distemist-lora/` |

> **Aclaración de nombres:** la carpeta se llama `clinical_BERT`, pero el modelo real es un **RoBERTa biomedical-clinical** (`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`). El adaptador seleccionado y usado por M2/M3/M4 es `M1/saved_models/clinical_bert-distemist-lora`.

### 2.2 Datos y particiones

- **Corpus:** DisTEMIST, 1.000 casos clínicos en español anotados con menciones de enfermedad; se usa el subtrack de entidades (`subtrack1_entities`), no el de normalización a SNOMED CT.
- **Origen:** SPACCC / SciELO, licencia CC BY 4.0, disponible en Zenodo.
- **Carga:** el notebook 1 lee `text_files/*.txt` y los `.tsv` de anotaciones (`filename`, `off0`, `off1`, `span`, `label`). Agrupa las anotaciones por `filename`, corrige escapes Unicode (`\uXXXX`) y arma registros `{doc_id, text, entities}`.
- **Split:** 80/10/10 sobre documentos **antes del chunking**, con semilla 42, para evitar fuga de datos entre fragmentos del mismo paciente.
- **Partición observada en el notebook versionado:** `max_docs=30` → 24 train / 3 dev / 3 test. Las métricas de M1 que se citan abajo corresponden a esa corrida.

> **Punto de atención:** el `README.md` raíz afirma que se usaron los 750 casos del training set, pero el notebook `01_dataset_preparation.ipynb` versionado deja `max_docs=30` y su salida registrada es de 30 documentos. Para reproducir las cifras hay que mantener ese valor; para usar 750 hay que cambiarlo y volver a generar los splits.

### 2.3 Implementación paso a paso

#### Notebook 1 — `01_dataset_preparation.ipynb`

1. Monta Google Drive si está en Colab.
2. Descarga/lee DisTEMIST desde una carpeta local de Drive.
3. `fix_unicode_escapes()` convierte secuencias `\uXXXX` en caracteres reales.
4. `load_from_tsv()`:
   - lee todos los `.tsv` de `subtrack1_entities/`;
   - agrupa por `filename`;
   - muestrea `max_docs` documentos de forma reproducible;
   - devuelve `doc_id`, `text` y lista de entidades con offsets.
5. Prueba cuatro tokenizadores y mide:
   - tokens por documento;
   - ratio tokens/palabra;
   - ventana segura de palabras con `compute_safe_window_words()`: usa el **percentil 90** del ratio tokens/palabra y un margen de seguridad de 0.9 para apuntar a ~480 tokens, dejando espacio a `[CLS]`/`[SEP]`.
6. `make_split()` mezcla con semilla 42 y parte 80/10/10.
7. Guarda `DatasetDict` en `distemist_raw_splits` con Apache Arrow (HuggingFace `datasets`).

#### Notebook 2 — `02_dataset_chunking_and_splits_base.ipynb`

1. Carga `distemist_raw_splits`.
2. Vuelve a medir longitudes con el tokenizador elegido.
3. `chunk_record_by_words()`:
   - busca palabras con `re.finditer(r"\S+", text)`;
   - si el documento cabe en la ventana, no lo parte;
   - corta por ventana de palabras con solapamiento;
   - reajusta los offsets de cada entidad al chunk;
   - deriva `doc_id = "<original>_chunk<k>"`.
4. `char_spans_to_bio()` convierte offsets de carácter a etiquetas BIO a nivel de palabra:
   - `O`, `B-ENFERMEDAD`, `I-ENFERMEDAD`.
5. `build_bert_split()` produce `{doc_id, tokens, ner_tags}`.
6. `build_mt5_split()` produce `{doc_id, input_text, target_text}`:
   - entrada: `"Identifica las enfermedades mencionadas en el siguiente texto clínico: " + texto`;
   - salida: términos separados por `" [SEP] "`, sin duplicados, o `"ninguna"`.
7. Guarda dos formatos: `distemist_bert_format` y `distemist_mt5_format`.

**Ventanas usadas por modelo:**

| Modelo | Tokenizador | Ventana de palabras | Overlap |
|---|---|---:|---:|
| Base / RoBERTa clínico | `Roberta-clinico-es` | 277 | 50 |
| BERT español | `bert` | 246 | 50 |
| Clinical BERT (RoBERTa biomedical) | `clinical_bert` | 237 | 50 |
| mT5 | `mT5` | 180 | 50 |

La ventana no es un valor único para los tres modelos: se calcula con el tokenizador real de cada uno para no truncar entidades.

#### Notebook 3 — `03_finetuning_base.ipynb`

1. `MODEL_REGISTRY` define checkpoint, arquitectura, módulos LoRA, tipo de tarea y dataset.
2. Carga el modelo base con `AutoModelForTokenClassification` o `AutoModelForSeq2SeqLM`.
3. Tokeniza y alinea etiquetas:
   - BERT: `tokenize_and_align_labels_bert()`; las subpalabras reciben `-100` o el tag convertido de `B-` a `I-`.
   - mT5: `tokenize_mt5()` con `max_length=512` en entrada y `128` en salida.
4. Evalúa el **baseline sin LoRA** para tener un punto de comparación.
5. Configura LoRA:
   - `r=8`, `lora_alpha=16`, `lora_dropout=0.05`;
   - `target_modules=["query", "value"]` en encoders y `["q", "v"]` en mT5;
   - `task_type=TOKEN_CLS` o `SEQ_2_SEQ_LM`.
6. Entrena con `Trainer` / `Seq2SeqTrainer`:
   - `learning_rate=2e-4`;
   - `num_train_epochs=10`;
   - `per_device_train_batch_size=2`, `per_device_eval_batch_size=4`;
   - evaluación y guardado por época;
   - `load_best_model_at_end=True`, `metric_for_best_model="f1"`;
   - `fp16` en GPU para encoders; **`fp16=False` en mT5** porque T5/mT5 produce NaN en `layer_norm` con fp16.
7. Calcula métricas **micro-PRF1 agregando por documento original**:
   - `compute_metrics_bert()` convierte predicciones BIO a conjuntos de entidades;
   - `compute_metrics_mt5()` parsea la salida generada;
   - `aggregate_entities_by_original_doc()` quita el sufijo `_chunkN` y une los chunks del mismo caso;
   - `micro_prf1_by_doc()` suma TP/FP/FN documento por documento, evitando que una entidad repetida entre pacientes infle el conteo.
8. Compara baseline vs. fine-tuned, grafica curvas y guarda el adaptador con `model.save_pretrained()` + `tokenizer.save_pretrained()`.

### 2.4 Resultados de M1

| Modelo | Baseline F1 | Fine-tuned Precision | Fine-tuned Recall | Fine-tuned F1 |
|---|---:|---:|---:|---:|
| BERT español | 0.0047 | 0.623 | 0.597 | **0.610** |
| Clinical BERT / RoBERTa biomedical | 0.0125 | 0.697 | 0.719 | **0.708** |
| mT5-small | 0.0000 | 0.161 | 0.094 | **0.119** |

**Decisión:** se selecciona **RoBERTa biomedical-clinical + LoRA** (`clinical_bert`) por el mejor F1 y por ser un encoder, adecuado para clasificación de tokens. El decoder mT5 genera texto y puede inventar una entidad que no está en el texto; además su F1 es mucho menor.

**Tamaño de los splits procesados:**

| Modelo | Train | Dev | Test |
|---|---:|---:|---:|
| BERT | 1.201 | 151 | 166 |
| Clinical BERT | 1.237 | 157 | 170 |
| mT5 | 1.668 | 212 | 233 |

### 2.5 Artefactos y reutilización

- Adaptador: `M1/saved_models/clinical_bert-distemist-lora/` con `adapter_config.json`, `adapter_model.safetensors`, tokenizer y README de PEFT.
- En M2/M3/M4 se carga así:
  1. `AutoModelForTokenClassification.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es", num_labels=3)`;
  2. `PeftModel.from_pretrained(base_model, adapter_path)`;
  3. `AutoTokenizer.from_pretrained(adapter_path)`.
- `id2label = {0: "O", 1: "B-ENFERMEDAD", 2: "I-ENFERMEDAD"}`.
- `sanity_check()` en M2 predice una frase de prueba para verificar que el adaptador cargó.

### 2.6 Límites de M1

- El NER se entrenó y evalúa a nivel de **span exacto**; los errores de boundary se miden en M2 con la dimensión semántica.
- El corpus es de casos clínicos, no de notas manuscritas ni de informes con formato hospitalario real.
- La partición de 3 documentos de test hace que las métricas tengan alta varianza.
- El modelo no normaliza a SNOMED CT; eso se hace después con la tool de M3.

---

## 3. Módulo 2 — Harness de evaluación del NER

### 3.1 Para qué sirve

M2 responde tres preguntas distintas sobre las predicciones de M1/M2:

1. **Exact-match:** ¿la entidad predicha coincide carácter a carácter con el gold?
2. **Similitud semántica:** ¿es equivalente aunque el span no coincida exacto?
3. **LLM-as-judge:** ¿un LLM considera que la predicción es clínicamente útil y cumple el criterio del caso?

Las tres se combinan en un **scorecard** que diagnostica el tipo de debilidad. La inferencia y la evaluación están separadas: `run_inference.py` produce un cache por documento; `run_harness.py` evalúa ese cache sin cargar el modelo NER.

### 3.2 Arquitectura del módulo

```text
M1/saved_models/clinical_bert-distemist-lora
        │
        ▼
M2/run_inference/run_inference.py
  ├── carga encoder + LoRA
  ├── chunking 277 palabras / 50 overlap
  ├── predice BIO por palabra
  ├── agrega chunks por doc_id original
  └── escribe M2/predictions/encoder_solo.json
        │
        ▼
M2/harness/run_harness.py
  ├── lee el cache y el gold set
  ├── exact-match
  ├── similitud semántica
  ├── LLM-as-judge
  └── scorecard_m2.json / scorecard_m2.md
```

**Archivos clave:**

| Archivo | Para qué sirve | Cómo está implementado |
|---|---|---|
| `M2/run_inference/run_inference.py` | Orquesta la inferencia | Resuelve `PROJECT_ROOT`, carga config, elige fuente de texto (`texto` o `vision`), ejecuta NER por chunks y escribe el cache |
| `M2/run_inference/model_loader.py` | Carga el modelo | `AutoModelForTokenClassification` + `PeftModel` + tokenizer del adaptador |
| `M2/run_inference/inference.py` | Predicción | `predict_word_level()` colapsa subtokens a palabra; `predict_entities_chunked()` recorre chunks |
| `M2/common.py` | Utilidades compartidas | Normalización, BIO→entidades, chunking, micro-PRF1, agregación por documento |
| `M2/harness/harness.py` | Orquesta dimensiones | Siempre exact-match; semántica y juez según `solo` |
| `M2/harness/metrics_exact.py` | Dimensión 1 | Compara conjuntos normalizados por documento y detecta boundary |
| `M2/harness/metrics_semantic.py` | Dimensión 1b | Embeddings, coseno, algoritmo húngaro, umbral calibrado por Youden |
| `M2/harness/metrics_judge.py` | Dimensión 3 | Groq, rúbrica 1-5, `cumple_criterio`, sesgos de posición y longitud |
| `M2/harness/scorecard.py` | Integrador | Cruza las tres dimensiones y emite diagnóstico por documento |
| `M2/harness/cached_system.py` | Adaptador del cache | Valida que los `doc_id` coincidan exactamente con el eval set |
| `M2/harness/gold_loader.py` | Carga y rutas | Lee JSONL, normaliza `input`/`esperado`, deriva rutas desde el YAML de inferencia |

### 3.3 Dimensión 1 — Exact-match

Implementación en `metrics_exact.py`:

- Normaliza cada entidad con `normalizar_entidad()`: minúsculas, `strip()` y eliminación de puntuación en los bordes.
- Convierte gold y predicción en `set[str]` por documento.
- `micro_prf1_by_doc()` suma TP/FP/FN de todos los documentos y calcula precision, recall y F1 global.
- `detectar_boundary_errors()` busca pares donde una entidad es substring de la otra; esos casos alimentan la calibración de la dimensión semántica.
- Exporta `casos_boundary` y los errores por documento a `resultado_dimension1.json`.

**Limitación:** es estricto; no reconoce un span más corto/largo ni sinónimos. Es el criterio usado para seleccionar el mejor checkpoint en M1.

### 3.4 Dimensión 1b — Similitud semántica

Implementación en `metrics_semantic.py`:

1. **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
2. **Matriz de similitud:** coseno entre entidades predichas y gold del mismo documento.
3. **Asignación 1:1:** algoritmo húngaro (`match_hungaro()`) con penalización fuerte para pares bajo el umbral.
4. **Calibración del umbral:**
   - positivos: pares `casos_boundary` (una entidad es substring de la otra);
   - negativos: 4.000 pares cruzados entre documentos distintos, semilla 42;
   - se elige el umbral que maximiza el índice de Youden (TPR - FPR).
5. **Categorización de aciertos:** exacto, puntuación/formato, boundary, solapamiento parcial, sinonimia.
6. **Ambigüedad:** mide entidades gold del mismo documento que son difíciles de distinguir.

Salida: `resultado_dimension1b.json` con `precision`, `recall`, `f1`, `soft_f1`, umbral y detalle de emparejamientos.

### 3.5 Dimensión 3 — LLM-as-judge

Implementación en `metrics_judge.py`:

- **Modelo:** `qwen/qwen3.8-27b` vía Groq, `temperature=0`, `max_tokens=600`, pausa de 3 s entre llamadas.
- **Rúbrica pointwise 1-5** con anclas; el juez recibe el gold, la predicción y el campo `criterio` del caso.
- **Dimensión de dominio:** emite `cumple_criterio` (`si`/`no`) además del score. Se normaliza a booleano y se agrega como tasa de aciertos de dominio.
- **Sesgo de posición (protocolo pairwise A/B de S06):**
  - se comparan dos respuestas anónimas A/B;
  - se intercambia el orden;
  - solo se declara ganador si el veredicto coincide en ambos órdenes; si no, es empate;
  - el test es **diagnóstico** y no se promedia con el score final.
- **Sesgo de longitud:** pares de respuesta corta correcta vs. respuesta larga **parcialmente correcta** (no ruido sin relación).
- **Auto-preferencia:** se documenta; no hay test cross-family disponible porque el juez y el generador comparten familia Qwen.
- **Rotación de claves:** `GROQ_API_KEY`, `GROQ_API_KEY_2`, … para repartir cupo diario.
- **Reanudación:** guarda un checkpoint de resultados del juez para no repetir llamadas.

### 3.6 Scorecard integrador

`scorecard.py`:

- Carga las tres dimensiones y verifica que hayan evaluado el mismo número de ejemplos.
- Calcula F1 semántico por documento a partir de `detalle_emparejamientos` y `no_emparejados`.
- Aplica `diagnosticar_debilidad()` con umbrales:
  - F1 exacto bajo/alto: `0.2` / `0.7`;
  - score del juez bajo/alto: `2.5` / `3.5`.
- Diagnósticos:
  - `boundary/formato`: exacto bajo, semántico alto, juez alto;
  - `calidad clínica`: exacto y semántico altos, juez bajo;
  - `fallo real`: todas las métricas bajas;
  - `funcionamiento correcto`: todas altas;
  - `caso mixto`: cualquier otra combinación.
- Escribe `scorecard_m2.json` y `scorecard_m2.md`.
- Si existe el sidecar de visión, agrega una sección **Componente visual (M4)** con backend, modelo, latencia, válvulas y errores.

### 3.7 Configuración relevante

| Archivo | Clave | Valor actual |
|---|---|---|
| `M2/run_inference/config.yaml` | `modelo.checkpoint` | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` |
| | `modelo.adapter_path` | `M1/saved_models/clinical_bert-distemist-lora` |
| | `chunking` | `window_words: 277`, `overlap_words: 50` |
| | `fuente.tipo` | `texto` |
| | `eval_set_path` | `M2/eval_harness/gold_examples_20.jsonl` |
| | `output_path` | `M2/predictions/encoder_solo.json` |
| `M2/harness/config.yaml` | `rutas.gold_set` | `M2/eval_harness/gold_examples_20.jsonl` |
| | `rutas.outputs_dir` | `M2/outputs` |
| | `dimension3_llm_judge.judge_model` | `qwen/qwen3.8-27b` |
| | `dimension1b...modelo_embeddings` | `paraphrase-multilingual-MiniLM-L12-v2` |

> **Punto de atención para reproducir:** `gold_examples_20.jsonl` es un subconjunto local de 20 documentos que **no está versionado en `main`**. El repositorio sí versiona `M2/examples/gold_examples.jsonl` (59) y `gold_examples_adversariales.jsonl` (10). Para correr desde cero hay que copiar el gold a `M2/eval_harness/` o cambiar la ruta a `M2/examples/gold_examples.jsonl`. Los resultados de referencia versionados están en `M2/ejecucion/outputs_gold/` y `M2/ejecucion/outputs_adversariales/`.

### 3.8 Resultados de M2

**Gold set (59 documentos):**

| Dimensión | Precision | Recall | F1 / score |
|---|---:|---:|---:|
| Exact-match | 0.704 | 0.728 | **0.716** |
| Similitud semántica | 0.879 | 0.909 | **0.894** |
| LLM-as-judge (1-5) | — | — | **3.869** |
| Aciertos de dominio (sí/no) | — | — | **0.593** |

**Adversariales (10 documentos):**

| Dimensión | Precision | Recall | F1 / score |
|---|---:|---:|---:|
| Exact-match | 0.571 | 0.333 | **0.421** |
| Similitud semántica | 1.000 | 0.583 | **0.737** |
| LLM-as-judge (1-5) | — | — | **3.950** |
| Aciertos de dominio (sí/no) | — | — | **0.300** |

**Lectura:** en el gold, la brecha entre exacto (0.716) y semántico (0.894) indica que la mayor parte del error es de **boundary/formato**, no de comprensión de la entidad. En adversariales el exacto cae y el juez detecta 5/10 empates por posición: ante textos clínicamente incoherentes el veredicto depende del orden. Los adversariales actuales incluyen términos que no son enfermedades (`mocos`, `muerto`, `fiebre`), así que parte de la caída mide una tarea mal planteada para un NER, no solo una debilidad del modelo.

### 3.9 Extensión de M2 para visión (M4)

No se modificó el harness. La integración se hizo aguas arriba, en `run_inference.py`:

- `fuente.tipo: "vision"` importa `M4/vision`, transcribe el manifiesto, une páginas por `doc_id` y usa ese texto como entrada del NER.
- `M2/run_inference/config_vision.yaml` produce el cache `M2/predictions/vision_<backend>.json` y un sidecar `*.vision_meta.json` con backend, latencia, válvulas y errores.
- `M2/harness/config_vision.yaml` apunta a ese cache y escribe en `M2/outputs_vision/`.
- `cached_system` exige que cache y eval set tengan exactamente los mismos `doc_id`.

Este diseño permite comparar texto gold, imagen limpia y foto realista con **el mismo NER y el mismo harness**.

---

## 4. Módulo 3 — RAG clínico sobre guías

### 4.0 Qué construye

M3 conecta las entidades de M1/M2 con fragmentos de guías clínicas y genera una respuesta. Se compone de cinco piezas:

```text
Corpus (PDFs → chunks → Chroma)
        │
        ▼
Retrieval (denso + BM25 + RRF + reranker + intención)
        │
        ├── Tool de normalización (SNOMED CT / BioPortal)
        │
        ▼
Orquestación (3 zonas de score + compuerta de evidencia)
        │
        ▼
Generación (LLM + prompt S07 + válvula de escape)
        │
        ▼
RAGAS + cruce con harness M2
```

La decisión de diseño que condiciona todo el módulo: **las guías no están organizadas como “enfermedad → tratamiento”**. Un chunk puede mencionar la enfermedad muchas veces y no contener ninguna recomendación terapéutica. Por eso se agregan transformación de consulta por intención, reranking y una compuerta que prefiere no responder antes que responder con ruido.

### 4.1 Corpus

**Para qué sirve:** convertir PDFs de guías clínicas en chunks con metadatos, embeddings y un índice Chroma consultable.

**Implementación:**

| Archivo | Función | Detalle |
|---|---|---|
| `M3/corpus/ingesta.py` | `cargar_fuentes`, `ingerir_documento`, `filtrar_secciones` | Usa Docling `DocumentConverter`; recorre `doc.iterate_items()`; cada `section_header` abre una sección; las tablas se exportan a Markdown; el texto se acumula por sección |
| `M3/corpus/chunking.py` | `cargar_tokenizer`, `chunk_seccion`, `renumerar_chunks`, `clasificar_seccion` | Tokenizer de `intfloat/multilingual-e5-base`; ventanas por tokens reales, `max_tokens=500`, `overlap_tokens=50`; el `chunk_id` se renumera a nivel de documento; `categoria_seccion` se asigna por regex sobre el título |
| `M3/corpus/embeddings.py` | `embeber_chunks`, `embeber_query` | `passage: ` al indexar y `query: ` al consultar; normalización L2 |
| `M3/corpus/corpus_store.py` | `guardar_chunks_json`, `construir_indice_chroma`, `cargar_coleccion` | Un JSON por chunk como fuente de verdad; colección Chroma `guias_clinicas` con espacio coseno; `upsert` para no duplicar |
| `M3/corpus/pipeline_corpus.py` | `run(cfg, project_root)` | Orquesta ingesta → filtro → chunking → embeddings → store; cachea por `doc_id` si los chunks ya existen; fusiona el manifiesto en vez de sobrescribirlo |
| `M3/corpus/run_corpus.py` | CLI | Resuelve `PROJECT_ROOT`, `config.yaml` y `fuentes.yaml` |
| `M3/corpus/fuentes.yaml` | Registro de procedencia | `doc_id`, título, URL, licencia, fecha, responsable, archivo local |

**Filtro de front matter:** descarta secciones cuyo título coincide con portada, resumen, abstract, índice, glosario, abreviaturas, etc., y cualquier sección con menos de `min_palabras=25`. El objetivo es no indexar boilerplate.

**Clasificación de sección:** `categoria_seccion` se asigna por palabras clave en el título (`tratamiento`, `manejo`, `terapéutic`, `farmacológic`, `intervención`, `diagnóstico`, `epidemiología`, etc.). Es una clasificación barata y aproximada; el retrieval actual no la usa como filtro principal, sino que tiene su propio clasificador.

**Configuración real:** `M3/corpus/config.yaml` usa `multilingual-e5-base`, `max_tokens=500`, `overlap_tokens=50`, `min_palabras=25`.

**Resultado del corpus real:** 3 guías y 64 chunks:

| Guía | Chunks |
|---|---:|
| `gpc_diabetes_mellitus_tipo_2` | 12 |
| `gpc_falla_cardiaca` | 23 |
| `gpc_depresion_adultos` | 29 |

### 4.2 Retrieval

**Para qué sirve:** dada una entidad (“diabetes mellitus tipo 2”, “DM2”, “ICC”), devolver los fragmentos de guía que probablemente contienen el tratamiento.

**Técnicas implementadas en `M3/retrieval/retrieval.py`:**

1. **Búsqueda densa (`densa`):** Chroma + embeddings e5, similitud coseno (`1 - distancia`). Es el RAG ingenuo de referencia.
2. **BM25 (`bm25`):** búsqueda léxica sobre `rank_bm25`; cubre erratas, siglas y tildes faltantes.
3. **Fusión RRF (`_rrf`):** Reciprocal Rank Fusion con `krrf=60`; combina densa y BM25 por rango, sin pesos manuales.
4. **Reranker cross-encoder (`_rerank`):** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sobre los 30 mejores candidatos. Si el modelo devuelve logits fuera de `[0,1]`, aplica sigmoide.
5. **Encabezado contextual:** BM25 y reranker leen `"título de la guía. sección. texto"`, para que un chunk de tratamiento que no repite la enfermedad pueda encontrarse por el título de su guía.
6. **Query transformation por intención:** la entidad se reescribe con tres plantillas:
   - `tratamiento de {e}`
   - `manejo farmacológico de {e}`
   - `{e}: recomendaciones terapéuticas`
   y el reranker recibe la pregunta `"¿Cuál es el tratamiento de {e}?"`.
7. **Filtro por sección (`intencion_seccion`):** el reranker ordena solo chunks clasificados como de tratamiento; si ninguno lo es, cae a todos.
8. **Clasificador de secciones:** primero regex sobre el título; si no es concluyente, similitud del embedding del chunk con prototipos de “tratamiento”, “diagnóstico”, “descripción” y “otro”.

**Modos disponibles:** `densa`, `hibrida`, `rerank`, `rerank_pregunta`, `intencion_sin_rerank`, `intencion`, `intencion_seccion`. El modo del pipeline es `intencion`.

**Contrato público:**

```python
retrieve_naive(query, k) -> list[Fragment]
retrieve_advanced(query, k) -> list[Fragment]
```

Cada `Fragment` incluye `chunk_id`, `doc_id`, `texto`, `score`, `score_tipo`, `fuente`, `seccion`, `seccion_tipo`.

**Configuración y robustez (`config_retrieval.py`):**

- Perfiles `mock` y `real` en `config_retrieval.yaml`.
- El índice **no se copia de Drive**: se reconstruye desde los chunks JSON con el mismo código de la ingesta y se verifica que número, ids y textos coincidan (`verificar_indice`). Si no coincide, se borra y se reconstruye.
- Se verifica que el modelo de embeddings del corpus y el del retrieval sean el mismo.
- Umbrales por prioridad: valores fijados en el YAML → `outputs/umbrales.json` calibrados → provisionales del YAML.
- `normalizador_desde_config()` elige el mapa simulado o la tool real según `mapa_normalizacion`.

**Gold set de relevancia:**

- 40 consultas: 22 dentro del corpus (nombre exacto, sigla, sinónimo, errata, guías parecidas, hiperespecíficas) y 18 fuera.
- 64 chunks anotados con “¿responde a cuál es el tratamiento?”; 27 relevantes.
- Anotado por una IA simulando un anotador humano, **sin revisión humana**. Así consta en `gold_meta.json`.
- El notebook y `experimento_s08.py` validan que los `chunk_id` existan, que cada consulta tenga chunks relevantes y que la huella del corpus coincida.

**Resultados de retrieval (22 consultas dentro del corpus):**

| Métrica | Ingenuo (denso) | Avanzado (`intencion`) | Sistema completo + regla de siglas |
|---|---:|---:|---:|
| Tratamiento en 1.er lugar | 0.045 | 0.682 | **0.773** |
| Tratamiento en top-5 | 0.409 | 0.909 | — |
| MRR del tratamiento | 0.161 | 0.784 | **0.875** |
| Precisión de tratamiento top-5 | 0.118 | 0.491 | 0.548 |
| Guía correcta en 1.er lugar | 0.955 | 0.909 | **1.000** |
| Latencia por consulta | 13.8 ms | 340.3 ms | 415.4 ms |

**Hallazgo principal:** el RAG ingenuo casi siempre encuentra la guía correcta, pero su primer fragmento suele ser abreviaturas, glosario o índice. La mayor ganancia viene de darle al reranker la **pregunta de tratamiento**, no solo la entidad. El costo es latencia (~25×).

### 4.3 Tool de normalización terminológica

**Para qué sirve:** convertir una mención cruda (“DM2”, “HTA”, “EPOC”, “TFNA”) en un término canónico de SNOMED CT antes de consultar las guías, para reducir el desajuste léxico.

**Implementación en `M3/tools/tool_normalizacion.py`:**

- Consulta `https://data.bioontology.org/search` con `ontologies=SNOMEDCT`, `pagesize=1`, timeout de 5 s.
- API key desde `BIOPORTAL_API_KEY` en `.env` o en Colab Secrets.
- Sin API key, corre en **modo mock** con un diccionario fijo (`diabetes tipo 2`, `hipertension`); el resto se marca como fallido.
- **Guardas de plausibilidad:**
  - `_plausible()` exige que la etiqueta comparta al menos un token significativo (>2 letras, sin tildes) con la entidad;
  - si la etiqueta repite la entidad tal cual, se considera “sin aporte” y se descarta.
- Si falla, devuelve la entidad original con `normalization_failed=True`; nunca lanza una excepción no controlada.
- **No tiene caché en memoria en el código actual**: cada invocación real hace una petición HTTP. El retrieval real usa por defecto un **mapa simulado** (`normalizacion_simulada.json`), no BioPortal.

**Cuándo se invoca:** ver sección 4.4. La decisión es determinista, no la toma el LLM.

### 4.4 Orquestación: tres zonas y compuerta de evidencia

**Implementación en `M3/generacion/orquestacion.py`:**

```python
resolver_query(entidad, retrieve_fn, normalizar_fn,
               umbral, k, umbral_evidencia, forzar_por_sigla)
```

El score del primer fragmento (probabilidad del reranker, en `[0,1]`) se divide en:

| Zona | Acción |
|---|---|
| `score >= umbral` | Confiable: no se normaliza; se entregan fragmentos que superen la compuerta |
| `umbral_evidencia <= score < umbral` | Dudoso: se normaliza, se vuelve a buscar y se conserva el intento con mejor score |
| `score < umbral_evidencia` | Sin evidencia: no se entregan fragmentos; la generación responde la válvula de escape |

Reglas adicionales:

- **Siglas:** si `forzar_por_sigla=True` y la entidad tiene forma de sigla (`^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\-]{1,7}$`), se normaliza aunque el score sea alto. Motivo: el reranker asigna scores inflados a guías no relacionadas ante siglas no desambiguadas.
- **Score RRF:** si el primer fragmento tiene `score_tipo="rrf"`, se lanza un error explícito porque no es comparable con un umbral fijo. Se debe usar un modo con reranker.
- **Tolerancia a fallos:** si retrieval o normalización fallan, se captura la excepción, se registra y se degrada; no se silencia.
- **Trazabilidad:** devuelve `query_final`, `tool_invoked`, `tool_reason` y `fragments`. `tool_reason` explica si fue por sigla, por score bajo o por falta de resultados.

**Umbrales usados en la corrida real:** `0.174` para normalización y `0.174` para evidencia, calibrados con validación cruzada de 3 pliegues en `experimento_s08.py`. Los provisionales del YAML (`0.754` / `0.554`) vienen del corpus de prueba y no se usan con el corpus real.

### 4.5 Generación final

**Implementación en `M3/generacion/generacion.py`:**

- `generar_respuesta(entidad, fragments, generar_fn, pregunta)`:
  1. Si `fragments == []`, **no llama al LLM**: devuelve `"No tengo esa información en mis fuentes."`, `fallback_used=True`, `sources_used=[]`.
  2. Si hay fragmentos, construye el prompt de S07:
     - instrucción de sistema: responder SOLO con el contexto, no inventar, citar la fuente;
     - contexto con `[Fuente: ... | chunk_id]` por fragmento;
     - pregunta del caso;
     - válvula de escape literal.
  3. Llama al generador y marca `fallback_used=True` si la respuesta es la válvula.
  4. `sources_used` son los `chunk_id` que respaldan la respuesta; si es fallback, queda vacío.
- `contextos_para_ragas()` devuelve **todos** los textos recuperados; RAGAS los usa para context precision/recall.
- `GroqGenerator`:
  - `qwen/qwen3.8-27b`, `temperature=0`, `max_tokens=700`, pausa de 3 s;
  - rotación de claves `GROQ_API_KEY`, `_2`, …;
  - reintentos con espera ante `429` y rotación si se agota el cupo diario.
- `run_generacion.py`:
  - carga el eval set desde `.json` o `.jsonl`;
  - usa piezas reales (retrieval + normalización) o mocks con `--mocks`;
  - lee parámetros de orquestación del YAML de retrieval, sin duplicarlos;
  - usa la misma pregunta que el reranker (`pregunta_intencion`) salvo que el caso traiga `pregunta`;
  - escribe `M3/outputs/resultado_generacion.json` con la traza completa.

**Resultados de generación (15 casos):**

| Métrica | Valor |
|---|---:|
| Casos con fragmentos recuperados | 13/15 (86.7%) |
| Tool invocada | 5/15 (33.3%) |
| `fallback_used=true` | 10/15 (66.7%) |
| Latencia por caso | ~3-10 s |

**Lectura:** el sistema se abstiene mucho, incluso cuando recuperó fragmentos. El caso `DM2` es el ejemplo claro: recuperó 5 fragmentos de la guía de diabetes y aun así no respondió porque la pregunta pedía metas de ADA/EASD que la guía colombiana del corpus no cubre. Es una abstención conservadora, no una alucinación.

### 4.6 Evaluación RAGAS

**Para qué sirve:** medir la calidad del RAG con cuatro métricas estándar y diagnosticar en qué etapa falla.

| Métrica | Pregunta | Si falla, el problema está en… |
|---|---|---|
| Faithfulness | ¿La respuesta se infiere del contexto? | Generación |
| Context precision | ¿Los chunks recuperados son relevantes? | Retrieval |
| Context recall | ¿Se recuperó todo lo necesario? | Corpus / retrieval |
| Answer relevancy | ¿La respuesta contesta la pregunta? | Retrieval + generación |

**Implementación en `M3/ragas/evaluacion_ragas.py`:**

- **Modo real:**
  - `ragas.evaluate` con `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`;
  - LLM juez `qwen/qwen3.8-27b` vía `langchain-groq` (`temperature=0`);
  - embeddings locales `paraphrase-multilingual-MiniLM-L12-v2` mediante un wrapper de LangChain, sin OpenAI;
  - `RunConfig` con `max_workers=2`, timeout y reintentos;
  - si falla, **no sustituye** por métricas mock: lanza y deja un reporte de error.
- **Modo mock:** aproximaciones locales con embeddings o Jaccard, sin API.
- **Carga de datos:** lee `resultado_generacion.json` (`question`, `contexts`, `answer`, `ground_truth`); si no existe y el modo es real, falla en vez de evaluar el set simulado.
- **`cruce_harness.py`:** compara el F1 de extracción de M2 con las métricas RAGAS. En el código actual **no asigna diagnóstico por documento** porque no hay correspondencia entre los 59 documentos de M2 y los 15 casos de M3; devuelve `alcance="agregados_no_pareados"` y una tabla separada. La tabla histórica de 58/59 diagnósticos que aparece en `M3/README.md` §7.5 no se reproduce con este código.

**Resultados RAGAS (15 casos):**

| Métrica | Modo mock | Modo real |
|---|---:|---:|
| Faithfulness | 0.884 | **0.5973** |
| Context precision | 0.812 | **0.0833** |
| Context recall | 0.745 | **0.0000** |
| Answer relevancy | 0.831 | **0.2772** |

**Lectura:** el context recall de 0 no significa que el sistema invente; significa que el `esperado` del eval set pide recomendaciones que el corpus de 3 guías no contiene. El pipeline prefiere abstenerse. El context precision bajo indica que los fragmentos recuperados no son los que el juez considera relevantes para la respuesta de referencia.

### 4.7 Integración y estado de los checkpoints

| Checkpoint | Contrato | Estado |
|---|---|---|
| Corpus → Retrieval | Chunk / índice Chroma | Validado: 3 guías, 64 chunks |
| Normalización → Retrieval | `NormalizationResult` | Validado con mapa simulado; BioPortal real integrado pero no siempre aporta |
| Retrieval → Generación | `Fragment` / `QueryResuelta` | Validado: 13/15 con fragmentos |
| Generación → RAGAS | `contexts`, `answer`, `ground_truth` | Validado |

### 4.8 Límites de M3

- **Guías simuladas con IA:** el contenido clínico se generó a partir de guías reales, pero no es una guía oficial; ninguna conclusión clínica es extrapolable.
- **Corpus pequeño:** 3 guías y 64 chunks. Las métricas a nivel de guía están saturadas (el ingenuo acierta la guía en top-5 en 22/22).
- **Gold set anotado por IA, sin revisión humana.**
- **Normalización simulada en la evaluación:** el mapa se escribió a partir de las siglas del gold set, así que la mejora de la regla de siglas está favorecida por construcción.
- **Context recall 0:** el `esperado` no está alineado con el contenido real del corpus.
- **Abstención excesiva:** el modelo no distingue “no está en la guía” de “no lo encontré en el contexto”.
- **Dependencia de Groq/BioPortal** y de sus cuotas.
- **Auto-preferencia:** generador y juez comparten familia Qwen.

---

## 5. Módulo 4 — Componente visual y app

### 5.1 Para qué sirve

M4 permite que la nota clínica entre como **imagen** (foto o escaneo), no solo como texto. El patrón de integración es **input**: la imagen se convierte en texto al inicio y ese texto entra al mismo pipeline de M1/M2/M3. No se reescribe el NER, ni el retrieval, ni el harness.

Flujo:

```text
Imagen(es)
   │
   ▼
[M4/vision] transcribir()  ← VLM Qwen2-VL-2B, Docling o Tesseract
   │  texto + metadatos (backend, latencia, válvula, error, sha256)
   ▼
Unión de páginas por doc_id
   │
   ▼
[M2/run_inference] NER Clinical BERT + LoRA (277/50)
   │  cache {doc_id: [entidades]}
   ▼
[M3] resolver_query + retrieval + generación
   │
   ▼
[M4/app] presentación al médico
```

### 5.2 Módulo `M4/vision`

**Archivos y responsabilidades:**

| Archivo | Para qué sirve | Implementación |
|---|---|---|
| `prompts.py` | Prompt de transcripción y válvula | System: “transcriptor clínico, no diagnostiques, no interpretes, no inventes”. Instrucción literal que conserva tildes, ñ, siglas, dosis, saltos de línea. Válvula: `"No se aprecia texto legible en la imagen."` |
| `contratos.py` | Tipos y protocols | `Transcripcion` (texto, backend, segundos, válvula, error, imagen, sha256), `CasoVision`, protocols de backend |
| `backends.py` | Implementaciones de visión | `BackendVLMQwen` (Qwen2-VL-2B, `device_map="auto"`, float16 en GPU, `do_sample=False`), `BackendOCRDocling` (Docling, export Markdown, OCR español best-effort), `BackendOCRTesseract` (`spa+eng`, `psm=6`, `oem=3`), `BackendFake` para tests |
| `transcripcion.py` | `ver()` y `transcribir()` | Carga imagen local/URL/PIL; `limpiar_salida()` quita Markdown y ecos del prompt; mide latencia; calcula sha256; nunca propaga el error del backend, lo registra |
| `manifiesto.py` | Carga CSV/JSON de casos | Acepta `id`/`doc_id`, `imagen`/`image`/`path`, `pregunta`, `entidad`, `esperado`, `criterio`, `nivel`, `grupo`; filtra por nivel/grupo y rechaza ids repetidos |
| `cache.py` | Persistencia | Cache de transcripciones `doc_id -> Transcripcion`; cache NER `doc_id -> list[str]`; valida el contrato del harness |
| `ner_bridge.py` | Puente visión → NER | Carga Clinical BERT + LoRA y el chunking de M2; si la transcripción está vacía o es válvula, cachea `[]`; escribe el cache que lee el harness |
| `integracion.py` | `sistema_multimodal()` | Enchufa la transcripción a M3 mediante `responder_m3`; permite mocks marcados |
| `juez.py` | Juez visual | `juez_fidelidad_visual(afirmacion, imagen, preguntar_fn)` pide al VLM “¿Se sostiene en la imagen? SI/NO”; devuelve 1.0/0.0 |
| `run_vision.py` | `run(cfg, project_root)` + CLI | Recorre el manifiesto, transcribe, escribe cache y, con `--ner`, cache de entidades |
| `config.yaml` | Configuración | Backend por defecto `vlm`, fallback order, modelo, dtype, max tokens, OCR, rutas, NER, integración |
| `test_vision.py` | Pruebas | 31 pruebas con `BackendFake`, sin GPU ni API |

**Comportamiento clave:**

- `ver()` respeta la firma del laboratorio S13: `imagen -> (texto, None)`.
- `transcribir()` es la interfaz rica que usa el harness: devuelve texto + metadatos.
- `transcribir_auto()` solo hace fallback si un backend **falla** (dependencia ausente, GPU, binario). Si un backend funciona pero no ve texto, esa es una respuesta válida (válvula) y no prueba otro motor.
- `limpiar_salida()` elimina bloques de código, encabezados Markdown, negritas y barras de tabla, porque el NER espera texto plano.
- **No hay alucinación silenciosa:** si no hay texto legible, el texto es la válvula y el NER no recibe texto útil.

### 5.3 Set sintético de imágenes (`M4/eval_visual`)

**Para qué sirve:** generar imágenes controladas a partir de los 15 casos de M3 para medir cuánto cae el sistema cuando la nota llega como imagen.

**Implementación:**

- `render.py`: render impreso; guarda cada palabra con su posición en el texto y su caja en la página.
- `manuscrito.py`: render a mano con 16 caligrafías libres (SIL OFL 1.1, una Apache 2.0); varía tamaño, posición, presión, inclinación, ondulación y deformación de trazos; incluye tachaduras, palabras pegadas, tinta que falla y papel blanco/rayado/cuadriculado.
- `degradaciones.py`: sombra, flash, perspectiva, desenfoque, rotación/recorte, orientación, baja resolución, ruido, JPEG, movimiento, etc.
- `marcas.py`: sello, firma, mancha de café, resaltador, perforaciones, notas al margen.
- `visibilidad.py`: cada palabra lleva 15 puntos de muestra que siguen los efectos geométricos; una palabra es visible si sobrevive al menos el 50% de sus puntos (`umbral_token=0.5`).
- `generar_set.py`: orquesta todo, escribe imágenes, `manifest.json` y un CSV por nivel.
- `config.yaml`: 28 niveles en dos grupos:
  - **base:** `limpio`, `foto_realista`, `manuscrito_foto` (15 documentos);
  - **barridos:** 8 familias × 3 severidades + `orientacion` (20 documentos fijos): `manuscrito`, `sombra`, `flash`, `perspectiva`, `desenfoque`, `rotacion_recorte`, `baja_resolucion`, `marcas`.

**Reproducibilidad:** semilla global 42; cada sorteo usa una semilla derivada con SHA-256 de `(semilla, documento, página, efecto)`, independiente del orden de ejecución y del número de procesos. El manifiesto registra qué se sorteó en cada página.

**Salidas:** `<PROJECT_ROOT>/M4/data/imagenes/<nivel>/<doc_id>_p<k>.png|jpg`, `manifest.json`, `casos/<nivel>.csv`.

**Cifras de la corrida reportada:** 15 documentos × 28 niveles = **420 imágenes**; 1 página por documento en todos los niveles; 16 caligrafías. El `M4/eval_visual/README.md` conserva cifras de una corrida anterior con 59 documentos y 1.142 imágenes; para el set actual de 15 casos hay que leer el `manifest.json` generado.

### 5.4 Métricas de transcripción (`M4/metricas_transcripcion`)

**Para qué sirve:** explicar **por qué** cae el F1 cuando entra una imagen: mala lectura, texto inventado, texto que no estaba en la imagen o error del NER.

**Implementación:**

| Archivo | Función | Detalle |
|---|---|---|
| `paginas.py` | Une páginas por documento | Lee caches de `M4/vision`; identifica `doc_id` y página desde `id` o ruta; une en orden; una página con válvula/error cuenta como vacía |
| `cer_wer.py` | CER/WER | Distancia de edición con `rapidfuzz`; micro (errores totales / longitud total) y macro; normalización Unicode NFC, minúsculas, espacios colapsados; tildes se conservan; WER quita puntuación de bordes |
| `auditoria.py` | Origen de cada entidad predicha | Categorías: `acierto`, `fp_ner`, `texto_extra`, `mal_leida`, `inventada`, `fuera_de_transcripcion`; `mal_leida` si partial ratio ≥ 85 |
| `simular.py` / `validar_simulacion.py` | Validación del módulo | Inyecta errores conocidos y compara lo medido con lo inyectado; sirve para probar sin datos reales |
| `run_metricas.py` | Entry point | Calcula CER/WER, lectura de entidades, auditoría, tablas y `candidatos_juez.jsonl` |

**Dos referencias:**

- **visible:** solo el texto que quedó en la imagen (`texto_visible_por_pagina`). Es la principal.
- **completo:** todo el texto del documento. La diferencia entre ambas es lo que la imagen ocultó y no se le cobra al VLM.

**Auditoría de entidades:**

| Categoría | Regla |
|---|---|
| Acierto | Está en el gold |
| FP del NER | No está en el gold, pero sí en el texto original |
| Texto extra | No está en el original, pero sí en un sello, nota o tachadura |
| Mal leída | No está en el original y se parece (partial ratio ≥ 85) a un fragmento |
| Inventada | No está en el original ni se parece a nada |
| Fuera de transcripción | Tampoco está en la transcripción |

Las categorías `mal_leida` e `inventada` se exportan a `candidatos_juez.jsonl` para el juez visual; el módulo no llama al VLM porque requiere GPU.

### 5.5 Integración con el harness

No se modifica el harness. `M2/run_inference/run_inference.py` gana un bloque `fuente:`:

- `fuente.tipo: "texto"` → comportamiento histórico: usa `rec["text"]` del gold.
- `fuente.tipo: "vision"` → importa `M4/vision`, carga el manifiesto, transcribe cada página, une por `doc_id` y usa ese texto como entrada del NER.

El resto del pipeline (chunking 277/50, modelo, agregación por documento, formato del cache) es idéntico. Se generan dos configs dedicados:

- `M2/run_inference/config_vision.yaml`: backend, manifiesto, nivel, imágenes, `output_path`.
- `M2/harness/config_vision.yaml`: gold set, `outputs_dir`, `inference_config`.

El cache de visión y el gold deben tener exactamente los mismos `doc_id`; `cached_system` falla si falta o sobra uno.

### 5.6 App de punto de entrada (`M4/app`)

**Para qué sirve:** que un médico escriba un caso o adjunte una foto y reciba las entidades detectadas, la enfermedad principal y lo que dicen las guías, con fuentes y detalles técnicos.

**Archivos:**

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Gradio `ChatInterface` multimodal; arma la respuesta para el médico; opciones de backend, máximo de entidades y slider |
| `pipeline_app.py` | Carga las piezas y expone `transcribir`, `extraer_entidades`, `buscar`, `elegir_principal`, `redactar`, `analizar` |
| `evaluar_principal.py` | Compara la enfermedad principal elegida por la app con la `entidad` de los 15 casos de M3 |
| `config_app.yaml` | Rutas, umbrales, backend por defecto, `max_entidades=15`, `vlm_max_new_tokens=1024` |
| `requirements.txt` | Dependencias de la app (reutiliza M2 y M4/vision) |

**Flujo interno:**

1. **Lectura de imagen:** `M4/vision.transcribir()`. Con GPU usa VLM; sin GPU usa Docling (el VLM en CPU tarda minutos por página). El usuario puede cambiar el motor.
2. **NER:** Clinical BERT + LoRA con chunking 277/50; entidades en orden de aparición y sin repetir.
3. **Búsqueda de todas las entidades:** `resolver_query` de M3 (retrieval + normalización + compuerta). No llama al LLM.
4. **Elección de una enfermedad principal:** entre las entidades con al menos un fragmento sobre el umbral de evidencia, la de mayor score del reranker. Es un criterio determinista, no un diagnóstico clínico. Si ninguna pasa la compuerta, no hay principal y no se responde.
5. **Generación:** solo para la enfermedad principal, con `generar_respuesta` de M3. Si el usuario escribió una pregunta entre `¿` y `?`, se usa esa; si no, la pregunta de intención del retrieval.
6. **Presentación:** resumen con enfermedad principal, pregunta, “Qué dicen las guías”, fuentes con relevancia alta/media/baja (cortes de presentación 0.5/0.3, no calibrados), otros hallazgos plegados y “Detalles técnicos” con scores, normalización, `chunk_id` y tiempos.

**Manejo de errores:**

- Falta un recurso en `PROJECT_ROOT`: no arranca y lista qué falta.
- Imagen sin texto legible: lo informa y no inventa entidades.
- NER sin entidades: lo informa; una consulta corta se busca tal cual.
- Ningún fragmento pasa la compuerta: válvula de escape, sin llamar al LLM.
- Error del LLM en una entidad: lo muestra en esa entidad; las demás siguen.

### 5.7 Resultados de M4

**Transcripción (15 documentos):**

| Backend | Nivel | CER visible | WER visible | Entidades leídas | Inventadas | Mal leídas |
|---|---|---:|---:|---:|---:|---:|
| VLM | limpio | 0.000 | 0.001 | 100.0% | 0.0% | 0.0% |
| VLM | foto_realista | 0.003 | 0.004 | 100.0% | 0.0% | 0.0% |

**Harness (mismos 15 casos):**

| Sistema | Nivel | Exacto | Semántico | Juez |
|---|---|---:|---:|---:|
| Texto gold (techo) | — | 0.617 | 0.660 | 4.417 |
| Imagen → VLM → NER | limpio | 0.617 | 0.660 | 4.417 |
| Imagen → VLM → NER | foto_realista | 0.617 | 0.660 | 4.417 |

**El delta es cero en las tres dimensiones.** La visión no pierde calidad frente al texto en el uso realista; el límite de F1 (recall 0.468, precision 0.906) ya existe en el baseline de texto y es del NER, no de la visión. El costo es la latencia de transcripción (el harness reporta ~4.5 s por caso; la app reporta 75.7 s por página en T4; son mediciones de corridas/entornos distintos y no deben compararse directamente).

**Auditoría de entidades:** en limpio y foto realista, 32 entidades predichas por nivel: 7 aciertos, 25 falsos positivos del NER, 0 texto extra, 0 mal leídas, 0 inventadas. Los 25 FP son el mismo comportamiento que el NER tiene sobre texto limpio, no un fallo de la visión.

### 5.8 Límites de M4

- **Imágenes sintéticas:** las degradaciones son modelos simples de luz, cámara y papel; no equivalen a escaneos ni fotos reales.
- **Letra a mano tipográfica:** más regular que la de una persona y sin abreviaturas médicas reales.
- **No se evaluaron imágenes ilegibles** a simple vista, porque no se espera que un usuario las suba.
- **Un solo VLM:** Qwen2-VL-2B; no hay comparación sistemática contra un VLM más grande.
- **OCR clásico no evaluado en el harness:** el backend de producción es el VLM; Docling/Tesseract quedan como alternativas.
- **La app no usa `sistema_multimodal`** porque necesita entidades y fragmentos por separado.
- **RAGAS no se corre para visión:** el harness ya demostró delta cero; el RAG de M3 recibe el mismo texto.

---

## 6. Configuración, secretos y datos

### 6.1 Variables de entorno

| Variable | Para qué | Obligatoria |
|---|---|---|
| `PROJECT_ROOT` | Carpeta de datos (Drive o copia local) con adaptadores, corpus, gold sets e imágenes | Sí |
| `GROQ_API_KEY` | Generación M3 y juez de M2/RAGAS | Sí para modo real |
| `GROQ_API_KEY_2`, `_3`, … | Rotación de cupo | Opcional |
| `BIOPORTAL_API_KEY` | Tool de normalización SNOMED CT real | Opcional; sin ella hay mock |
| `TESSERACT_CMD` / `TESSDATA_PREFIX` | Rutas del binario/datos de Tesseract | Opcional |
| `M3_RETRIEVAL_CONFIG` | Ruta alternativa al YAML de retrieval | Opcional |
| `M3_REPO_ROOT` | Raíz del repo para resolver rutas de M3 | Opcional |

### 6.2 Estructura esperada de `PROJECT_ROOT`

```text
Proyecto-Salud/
├── M1/saved_models/clinical_bert-distemist-lora/
├── M2/eval_harness/gold_examples*.jsonl
├── M2/predictions/                 # cache NER
├── M3/data/guias_clinicas/chunks/*.json
├── M3/data/guias_clinicas/corpus_manifest.json
├── M3/data/chroma_guias/           # o se reconstruye localmente
├── M4/data/imagenes/               # set sintético
├── M4/vision/predictions/          # cache de transcripciones
└── M4/outputs/                     # métricas
```

### 6.3 Estilo de configuración

- Cada módulo tiene su propio `config.yaml`; no se duplican parámetros entre módulos.
- `M3/generacion/config.yaml` referencia los YAML de retrieval y tools en vez de copiar sus umbrales.
- `M4/app/config_app.yaml` reescribe las rutas del perfil `real` de M3 para usar `PROJECT_ROOT` y un Chroma local en `M4/app/.cache/`.
- Los umbrales del app: si existe `M3/retrieval/outputs/umbrales.json`, se usan; si no, `0.174/0.174`.

---

## 7. Cómo ejecutar el proyecto de punta a punta

### 7.1 M1

```bash
# En Colab, subir DisTEMIST a Drive y ejecutar en orden:
# 1) 01_dataset_preparation.ipynb
# 2) 02_dataset_chunking_and_splits_base.ipynb (o el de cada modelo)
# 3) 03_finetuning_base.ipynb
```

### 7.2 M2

```bash
pip install -r M2/run_inference/requirements.txt
pip install -r M2/harness/requirements.txt

cd M2/run_inference
python run_inference.py                 # escribe M2/predictions/encoder_solo.json
cd ../harness
python run_harness.py --config config.yaml
```

Para depurar:

```bash
python run_harness.py --config config.yaml --solo semantica
python run_harness.py --config config.yaml --solo judge
python run_harness.py --config config.yaml --solo scorecard
```

### 7.3 M3

```bash
# Corpus (run_corpus resuelve --config/--fuentes relativos a su propio directorio)
python M3/corpus/run_corpus.py --config config.yaml --fuentes fuentes.yaml

# Retrieval / calibración
python M3/retrieval/experimento_s08.py --corpus real

# Generación
python M3/generacion/run_generacion.py --config M3/generacion/config.yaml

# RAGAS + cruce (run_ragas resuelve --config relativo a M3/ragas)
python M3/ragas/run_ragas.py --config config.yaml --modo real
```

O todo desde `M3/ejecucion/start_pipeline_M3.ipynb` (Colab o local).

### 7.4 M4

```bash
# 1. Generar imágenes sintéticas
python M4/eval_visual/generar_set.py

# 2. Transcribir
python M4/vision/run_vision.py --config M4/vision/config.yaml \
  --manifest M4/data/imagenes/casos/limpio.csv --backend vlm

# 3. Métricas de transcripción
python M4/metricas_transcripcion/run_metricas.py \
  --transcripciones M4/vision/predictions/transcripciones_vlm_limpio.json \
  --ner M2/predictions/vision_vlm_limpio.json

# 4. Harness con visión
# run_inference resuelve --config relativo a M2/run_inference
python M2/run_inference/run_inference.py --config config_vision.yaml
python M2/harness/run_harness.py --config M2/harness/config_vision.yaml

# 5. App
python M4/app/app.py
```

Los notebooks de `M4/ejecucion/` encapsulan la misma secuencia en Colab.

---

## 8. Resumen de resultados

| Etapa | Métrica | Valor | Fuente |
|---|---|---:|---|
| M1 extracción | F1 test Clinical BERT | 0.708 | Notebook `M1/clinical_BERT/03_finetuning.ipynb` |
| M2 extracción gold | F1 exacto | 0.716 | `M2/ejecucion/outputs_gold/scorecard_m2.md` |
| M2 extracción gold | F1 semántico | 0.894 | Idem |
| M2 extracción gold | Juez 1-5 | 3.869 | Idem |
| M2 adversariales | F1 exacto | 0.421 | `M2/ejecucion/outputs_adversariales/scorecard_m2.md` |
| M3 retrieval | Tratamiento en 1.er lugar | 0.045 → 0.773 | `M3/README.md` §5.5 |
| M3 retrieval | MRR tratamiento | 0.161 → 0.875 | Idem |
| M3 generación | Fallback | 10/15 | `M3/README.md` §8.1 |
| M3 RAGAS | Faithfulness | 0.597 | `M3/README.md` §7.4 |
| M4 transcripción | CER foto realista | 0.003 | `M4/README.md` §3.3 |
| M4 harness | Delta vs texto | 0.000 | `M4/README.md` §3.5 |

---

## 9. Limitaciones, deuda técnica y discrepancias detectadas

Esta sección es importante para que el documento sirva para reproducir y para defender el proyecto con honestidad.

| # | Punto | Detalle | Impacto |
|---|---|---|---|
| 1 | M1 usó 30 documentos, no 750 | `01_dataset_preparation.ipynb` deja `max_docs=30`; salida 24/3/3. El README raíz afirma 750 | Las métricas de M1 tienen alta varianza; hay que unificar la cifra |
| 2 | `gold_examples_20.jsonl` no está versionado | M2 apunta a `M2/eval_harness/gold_examples_20.jsonl`, que no está en `main` | Para reproducir hay que copiarlo o cambiar la ruta a `M2/examples/gold_examples.jsonl` |
| 3 | Resultados M2 versionados son de 59/10 docs | `M2/ejecucion/outputs_gold` y `outputs_adversariales`; `M2/outputs` local contiene la corrida adversarial | No confundir `M2/outputs/scorecard_m2.md` (10 adversariales) con el gold de 59 |
| 4 | Rutas de M3 para Colab | El perfil `real` apunta a `/content/drive/...`; la app y el notebook las reescriben | En local hay que usar el override o `M3_RETRIEVAL_CONFIG` |
| 5 | Tool de normalización sin caché | `M3/README.md` menciona `@lru_cache` y timeout de 10 s; el código no tiene caché y usa 5 s | Cada invocación real hace HTTP; la latencia documentada de 0.05 ms no se reproduce |
| 6 | Cruce RAGAS-M2 sin diagnóstico por documento | `cruce_harness.py` devuelve `diagnosticos={}` y `alcance="agregados_no_pareados"`; la tabla 58/59 de `M3/README.md` §7.5 es histórica | No presentar 58/59 como resultado del código actual |
| 7 | `max_tokens` del corpus | `M3/corpus/README.md` muestra 350; `config.yaml` usa 500 | El valor efectivo es 500 |
| 8 | `eval_visual` con cifras de 59 docs | El README conserva 1.142 imágenes de una corrida anterior; el config actual usa 15 casos de M3 y 28 niveles | Las cifras actuales son 420 imágenes; usar el `manifest.json` generado |
| 9 | Latencias de M4 no comparables | 4.5 s/caso en sidecars del harness vs. 75.7 s/página en la app/T4 | Son corridas y entornos distintos; reportar con contexto |
| 10 | `M3/tools/orquestacion.py` duplicado | `M3/generacion/orquestacion.py` es el canónico; el de tools es un borrador | Riesgo de divergencia; conviene eliminar el duplicado |
| 11 | `categoria_seccion` no se usa en retrieval | El retrieval tiene su propio clasificador por título/embedding | No asumir que el filtro del corpus es el que usa el pipeline |
| 12 | Gold sets anotados por IA | M3 retrieval y M4 usan anotación/verificación automática; no hay revisión humana | Las métricas son indicativas, no una validación clínica |
| 13 | Guías simuladas | El corpus de M3 se generó con IA a partir de guías reales | Ninguna conclusión clínica es extrapolable |
| 14 | Auto-preferencia del juez | Generador y juez comparten familia Qwen | Faithfulness/answer relevancy pueden estar inflados |
| 15 | M2 adversariales mal planteados | Incluyen términos que no son enfermedades | La caída de F1 mide en parte una tarea incorrecta para un NER |

---

## 10. Glosario

| Término | Significado |
|---|---|
| **NER** | Named Entity Recognition; aquí, detección de menciones de enfermedad |
| **BIO** | Esquema de etiquetado `B-` (inicio), `I-` (continuación), `O` (fuera) |
| **LoRA** | Low-Rank Adaptation; adaptadores de bajo rango que se entrenan sobre un modelo congelado |
| **Chunk** | Fragmento de una guía clínica con metadatos |
| **Embedding** | Vector denso que representa el significado de un texto |
| **Chroma** | Base vectorial usada para el índice de guías |
| **BM25** | Búsqueda léxica por frecuencia de términos |
| **RRF** | Reciprocal Rank Fusion; combina rankings por posición |
| **Reranker** | Cross-encoder que reordena candidatos según la pregunta |
| **SNOMED CT** | Ontología clínica controlada |
| **BioPortal** | API que expone SNOMED CT y otras ontologías |
| **RAG** | Retrieval-Augmented Generation; recuperar contexto y generar con él |
| **RAGAS** | Framework de evaluación de RAG (faithfulness, context precision/recall, answer relevancy) |
| **VLM** | Vision-Language Model; modelo que lee imágenes y genera texto |
| **CER / WER** | Character/Word Error Rate; distancia de edición normalizada |
| **Válvula de escape** | Respuesta fija cuando no hay evidencia: “No tengo esa información en mis fuentes.” |
| **Compuerta de evidencia** | Filtro que descarta fragmentos con score por debajo del umbral |
| **Delta cero** | Diferencia nula entre dos configuraciones; en M4, imagen vs. texto |

---

## 11. Dónde profundizar

| Tema | Documento |
|---|---|
| Definición del proyecto, usuario y métricas | `README.md` |
| M1: ejecución de notebooks y estructura de datos | `M1/README.md` |
| M2: pipeline modular, YAML y ejecución | `M2/README.md`, `M2/ejecucion/README.md` |
| M3: contexto, decisiones y resultados por área | `M3/README.md` |
| M3 corpus | `M3/corpus/README.md` |
| M3 retrieval | `M3/retrieval/README.md` |
| M3 generación y orquestación | `M3/generacion/README.md` |
| M3 tool de normalización | `M3/tools/README.md` |
| M3 RAGAS | `M3/ragas/README.md` |
| M3 ejecución integrada | `M3/instrucciones.md` |
| M4 visión | `M4/vision/README.md` |
| M4 set sintético | `M4/eval_visual/README.md` |
| M4 métricas de transcripción | `M4/metricas_transcripcion/README.md` |
| M4 app | `M4/app/README.md` |
| M4 componente visual completo | `M4/README.md` |

---

*Documento generado a partir del código y la documentación de `main` en el commit `2958935`. Si se cambia la rama o se actualizan los notebooks, revisar especialmente las secciones 2.2, 3.7, 4.6 y 9, que son donde el repositorio tiene más diferencias entre lo documentado y lo ejecutable.*
