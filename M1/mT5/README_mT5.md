# Pipeline de Fine-Tuning para `google/mt5-small` (DisTEMIST)

Este directorio contiene la implementación completa, autocontenida e independiente del experimento con el modelo **`google/mt5-small`** para la tarea de extracción de menciones de enfermedades sobre el corpus clínico en español **DisTEMIST**.

---

## 1. Objetivo del Experimento

Adaptar el modelo multilingüe Sequence-to-Sequence **`google/mt5-small`** (~300M parámetros, arquitectura Encoder-Decoder) mediante fine-tuning eficiente con **LoRA (PEFT)** para resolver la tarea de detección y extracción de menciones de enfermedades en historias clínicas en español, registrando métricas reproducibles a nivel de documento para la posterior comparación a nivel de equipo.

---

## 2. Modelo Utilizado y Justificación

- **Checkpoint Base:** `google/mt5-small`
- **Arquitectura:** Transformer Encoder-Decoder (Seq2Seq LM)
- **Tokenizador:** SentencePiece Unigram multilingüe (vocabulario de 250,112 tokens)
- **Clases Hugging Face:** `AutoTokenizer`, `AutoModelForSeq2SeqLM`, `DataCollatorForSeq2Seq`, `Seq2SeqTrainer`
- **Aislamiento:** Este pipeline no utiliza clasificadores de tokens (`AutoModelForTokenClassification`), esquemas de etiquetado BIO ni cabezas lineales de clasificación por token. La tarea se formula puramente como generación de texto a texto.

---

## 3. Dataset y Política de Splits Compartidos

- **Corpus:** DisTEMIST (NER de entidades de enfermedades en español).
- **Ruta de Entrada:** `/content/drive/MyDrive/Colab Notebooks/Topicos/distemist_raw/`
  - `text_files/`: Archivos `.txt` con el texto completo de cada historia clínica.
  - `subtrack1_entities/`: Archivos `.tsv` con las menciones anotadas (offsets de inicio `off0` y fin `off1`, span y etiquetas).
- **Partición Compartida:**
  - Si el archivo `distemist_final/distemist_raw_splits` ya existe en Google Drive (generado previamente por el equipo), el **Notebook 01 lo detecta y lo carga directamente sin sobrescribirlo**, garantizando que los splits de `train`, `dev` y `test` sean 100% idénticos entre modelos.
  - Si no existe, se genera a nivel de documento completo con `seed=42` en proporciones **80% Train, 10% Dev y 10% Test**.

---

## 4. Estructura de los Notebooks y Flujo de Trabajo

```
M1/mT5/
├── 01_dataset_preparation.ipynb       ← Carga y verificación de splits documentales
├── 02_dataset_chunking_and_splits.ipynb ← Tokenización SentencePiece, chunking y formato Seq2Seq
├── 03_finetuning.ipynb                ← Baseline, Smoke Test, LoRA Fine-Tuning y Evaluación
└── README_mT5.md                      ← Esta documentación técnica
```

### Flujo de Datos:
$$\text{DisTEMIST Raw (.txt / .tsv)} \xrightarrow{\text{NB 01}} \text{distemist\_raw\_splits} \xrightarrow{\text{NB 02}} \text{distemist\_mt5\_format} \xrightarrow{\text{NB 03}} \text{LoRA Adapter + Métricas JSON}$$

---

## 5. Estrategia de Chunking y Formato Seq2Seq

### A. Ventana de Contexto (Encoder)
- **Límite máximo del Encoder:** 512 tokens.
- **Prefijo del Prompt:** `"Identifica las enfermedades mencionadas en el siguiente texto clínico: "` (~14 tokens).
- **Cálculo de `window_words`:** Basado en el percentil 95 del ratio tokens/palabra de `google/mt5-small` con margen de seguridad del 10% (`target_max_tokens = 480`).
- **Solapamiento (*Overlap*):** 50 palabras entre chunks consecutivos para asegurar que las entidades en los bordes no se fragmenten ni se pierdan.

### B. Formato de Entrada y Salida
- **INPUT:**
  `"Identifica las enfermedades mencionadas en el siguiente texto clínico: " + <texto_del_chunk>`
- **TARGET:**
  Lista de menciones separadas por `" [SEP] "` (o `"ninguna"` si el chunk no contiene entidades).
  *Ejemplo:* `neumonía adquirida en la comunidad [SEP] diabetes mellitus tipo 2`
- **Nota sobre `[SEP]`:** En mT5, `[SEP]` no es un token especial sino una secuencia de texto regular que el decoder aprende a generar y que se parsea unívocamente en evaluación con `.split("[SEP]")`.

---

## 6. Configuración de LoRA (PEFT)

- **Tipo de Tarea:** `TaskType.SEQ_2_SEQ_LM`
- **Rango ($r$):** 8
- **Factor de Escala ($\alpha$):** 16 ($\alpha = 2 \times r$)
- **Dropout:** 0.05
- **Módulos Adaptados (`target_modules`):** `["q", "v"]` (proyecciones Query y Value de las capas de autoatención del encoder, decoder y atención cruzada).
- **Bias:** `"none"`
- **Justificación Académica:** Adaptar únicamente $q$ y $v$ permite un ajuste eficiente de la atención cruzada y autoatención reduciendo drásticamente los parámetros entrenables a menos del 0.4% (~1.1M parámetros vs ~300M totales), optimizando el consumo de VRAM en GPUs estándar (T4 de 16 GB).

---

## 7. Hiperparámetros de Entrenamiento

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| **Learning Rate** | `2e-4` | Tasa óptima estándar para adaptación LoRA en modelos T5 |
| **Épocas** | `10` | Convergencia adecuada para datasets médicos pequeños |
| **Batch Size por Dispositivo** | `2` (Train) / `4` (Eval) | Ajustado al límite de memoria de GPU T4 |
| **Gradient Accumulation Steps** | `2` | **Batch efectivo = 4** |
| **Estrategia de Evaluación** | `"epoch"` | Evaluación al final de cada época sobre el conjunto de `dev` |
| **Selección del Mejor Modelo** | `load_best_model_at_end=True`, `metric_for_best_model="f1"`, `greater_is_better=True` | Retiene el checkpoint con mayor F1 en `dev` |
| **Generación en Evaluación** | `predict_with_generate=True`, `generation_max_length=64` | Generación autoregresiva real en validación y test |
| **Precisión Mixta (FP16)** | `torch.cuda.is_available()` | Aceleración en GPU T4 |
| **Semilla Global** | `42` | Fijada en Python, NumPy, PyTorch, CUDA y Trainer |

---

## 8. Metodología de Evaluación (Micro-F1 por Documento)

Para hacer comparable la salida generativa de mT5 con otros modelos sin inflar artificialmente las métricas por el solapamiento de chunks:

1. **Decodificación:** Las secuencias generadas se convierten a texto eliminando tokens especiales.
2. **Parsing:** Se extraen las menciones individuales separando por `[SEP]` y se normalizan a minúsculas sin espacios redundantes.
3. **Reagrupación Documental:** Se eliminan los sufijos `_chunkN` y se unen las menciones en un `set()` único por `doc_id` original. Esto **elimina automáticamente los duplicados producidos por el overlap**.
4. **Cálculo de TP, FP y FN por Documento:**
   - $\text{TP} = |\text{True} \cap \text{Pred}|$
   - $\text{FP} = |\text{Pred} \setminus \text{True}|$
   - $\text{FN} = |\text{True} \setminus \text{Pred}|$
5. **Métrica Global Micro-F1:**
   $$\text{Precision} = \frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FP}}, \quad \text{Recall} = \frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FN}}, \quad \text{Micro-F1} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}}$$

---

## 9. Estructura de Rutas y Persistencia en Google Drive

Todas las rutas permanentes están centralizadas bajo `/content/drive/MyDrive/Colab Notebooks/Topicos/`:

| Elemento | Ruta en Google Drive | Descripción |
|---|---|---|
| **Datos Crudos** | `distemist_raw/` | Textos `.txt` y anotaciones `.tsv` |
| **Splits Crudos** | `distemist_final/distemist_raw_splits/` | Dataset Arrow compartido (train/dev/test) |
| **Dataset mT5** | `distemist_final/distemist_mt5_format/` | Pares Seq2Seq listos para entrenamiento |
| **Adaptador LoRA** | `saved_models/mt5-distemist-lora/` | Pesos `adapter_model.safetensors`, `adapter_config.json` y tokenizador |
| **Resumen JSON** | `outputs/results_mt5.json` | Métricas de baseline, best dev, test final y configuración |
| **Gráficas** | `outputs/eval_curves_mt5.png`, `outputs/baseline_vs_finetuned_mt5.png` | Curvas de pérdida, F1 por época y comparativa |

---

## 10. Integración con Weights & Biases (W&B)

- **Proyecto:** `M1-fine-tuning`
- **Entidad:** `medical-research-ai`
- **Runs Registrados:**
  1. `baseline-mt5`: Evaluación previa de `google/mt5-small` sin fine-tuning sobre `dev`.
  2. `finetuned-mt5`: Registro de loss, métricas de dev por época y evaluación final en test.
- **Autenticación:** Mediante `!wandb login` interactivo en Colab (sin almacenar API keys en el repositorio).

---

## 11. Instrucciones de Ejecución en Google Colab

1. **Abrir en Google Colab con GPU:**
   - Seleccionar un entorno de ejecución con **GPU (T4)** (`Entorno de ejecución > Cambiar tipo de entorno > T4 GPU`).
2. **Ejecutar Notebook 01 ([`01_dataset_preparation.ipynb`](file:///d:/Clase-topicos-IA/Entrega1/M1/mT5/01_dataset_preparation.ipynb)):**
   - Montar Drive.
   - Si los splits existen, los cargará e informará. Si no existen, los generará y guardará.
3. **Ejecutar Notebook 02 ([`02_dataset_chunking_and_splits.ipynb`](file:///d:/Clase-topicos-IA/Entrega1/M1/mT5/02_dataset_chunking_and_splits.ipynb)):**
   - Construye los chunks solapados y formatea los pares `input_text` / `target_text`.
   - Revisa el reporte de longitudes de target al final de la Sección 5.
4. **Ejecutar Notebook 03 ([`03_finetuning.ipynb`](file:///d:/Clase-topicos-IA/Entrega1/M1/mT5/03_finetuning.ipynb)):**
   - Autenticar en W&B (`!wandb login`).
   - Evalúa el **Baseline**, ejecuta el **Smoke Test** de 2 pasos, realiza las **10 épocas de fine-tuning**, selecciona el mejor checkpoint en `dev`, evalúa **una única vez en `test`** y persiste todos los artefactos en Drive.
