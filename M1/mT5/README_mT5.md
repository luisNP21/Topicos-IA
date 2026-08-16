# Pipeline de Fine-Tuning para `google/mt5-small` (DisTEMIST)

Este directorio contiene la adaptación oficial y estandarizada del modelo **`google/mt5-small`** sobre el corpus clínico en español **DisTEMIST**, siguiendo el pipeline experimental común de los notebooks compartidos del equipo (`01_dataset_preparation.ipynb`, `02_dataset_chunking_and_splits.ipynb` y `03_finetuning.ipynb`).

---

## 1. Objetivo del Experimento

Adaptar el modelo multilingüe Sequence-to-Sequence **`google/mt5-small`** (~300M parámetros, arquitectura Transformer Encoder-Decoder) mediante fine-tuning eficiente con **LoRA (PEFT)** para la tarea de detección y extracción de menciones de enfermedades en historias clínicas en español, manteniendo **estricta comparabilidad experimental** con los modelos basados en BERT y RoBERTa del proyecto.

---

## 2. Modelo Utilizado y Especificaciones

- **Checkpoint Base:** `google/mt5-small`
- **Arquitectura:** Transformer Encoder-Decoder (`AutoModelForSeq2SeqLM`, `Seq2SeqTrainer`)
- **Tokenizador:** SentencePiece Unigram multilingüe (vocabulario de 250,112 tokens)
- **Adaptación PEFT:** LoRA (`TaskType.SEQ_2_SEQ_LM`, `target_modules=["q", "v"]`, $r=8$, $\alpha=16$, dropout=0.05)

---

## 3. Estructura de los Notebooks

```
M1/mT5/
├── 01_dataset_preparation.ipynb       ← Carga de datos, exploración multi-tokenizador y splits reproducibles (33 celdas)
├── 02_dataset_chunking_and_splits.ipynb ← Cálculo de ventana segura mT5, chunking y formatos BIO/Seq2Seq (34 celdas)
├── 03_finetuning.ipynb                ← Skeleton multi-modelo (MODEL_KEY="mt5"), baseline, LoRA y evaluación (40 celdas)
└── README_mT5.md                      ← Esta documentación técnica
```

---

## 4. Configuración de Rutas

Cada notebook incluye la variable configurable `BASE_DIR` en la sección de setup:

```python
BASE_DIR = Path("/content/drive/MyDrive/TopicosIA")
```

A partir de `BASE_DIR` se derivan automáticamente las rutas para:
- `distemist_final/` (splits crudos y datasets procesados)
- `outputs/` (resultados JSON y gráficas)
- `saved_models/` (pesos del adaptador LoRA)

---

## 5. Cambios Específicos para mT5 y Justificaciones

| Notebook | Modificación | Justificación Técnica |
|---|---|---|
| **NB 01** | Ninguna en lógica ni metodología. Se añade `BASE_DIR` configurable. | El procesamiento crudo y la división `train/dev/test` (80/10/10, seed 42) son idénticos para todos los modelos. |
| **NB 02** | `chosen_tokenizer_name = 'mT5'` y cálculo de ventana de palabras dinámico para mT5 (`window_words`). | SentencePiece fragmenta el español de manera diferente a BPE/WordPiece; requiere su propia ventana segura para no exceder 512 tokens. Genera tanto `distemist_bert_format` como `distemist_mt5_format`. |
| **NB 03** | `MODEL_KEY = "mt5"` como valor por defecto. | Activa la configuración de `google/mt5-small` dentro del skeleton multi-modelo original. |
| **NB 03** | `fp16 = False` (FP32). | **Incompatibilidad técnica documentada de T5/mT5:** `fp16` provoca underflow numérico (pérdida NaN) en las capas `layer_norm` de T5. |
| **NB 03** | `text_target=` en lugar de `as_target_tokenizer()`. | Actualización de API deprecada en versiones recientes de Hugging Face Transformers. |

---

## 6. Hiperparámetros de Entrenamiento

Todos los hiperparámetros se mantienen **idénticos** a los del pipeline original del proyecto:

| Hiperparámetro | Valor |
|---|---|
| **Learning Rate** | `2e-4` |
| **Épocas** | `10` |
| **Batch Size (Train / Eval)** | `2` / `4` |
| **Estrategia de Evaluación** | `"epoch"` |
| **Estrategia de Guardado** | `"epoch"` |
| **Selección del Mejor Modelo** | `load_best_model_at_end=True`, `metric_for_best_model="f1"` |
| **Logging Steps** | `5` |
| **Report To** | `wandb` |
| **LoRA $r$ / $\alpha$** | `8` / `16` |
| **LoRA Target Modules** | `["q", "v"]` |
| **LoRA Task Type** | `TaskType.SEQ_2_SEQ_LM` |
| **Semilla Global** | `42` |

---

## 7. Orden de Ejecución en Google Colab

1. **`01_dataset_preparation.ipynb`**:
   - Montar Google Drive.
   - Ajustar `BASE_DIR` si la carpeta en Drive tiene otro nombre.
   - Ejecutar todas las celdas para generar `distemist_raw_splits` en `distemist_final/`.

2. **`02_dataset_chunking_and_splits.ipynb`**:
   - Cargar `distemist_raw_splits`.
   - Ejecutar todas las celdas para realizar el chunking con solapamiento (50 palabras) y generar `distemist_mt5_format` (y `distemist_bert_format`).

3. **`03_finetuning.ipynb`**:
   - Iniciar sesión en Weights & Biases (`!wandb login`).
   - Cargar `distemist_mt5_format`.
   - Evaluar baseline (sin fine-tuning) en `dev`.
   - Entrenar 10 épocas con LoRA (`google/mt5-small`).
   - Evaluar modelo fine-tuneado en `test`.
   - Generar curvas de entrenamiento y gráficos comparativos guardados localmente y en Google Drive.
