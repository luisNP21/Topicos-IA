# Ejecutar el notebook en Google Colab

## Requisitos

- Tener en Google Drive el adaptador LoRA en:
  `MyDrive/TopicosIA/Proyecto-Salud/M1/saved_models/clinical_bert-distemist-lora/`
- Tener el gold set correspondiente al notebook elegido en:
  `MyDrive/TopicosIA/Proyecto-Salud/M2/eval_harness/`
- Crear una API key de Groq (en console.groq.com/keys) y agregarla en la ventana de **Secretos / Keys** de Colab (icono de llave  en la barra lateral izquierda):
  - **Nombre:** `GROQ_API_KEY`
  - **Valor:** tu clave de Groq (`gsk_...`)
  - Asegurarse de activar el interruptor de **Acceso de Notebook** (*Notebook access*).

## Configuracion

En `M2/harness/config.yaml` se deben establecer estas variables:

```yaml
rutas:
  gold_set: "M2/eval_harness/gold_examples.jsonl"
  outputs_dir: "M2/outputs"
  model_dir: "M1/saved_models/clinical_bert-distemist-lora"
```

Si se va correr la inferencia, establecer las variables en `M2/run_inference/config.yaml`:

```yaml
eval_set_path: "M2/eval_harness/gold_examples.jsonl"
  
output_path: "M2/predictions/encoder_solo.json"
```

Durante la ejecucion del notebook tambien se utilizan las variables
`PROJECT_ROOT` para la ruta del proyecto en Drive y `GROQ_API_KEY` para la API.

## Ejecucion

1. Cargar en Google Colab el notebook.
2. Asegurarse de tener la key `GROQ_API_KEY` configurada y activa en la ventana de Secretos de Colab.
3. Ejecutar las celdas en orden y autorizar el acceso a Google Drive cuando Colab lo solicite.
4. Verificar o ajustar en `M2/harness/config.yaml` estas variables:


5. El notebook instalará los requerimientos y ejecutará `run_harness.py` con `config.yaml`.

Al finalizar, los resultados se encontraran en `M2/outputs/` dentro de Google Drive.

---

# Configuracion vision (M4)

Para evaluar el componente visual (imagen -> OCR/VLM -> NER -> harness) se usa el
notebook `start_inference_harness_vision.ipynb`. El flujo es el mismo, pero el
texto que alimenta el NER se obtiene transcribiendo imagenes de notas clinicas en
lugar de leerlo del gold set. No se reescribe el NER ni el harness: solo cambia
la fuente del texto.

## Requisitos adicionales

Ademas de los requisitos del baseline de texto (adaptador LoRA, gold set y
`GROQ_API_KEY`), la fuente visual necesita:

- El **eval set con imagenes** en Drive:
  `MyDrive/TopicosIA/Proyecto-Salud/M4/eval_set/casos_imagen.csv`
  con una columna `id` (doc_id que coincida con el gold set) y una columna
  `imagen` (ruta de la nota). Si todavia no existe, el notebook incluye una celda
  que genera un manifiesto de **humo** con una nota sintetica (sin datos reales)
  para validar el cableado de punta a punta.
- Las dependencias de `M4/vision/requirements.txt` (Pillow, VLM Qwen2-VL y los
  backends OCR). El notebook las instala.
- Para el backend `ocr_tesseract`, el binario del sistema:
  `apt-get install -y tesseract-ocr tesseract-ocr-spa` (celda opcional). El VLM y
  Docling no lo necesitan.

## Backends disponibles

El backend se elige en una celda del notebook (variable `BACKEND`), que lo
escribe en los configs:

| Backend | Qué es | Requiere |
|---|---|---|
| `vlm` | Qwen2-VL-2B-Instruct | GPU (runtime T4) |
| `ocr_docling` | Docling (baseline OCR) | CPU; descarga modelos la 1ª vez |
| `ocr_tesseract` | Tesseract clásico | binario `tesseract-ocr` + `-spa` |
| `auto` | prueba `fallback_order` | el primero que funcione |

## Configuracion

La fuente visual usa dos configs dedicados (ya versionados), separados del
baseline para no pisar `scorecard_m2.*`:

En `M2/run_inference/config_vision.yaml` (produce el cache de entidades de
vision):

```yaml
sistema: "vision_vlm"            # cambia el sufijo segun el backend
fuente:
  tipo: "vision"                 # vision -> transcribe imagenes; texto -> gold set
  backend: "vlm"                 # vlm | ocr_docling | ocr_tesseract | auto
  vision_config: "M4/vision/config.yaml"
  manifest: "M4/eval_set/casos_imagen.csv"
output_path: "M2/predictions/vision_vlm.json"
```

En `M2/harness/config_vision.yaml` (evalua ese cache):

```yaml
rutas:
  gold_set: "M2/eval_harness/gold_examples_20.jsonl"
  outputs_dir: "M2/outputs_vision"
  inference_config: "../run_inference/config_vision.yaml"
```

> El bloque `fuente:` tambien existe en `M2/run_inference/config.yaml`, donde por
> defecto es `tipo: texto` (comportamiento historico del baseline). La fuente
> visual vive en el config aparte para no mezclar ambas corridas.

## Ejecucion

1. Cargar en Colab `start_inference_harness_vision.ipynb` con runtime **GPU T4**
   si se va a usar el backend `vlm`.
2. Confirmar `GROQ_API_KEY` activa en la ventana de Secretos de Colab.
3. Ejecutar las celdas en orden. En la seccion 4, fijar `BACKEND` y correr la
   celda que lo escribe en los configs. Si no hay eval set de imagenes, correr la
   celda 4.1 para el manifiesto de humo.
4. La seccion 5 corre `run_inference.py --config config_vision.yaml`: escribe el
   cache `M2/predictions/vision_<backend>.json` y, al lado, un **sidecar**
   `vision_<backend>.vision_meta.json` con los hiperparametros de la corrida.
5. La seccion 6 corre `run_harness.py --config config_vision.yaml`.

## Resultados

Al finalizar, en `M2/outputs_vision/` dentro de Drive quedan:

- `resultado_dimension1.json`, `resultado_dimension1b.json`,
  `resultado_dimension3.json` (las 3 dimensiones de siempre).
- `scorecard_vision.json` y `scorecard_vision.md`, que ademas de las metricas
  incluyen una seccion **Componente visual (M4)** con los hiperparametros de
  vision registrados: backend, modelo, numero de casos, transcripciones OK,
  errores de backend, valvulas (imagenes sin texto legible) y latencia media y
  total. Esa latencia y la tasa de fallos son el costo de operar la vision y van
  junto al delta de F1.

El gold set del harness sigue siendo el **texto original anotado** (es el techo).
El delta de M4 es F1(texto) vs. F1(vision); la seccion 8 del notebook lo calcula
si ya corriste el baseline de texto. Los `doc_id` del manifiesto de imagenes
deben coincidir con los del gold set: si falta o sobra alguno, el harness falla
de forma explicita en vez de reportar un F1 enganoso.