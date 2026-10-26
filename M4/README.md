# M4 - Componente visual: notas clínicas escaneadas o fotografiadas → texto → entidades → guías

Este módulo agrega al sistema la capacidad de recibir una nota clínica como **imagen**
(escaneo o foto), transcribirla con un modelo de visión-lenguaje (VLM) y pasarla al pipeline
existente: chunking → encoder clínico → entidades de enfermedad → retrieval de fragmentos de
guía. Patrón de integración: **input**.

---

## Ejecución

Los notebooks que producen las cifras de este README están en `M4/ejecucion/` (se corren en
Google Colab, en el orden de abajo). Cada uno encadena con el siguiente a través de artefactos en
Drive (`PROJECT_ROOT/M4/data/`, `M4/vision/predictions/`, `M2/predictions/`, `M2/outputs*/`).

| Orden | Notebook | Responsabilidad | Dónde quedan los resultados |
|---|---|---|---|
| 1 | `M4/ejecucion/start_eval_visual.ipynb` | Genera las imágenes sintéticas de las notas clínicas con degradaciones realistas (sombra, desenfoque, foto de cámara, manuscrito) que sirven de entrada al modelo de visión. | Celda 7 (muestra) y celda 12 (set completo): imágenes y `manifest.json` en `M4/data/imagenes/` |
| 2 | `M4/ejecucion/start_vision.ipynb` | Corre el VLM para transcribir cada imagen a texto. Solo ejecución, sin métricas. | Caches de transcripción en `M4/vision/predictions/` |
| 3 | `M4/ejecucion/start_metricas.ipynb` | Mide la calidad de la transcripción del VLM (CER, WER, lectura de entidades, auditoría de inventadas). | Celda 18: `M4/outputs/metricas_transcripcion/resumen.md` |
| 4 | `M4/ejecucion/harness/start_inference_harness.ipynb` | **Baseline:** corre el harness (3 dimensiones) sobre el **texto** gold, sin imágenes. Es el techo de referencia. | Celda 16: `M2/outputs/scorecard_m2.{json,md}` |
| 5 | `M4/ejecucion/harness/start_inference_harness_vision.ipynb` | Corre el harness sobre el texto que el VLM extrajo de **imágenes limpias**. | Celda 16: `M2/outputs_vision/limpio_vlm/scorecard_vision.{json,md}` |
| 6 | `M4/ejecucion/harness/start_inference_harness_vision_realista.ipynb` | Igual que el anterior pero sobre la **foto realista** (nivel `foto_realista`). | Celda 16: `M2/outputs_vision/foto_realista_vlm/scorecard_vision.{json,md}` |

Los tres notebooks de harness (4, 5 y 6) evalúan **el mismo texto base** en tres formatos: texto
plano (techo), imagen limpia e imagen fotografiada. Los gold `gold_examples_20.jsonl`,
`gold_m4_limpio.jsonl` y `gold_m4_foto_realista.jsonl` tienen **el mismo contenido**: cambia el
nombre y el formato de la entrada, no los documentos ni sus entidades. Comparten el harness de M2
sin modificarlo; solo cambia la fuente del texto que entra al NER (ver sección 3.2).

> **Estado de la corrida reportada (26-10-2026, rama `LuisM4`, Colab GPU T4).** Los números de las
> secciones 3.2–3.5 salen de esta tanda. Como los tres gold comparten el mismo texto base, los
> deltas entre el baseline de texto y las corridas de visión son directamente comparables.

---

## 1. Componente visual implementado


### 1.1 Qué hace y dónde se conecta

El componente recibe una o varias imágenes de una nota clínica y las convierte en el texto que
alimenta el resto del sistema. No reemplaza ninguna pieza de M1–M3: es un adaptador de entrada.

Flujo:

1. **Manifiesto.** Cada fila trae `id` (`<doc_id>__p<k>` para una página), `imagen`, `pregunta` y,
   si existe, `entidad`. El módulo carga la imagen con `cargar_imagen()`.
2. **Transcripción.** `transcribir()` llama al backend elegido (`vlm`, `ocr_docling`,
   `ocr_tesseract` o `auto`) y devuelve una `Transcripcion` con `texto`, `backend`, `segundos`,
   `valvula`, `error`, `imagen` y `sha256`. Si el backend no ve texto, el texto es la válvula
   literal y `valvula=True`.
3. **Unión de páginas.** Antes del NER, las transcripciones de un mismo `doc_id` se unen en orden
   de página. Eso ocurre en `M2/run_inference/run_inference.py` para la ruta del harness y en
   `M4/metricas_transcripcion/paginas.py` para las métricas de Luis. Una entidad puede quedar
   partida entre dos páginas; unirlas evita perderla.
4. **NER de M1/M2.** El texto unido entra al mismo Clinical BERT + LoRA y al mismo chunking de M2
   (ventanas de 277 palabras con 50 de solapamiento). La visión no reimplementa el NER: solo le
   cambia la fuente del texto.
5. **Entidades → M3.** El cache `{doc_id: [entidades]}` lo lee el harness de M2 o, en la app, las
   entidades se pasan a la orquestación de M3 (`resolver_query`) y a la generación.

En `run_inference/` el punto de inserción es `_fuente_texto()` de
`M2/run_inference/run_inference.py`: con `fuente.tipo: "vision"` importa `M4/vision`, transcribe
el manifiesto, une páginas por `doc_id` y devuelve `textos_por_doc`; el resto de `run()` (NER,
chunking y cache) queda igual. La configuración dedicada es
`M2/run_inference/config_vision.yaml`; el harness la lee desde
`M2/harness/config_vision.yaml`.

La ruta independiente `M4/vision/run_vision.py` produce el cache de transcripciones (para CER/WER y
auditoría) y, con `--ner`, el cache de entidades. La app de Pau usa `transcribir()` directamente
(sección 1.6). En ningún caso hay un notebook aparte: el módulo se importa.

### 1.2 Modelos y parámetros

**VLM.** `Qwen/Qwen2-VL-2B-Instruct`, cargado localmente con `transformers`
(`Qwen2VLForConditionalGeneration` + `AutoProcessor`). Se usa `device_map="auto"`,
`torch_dtype=float16` en GPU y `float32` en CPU, `do_sample=False` y `max_new_tokens=512` en
`M4/vision/config.yaml`. La app sube el límite a 1024 (sección 1.6). No usa API externa: el costo
es tiempo de GPU/CPU, no tokens de una API.

**Baseline OCR.** Dos backends, ambos con la misma interfaz:

- **Tesseract 5.4.0 (build 20240606)** vía `pytesseract`, `lang=spa+eng`, `psm=6`,
  `oem=3`. Para instalaciones fuera del `PATH` se configuran `TESSERACT_CMD` y
  `TESSDATA_PREFIX` (usado en la validación local). Es el baseline clásico que se ejecutó.
- **Docling** (`docling>=2.0.0`, `DocumentConverter`, export a Markdown) con OCR en español
  best-effort. Está implementado, pero **no se ejecutó** en la validación local; queda como
  segundo baseline para comparar. La app de Pau lo usa cuando no hay GPU (sección 1.6); esta
  validación local no cubrió Docling, solo Tesseract.

**Parámetros del set.** El manifiesto se filtra por `nivel` (`limpio`, `sombra_severo`, etc.) y
opcionalmente por `grupo` (`nucleo`/`retos`). La muestra versionada
`M4/eval_set/ejemplos_finales/ejemplos.csv` combina niveles y repite `id`; sin `--nivel` el
cargador falla con un mensaje explícito. El set completo generado fuera de git usa
`M4/data/imagenes/casos/<nivel>.csv`.

**Latencia y costo (medidos).** En la corrida de `run_inference.py` sobre la muestra `limpio`
(3 páginas, Windows, CPU), Tesseract tardó **1.732 s de promedio** (2.705, 1.394 y 1.097 s por
página); la corrida independiente de `run_vision.py` dio **1.1–2.4 s** por página. Fuente:
corrida local del 2026-10-20, `vision_meta.json` (no versionado). El VLM no se ejecutó en local: la
app reporta **75.7 s por página en T4** (fuente: sección 1.6, medición de la app). No hay costo de
API para la visión; el VLM requiere GPU y el OCR corre en CPU.

### 1.3 Prompt de transcripción

El prompt vive en `M4/vision/prompts.py`. Los backends OCR no usan prompt (no son generativos);
el VLM recibe dos textos:

**System:**

> Eres un transcriptor clínico. Transcribes literalmente lo que se ve en la imagen. No
> diagnosticas, no interpretas, no resumas y no inventas datos.

**Instrucción:**

> Transcribe literalmente el texto visible en la imagen, en español. Conserva tildes, la letra ñ,
> siglas, abreviaturas, números, dosis y saltos de línea, en el mismo orden en que aparecen. No
> interpretes, no resumas, no corrijas y no agregues información que no esté en la imagen.
> Devuelve únicamente el texto transcrito, sin comentarios ni explicaciones. Si la imagen no
> contiene texto legible, responde exactamente: "No se aprecia texto legible en la imagen."

**Válvula.** `MENSAJE_SIN_TEXTO = "No se aprecia texto legible en la imagen."`. Si el VLM responde
esa frase (o el OCR devuelve texto vacío), `transcribir()` marca `valvula=True`, deja ese mensaje
literal y el NER no recibe texto útil. `PATRON_VALVULA` solo acepta la negativa al inicio, para no
confundir una nota real que mencione "no se aprecia..." dentro de una frase clínica.

Esta es la única versión del prompt que está versionada; no se hicieron barridos de variantes. El
límite de salida del VLM es 512 tokens en el módulo y 1024 en la app (sección 1.6), para no cortar
el final de una página larga.

### 1.4 Interfaz, configuración y estructura de archivos

**Interfaz.** `M4/vision/run_vision.py` expone `run(cfg, project_root) -> dict`. Además:
`ver(imagen) -> (texto, None)`, `transcribir(imagen, ...) -> Transcripcion`,
`sistema_multimodal(pregunta, imagen, ...)` y `crear_responder_m3(...)`.

**`config.yaml`.** Bloques:

- `vision`: backend por defecto (`vlm`), `fallback_order`, backends de `--comparar`, instrucción.
- `vlm`: `model_id`, `device`, `torch_dtype`, `max_new_tokens`.
- `ocr.docling` / `ocr.tesseract`: export, idioma, `psm`/`oem` y rutas del binario/tessdata.
- `rutas`: manifiesto y carpeta de imágenes, `nivel`/`grupo`, caches de transcripciones/NER,
  salida y comparación.
- `ner`: si se activa el puente NER, checkpoint, adaptador LoRA y chunking.
- `integracion`: si usa mocks de M3, config de M3 y máximo de entidades.

**Archivos nuevos (`M4/vision/`):**

```text
M4/vision/
├── prompts.py          # instrucción literal + válvula
├── contratos.py        # Transcripcion, CasoVision, Protocols
├── backends.py         # Qwen2-VL, Docling, Tesseract, fake
├── transcripcion.py    # ver(), transcribir(), carga y limpieza
├── manifiesto.py       # CSV/JSON, filtro nivel/grupo
├── cache.py            # cache de transcripciones y NER
├── ner_bridge.py       # visión -> Clinical BERT + LoRA
├── integracion.py      # sistema_multimodal() + responder_m3
├── juez.py             # primitivo de fidelidad visual
├── demo.py             # nota sintética marcada
├── run_vision.py       # run() + CLI
├── config.yaml
├── requirements.txt
├── start_vision.ipynb
├── test_vision.py      # 31 pruebas
├── __init__.py
└── README.md
```

**Config de integración (M2).** `M2/run_inference/config_vision.yaml` añade
`fuente.tipo: "vision"`, `fuente.nivel`, `fuente.grupo`, `fuente.imagenes_dir` y un
`output_path` por nivel. La muestra versionada se usa con `nivel: "limpio"`; para el set completo
se cambia a `M4/data/imagenes/casos/<nivel>.csv` y `imagenes_dir: M4/data/imagenes`. El
`M2/harness/config_vision.yaml` apunta a ese config de inferencia.

### 1.5 Integración con el cache de predicciones y el harness

Hay dos caches, con propósitos distintos:

1. **Transcripciones.** `M4/vision/predictions/transcripciones_<backend>_<nivel>.json`:
   `id -> Transcripcion` (`texto`, `backend`, `segundos`, `valvula`, `error`, `imagen`,
   `sha256`). Puede ser por página (`ex_45__p1`) o por documento. Lo usa Luis para CER/WER y
   auditoría, y `M4/vision/ner_bridge.py` para producir entidades.
2. **Entidades NER.** `M2/predictions/vision_<backend>_<nivel>.json`: `doc_id -> list[str]`.
   Es exactamente el formato de `M2/harness/cached_system.sistema_desde_cache`. Al lado queda
   `vision_<backend>_<nivel>.vision_meta.json` con backend, latencias, válvulas y errores por
   documento.

**Cómo se produce.**

- `M4/vision/run_vision.py` escribe el cache de transcripciones y, con `--ner`, el de entidades.
- `M2/run_inference/run_inference.py --config config_vision.yaml` hace la ruta del harness:
  transcribe el manifiesto, une páginas por `doc_id`, corre el mismo NER y escribe el cache de
  entidades más el sidecar. No reimplementa el NER ni el harness.

**Cómo lo lee el harness.** `M2/harness/config_vision.yaml` apunta con
`rutas.inference_config` al config de `run_inference`; de ahí `gold_loader.resolver_rutas` obtiene
`output_path`. `cached_system.sistema_desde_cache` exige que el cache y el eval set tengan
exactamente los mismos `doc_id`, y que cada valor sea `list[str]`; si falta o sobra un `doc_id`,
falla con un mensaje explícito. El harness corre exact-match, similitud semántica y juez con esas
entidades; el delta lo calcula Isabella (sección 3.5).

**Regla de `doc_id`.** El `id` del manifiesto es por página (`ex_45__p1`); el `doc_id` del gold es
`ex_45`. La unión de páginas ocurre antes del NER para que el cache quede con `ex_45` y para no
perder entidades partidas entre páginas.

**Verificación local (prueba de humo).** Sobre la muestra `limpio` (3 documentos) y OCR Tesseract:
`run_inference.py` procesó 3/3 documentos, 0 errores y 0 válvulas, con 1.732 s de latencia media;
el cache quedó en `M2/predictions/vision_ocr_tesseract_limpio.json` (no versionado). El
harness `--solo exact` leyó ese cache y dio precision/recall/F1 = 0.8261
(19 TP / 4 FP / 4 FN). Es una muestra de 3
documentos para validar el cableado, **no** el delta oficial de M4; el set completo (20/59
documentos) lo corre Isabella/Luis. Fuente: corrida local del 2026-10-20.

### 1.6 Punto de entrada del sistema

El punto de entrada es una app de chat en Gradio (M4/app/). La pensamos para un médico: escribe el caso o adjunta la foto de la nota, y recibe los hallazgos clínicos con lo que dicen las guías sobre cada uno. 

**Entradas aceptadas**
 
- Texto libre: un caso clínico o una consulta corta (por ejemplo, "DM2").
- Una o varias imágenes (PNG, JPG, WEBP, BMP, TIFF). Varias imágenes se tratan como páginas de la misma nota, en el orden en que se adjuntan, igual que en el set de evaluación.
- Texto e imagen juntos: el texto escrito se suma al texto leído de la imagen.
- Los archivos que no son imágenes se ignoran y la app lo avisa.

**Flujo interno** (`M4/app/pipeline_app.py`):
 
1. **Lectura de la imagen** con `M4/vision` (`transcribir`). Con GPU usa el VLM (Qwen2-VL-2B); sin GPU usa OCR con Docling, porque el VLM en CPU tarda minutos por página. El usuario puede cambiar el motor en "Opciones".
2. **NER** con el Clinical BERT y el adaptador LoRA de M1, con el mismo chunking de M2 (ventanas de 277 palabras con 50 de traslape). Las entidades se devuelven en orden de aparición y sin repetir.
3. **Búsqueda de todas las entidades** con la orquestación de M3 (`resolver_query`): retrieval
   híbrido con reranker sobre las guías, normalización terminológica cuando el score es bajo y
   compuerta de evidencia. Este paso no llama al LLM.
4. **Elección de una enfermedad principal.** El NER detecta todas las menciones de enfermedad, que
   es como se entrenó y como se evalúa en M2. Pero, como en el eval set de M3, la respuesta se da
   sobre una sola. Se elige con un criterio determinista: entre las entidades con al menos un
   fragmento sobre el umbral de evidencia, la de mayor score del reranker. Es la enfermedad sobre
   la que las guías pueden responder mejor, no un diagnóstico decidido por el sistema. Si ninguna
   pasa la compuerta, no hay principal y la app no responde.
5. **Generación** con el LLM externo de M3 (`generar_respuesta`), solo para la enfermedad principal
   y con sus fragmentos. Si el usuario escribió una pregunta (entre `¿` y `?`), se responde esa
   pregunta, como la columna `pregunta` del eval set de M3. Si no, se usa la pregunta de intención
   del retrieval ("¿Cuál es el tratamiento de …?").

**Generación final con LLM externo.**: Groq con `qwen/qwen3.8-27b`, temperatura 0, hasta 700 tokens y 3 s de pausa entre llamadas por los límites de la API (`M3/generacion/config.yaml`). El LLM solo se llama si al menos un fragmento supera el umbral de evidencia; si no, se devuelve ("No tengo esa información en mis fuentes.") sin llamar al modelo. Usamos los umbrales de la calibración final de M3.

**Salida mostrada.** La organizamos para que el médico vea primero lo útil:
 
- Un resumen con la enfermedad principal del caso.
- El texto leído de la imagen dado el caso, plegado, para que se pueda verificar la lectura.
- La enfermedad principal con su nombre normalizado (por ejemplo, "DM2" aparece como "Diabetes
  mellitus tipo 2"), la pregunta respondida, "Qué dicen las guías" (la respuesta del LLM) y sus
  fuentes. Cada fuente muestra guía, sección y relevancia (alta, media o baja), y se despliega con
  el texto del fragmento. La relevancia sale del score del reranker con cortes en 0.5 y 0.3; esos
  cortes son solo de presentación, no están calibrados
- Los otros hallazgos detectados en la nota, plegados, indicando cuáles tienen fragmentos en las
  guías.
- Si ninguna enfermedad tiene respaldo, la lista de guías disponibles y una nota que explica que el
  sistema no responde para no inventar.
- "Detalles técnicos", plegado: motor y tiempo de lectura por página, scores, si se normalizó la
  consulta, `chunk_id` y tiempo por hallazgo.
Se buscan hasta 15 entidades por mensaje (configurable). Buscar una tarda unos 0,35 s, y el LLM se
llama una sola vez por caso. Lo que más tarda es leer la imagen: 75,7 s por página con el VLM en
una T4 (medido en Colab).
 
`M4/app/evaluar_principal.py` mide cuántas veces la enfermedad que elige la app coincide con la
`entidad` de los 15 casos de `M3/ejecucion/eval_set_casos.json`. Resultado: [COMPLETAR – correr el
script y poner "X de 15", con la fecha de la corrida].
 
**Manejo de errores**
 
| Situación | Qué hace la app |
|---|---|
| Falta el adaptador LoRA o el corpus en `PROJECT_ROOT` | No arranca y lista qué falta y en qué ruta |
| No hay `GROQ_API_KEY` | Arranca y muestra entidades y fragmentos, sin respuesta redactada, y lo avisa |
| La imagen no tiene texto legible (válvula del VLM/OCR) | Lo informa y no inventa entidades |
| El NER no encuentra entidades | Lo informa. Si el texto es una consulta corta, la busca tal cual |
| Ningún fragmento pasa la compuerta | Válvula de escape, sin llamar al LLM |
| Error del LLM en una entidad | Lo muestra en esa entidad; las demás siguen |
 
**Cómo ejecutarla.** Se necesita un `.env` en la raíz del repo con `PROJECT_ROOT` (la carpeta
`Proyecto-Salud` de Drive) y `GROQ_API_KEY`. Dentro de `PROJECT_ROOT` deben estar el adaptador
LoRA (`M1/saved_models/clinical_bert-distemist-lora/`) y los chunks y el manifiesto del corpus
(`M3/data/guias_clinicas/`).
 
```bash
pip install -r M4/app/requirements.txt
python M4/app/app.py            # local: http://127.0.0.1:7860
python M4/app/app.py --share    # en Colab con GPU: enlace público temporal
```
 
En Colab se usa `M4/app/start_app.ipynb` donde se encuentran todas las instrucciones correspondientes.
 
**Capturas**.
 
Caso escrito con cobertura: la app identifica diabetes mellitus tipo 2 e insuficiencia cardíaca y
responde con base en las guías, citando sus fuentes.
 
![captura del caso con cobertura](https://github.com/user-attachments/assets/76ad1d05-68f7-480e-8046-7453f791e611)
 
Caso desde una imagen sin cobertura: la nota es de cirugía maxilofacial. El VLM la leyó
correctamente (75,7 s para una página, en Colab con T4) y el NER detectó 7 hallazgos. Ninguno
pertenece a las guías del corpus, así que la app no responde en vez de inventar.
 
![captura del caso sin cobertura](https://github.com/user-attachments/assets/e93cc73f-dfa0-4fce-8e23-de0426d2afdb)


### 1.7 Cómo ejecutarlo

**Dependencias.**

```bash
pip install -r M4/vision/requirements.txt
# Tesseract (Linux/Colab):
apt-get install -y tesseract-ocr tesseract-ocr-spa
# o en Windows: instalar Tesseract y definir TESSERACT_CMD/TESSDATA_PREFIX
```

Para el adaptador real de M3 (solo si se usa `sistema_multimodal` con `usar_mocks: false`):

```bash
pip install -r M3/corpus/requirements.txt -r M3/generacion/requirements.txt -r M3/tools/requirements.txt
```

Se necesita `PROJECT_ROOT` en el `.env`; `GROQ_API_KEY` solo para la generación de M3.

**Pruebas y demo.**

```bash
python M4/vision/test_vision.py                         # 31/31
python M4/vision/run_vision.py --config M4/vision/config.yaml --demo --backend fake
python M4/vision/run_vision.py --config M4/vision/config.yaml --demo --backend ocr_tesseract
```

**Muestra versionada (nivel limpio).**

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml \
  --manifest M4/eval_set/ejemplos_finales/ejemplos.csv \
  --nivel limpio --backend ocr_tesseract

python M2/run_inference/run_inference.py --config config_vision.yaml
python M2/harness/run_harness.py --config M2/harness/config_vision.yaml --solo exact
```

`config_vision.yaml` ya apunta a la muestra y a `nivel: limpio`; su `fuente.backend` por defecto es
`vlm`. Para CPU, cambiar `fuente.backend` a `ocr_tesseract`. El harness usa
`M2/eval_harness/gold_examples_20.jsonl`; hay que asegurar que exista y que cubra los `doc_id` del
manifiesto.

**Set completo y VLM.**

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml \
  --manifest M4/data/imagenes/casos/sombra_severo.csv --backend vlm
```

Para el set completo, ajustar `fuente.manifest`, `fuente.nivel`, `fuente.imagenes_dir` y
`output_path` en `M2/run_inference/config_vision.yaml`. El VLM requiere GPU (T4 en Colab); en CPU
es muy lento. La validación local se hizo con Tesseract, no con el VLM.

**Reproducibilidad.** `seed_global: 42`; el VLM usa `do_sample=False`; el OCR es determinista.
Los caches y salidas están en `.gitignore` y no se versionan.

---

## 2. Justificación del patrón de integración

### 2.1 Por qué input

La pregunta del usuario de nuestro sistema es: "tengo esta nota clínica, ¿qué dicen las guías sobre
lo que tiene el paciente?". En clase vimos que el patrón correcto es el que responde esa pregunta con
menos piezas nuevas. En nuestro caso, la imagen no es un dato extra que el sistema consulta cuando
lo necesita: **es la entrada misma**. Sin leerla no hay texto, sin texto no hay entidades y sin
entidades el RAG de M3 no tiene nada que buscar.

Esto pesa más si se piensa en cómo le llega la información al médico. Muchas veces el caso no
está en un sistema que pueda copiar y pegar: llega como una remisión impresa de otra institución,
una historia clínica en papel, una nota escrita a mano o una foto que otro médico le envió. En esos casos, pedirle al médico que transcriba la nota antes de consultar las guías es justo el trabajo que el sistema debería ahorrarle, y además es una fuente de errores. Si el sistema solo aceptara texto, quedaría por
fuera la forma más común en que el médico tiene el caso a la mano.

Ese contexto es el que guía el diseño del set de evaluación (sección 3.1): las degradaciones
simulan lo que pasa al fotografiar una nota con el celular (sombra de la mano o del teléfono,
reflejo del flash, foto en ángulo, desenfoque, baja resolución), además de sellos, firmas y letra
a mano. No es una condición ideal de escáner.
 
Por eso pusimos la visión **al inicio** del pipeline, con una sola tarea: convertir la imagen en
texto, que es lo que el sistema ya sabía procesar. A partir de ahí no construimos nada nuevo:
el texto pasa por el mismo NER de M1/M2, el mismo retrieval y la misma orquestación de M3, y se
evalúa con el mismo harness de M2.

### 2.2 Qué se habría perdido con tool y con output

| Patrón | Qué se habría perdido |
|---|---|
| Tool | **Omisiones silenciosas.** En el patrón tool, el LLM decide si llama a la herramienta que mira la imagen. Si el médico escribe "paciente con disnea, adjunto la nota" y el agente responde con lo escrito sin leer la foto, la recomendación se basa en un caso incompleto, y el médico no tiene cómo saber que la nota no se leyó. En un contexto clínico, una omisión que nadie ve es peor que un error visible.<br><br>**Respuestas que cambian de una consulta a otra.** La decisión de mirar la imagen depende del LLM y no de una regla, así que la misma foto podría leerse en una consulta y no en otra. Eso impide explicar por qué el sistema dio una recomendación, que es justo lo que evitamos en M3 al decidir la normalización con una regla determinista en lugar de function-calling (`M3/README.md`).<br><br>**Pérdida de detalle clínico.** Una herramienta devuelve un resultado corto: una clase si se usa CLIP, o un resumen si se usa un VLM. Un resumen interpreta, y en la interpretación se pierden las dosis, los valores de laboratorio, la lateralidad y sobre todo las negaciones ("niega dolor torácico"), que cambian el sentido de la nota. El NER necesita el texto literal completo.<br><br>**No se podría medir igual.** En el harness, una caída del F1 podría deberse a que el agente no llamó la herramienta o a que el VLM leyó mal, y no habría forma de separar las dos causas. |
| Output | **No resuelve el problema del médico.** El patrón output genera una imagen después de responder, por ejemplo con difusión. Los casos clínicos que llegan como foto seguirían sin poder entrar al sistema, que es la necesidad real.<br><br>**Riesgo clínico sin control.** El uso más realista sería generar un esquema del manejo recomendado o una infografía para el paciente. Pero los modelos de difusión no garantizan texto legible ni correcto dentro de la imagen: una dosis o un umbral de laboratorio mal dibujado en un esquema es un error que el médico podría seguir.<br><br>**Sin respaldo ni evaluación.** La imagen generada no se apoya en los fragmentos de la guía. No tiene fuente que citar ni válvula de escape, y el harness de entidades no puede evaluarla. El CLIP score solo mide si la imagen se parece al prompt, no si es clínicamente correcta. |

**¿Fue input la mejor opción?** Para la pregunta que responde nuestro sistema ("¿qué dicen las
guías sobre lo que tiene este paciente?"), sí: esa pregunta necesita la nota completa y leída
siempre. Pero
elegirlo también tiene costos, y conviene decirlos:

- **Siempre se lee la imagen completa,** aunque el médico solo necesite un dato. Una nota de tres
  páginas tarda unos 4 minutos en leerse (3 × 75,7 s, cálculo a partir de lo medido en la app).
- **Si el VLM lee mal, el error pasa directo al NER.** Lo mitigamos mostrándole al médico el texto
  leído, pero el sistema no lo corrige solo.
- **Solo entra el texto de la imagen.** En casos como una
  radiografía no se interpretan.
- **No se pueden hacer preguntas puntuales sobre la imagen,** como "¿cuánto da la HbA1c en esta
  foto?". El sistema extrae enfermedades y busca en las guías; no responde preguntas libres.

Si el uso principal fuera ese último (conversar con el médico sobre un documento que ya tiene
abierto), el patrón tool, con un VLM que extraiga campos puntuales, habría sido la mejor opción.
Para nuestro caso, donde la nota es la entrada y la pregunta es siempre sobre lo que tiene el
paciente, input fue la mejor opción.

### 2.3 Alternativa visual descartada y conexión con la teoría

**Por qué no CLIP sobre figuras de las guías.** Consideramos indexar con CLIP las figuras de las
guías (algoritmos de manejo, tablas) para recuperarlas junto con el texto. La descartamos por
cuatro razones:
 
1. **No responde la pregunta del usuario.** Lo que entra al sistema es la nota del paciente, no una imagen
   como de lesión o radiografias.
2. **CLIP no lee.** Mide parecido entre una imagen y una frase. Las figuras de
   una guía clínica son sobre todo texto dentro de cajas y tablas, y CLIP no extrae ese texto: solo
   diría que la figura "se parece" a una consulta.
3. **Idioma y costo.** En el lab vimos que CLIP rinde mejor con prompts en inglés, y nuestras guías
   y consultas están en español. Además, el retrieval de M3 ya cubre el contenido de las guías como
   texto (64 chunks de 3 guías), con métricas medidas.

**Parches de ViT y resolución.** El encoder visual de un VLM es un ViT: corta la imagen en
parches y cada parche se vuelve un token. CLIP (S11) reduce toda imagen a 224 × 224 y la corta en
49 parches, y a ese tamaño una página de texto es ilegible. Qwen2-VL no la reduce: la corta tal
como llega, con cada token cubriendo 28 × 28 px, así que más imagen significa más tokens. De ahí
salen dos decisiones:

- **Una imagen por página.** Una página A4 a 150 dpi son unos 2.770 tokens visuales (cálculo
  nuestro, no medido); tres páginas en una sola imagen serían unos 8.300, con mucha más memoria
  en la T4 y toda la transcripción en un solo límite de salida. Por eso la app lee página por
  página, como llegan las fotos, y une el texto en orden antes del NER.
- **Más tokens de salida.** Una página llena puede superar los 512 tokens que generaba el VLM y
  cortar el final de la nota, donde suele estar el diagnóstico. En la app subimos el límite a 1024.

**Alucinación del VLM.** Sabemos que un VLM alucina como un LLM porque es un LLM. En una nota
clínica eso es lo más peligroso: una entidad inventada termina en una recomendación de una guía. Lo
abordamos en varias capas:
 
- **Prompt de transcripción literal:** "transcribe literalmente, no diagnostiques, no interpretes,
  no corrijas" (`M4/vision/prompts.py`), con una válvula de escape: si no hay texto legible, el
  modelo debe responder exactamente "No se aprecia texto legible en la imagen.", y en ese caso no
  se extraen entidades.
- **Texto oculto medido, no supuesto.** El manifiesto del set registra qué palabras y entidades
  quedaron fuera de la imagen (sombra, recorte), para no castigar al VLM por lo que no se ve y para
  detectar cuándo escribe algo que no podía ver (sección 3.1).
- **Auditoría de entidades inventadas.** Cada entidad que sale de una transcripción se clasifica
  como acierto, falso positivo del NER, texto extra, mal leída o inventada (sección 3.3).
- **Retos con texto oculto** entre las entradas finales (sección 4.1), para ver si el modelo inventa
  o avisa.
- **En la app,** el médico ve el texto leído y la fuente de cada recomendación, y el LLM de M3 solo
  responde con fragmentos que pasaron la compuerta de evidencia.

---

## 3. Impacto medido en el harness

### 3.1 Set de evaluación

**Origen y gold.** Las imágenes se generan a partir de los 15 documentos del eval set de M3
(`M3/ejecucion/eval_set_casos.json`, casos clínicos). Cada documento se dibuja como una imagen,
impresa o escrita a mano, y luego se degrada. **El gold no cambia: sigue siendo el texto original
y sus entidades.** Cada imagen conserva el `doc_id` del eval set, el mismo que usa el harness, así
que un sistema que lea las imágenes se evalúa sin cambiar el gold.

**Cómo se genera.** `python M4/eval_visual/generar_set.py` (en Colab,
`M4/ejecucion/start_eval_visual.ipynb`, rama `LuisM4`). La configuración completa está en
`M4/eval_visual/config.yaml`. Las imágenes y el manifiesto se escriben en
`<PROJECT_ROOT>/M4/data/imagenes/`, fuera de git.

**Render.** Cada documento se renderiza como una imagen de página (600 × 849 px en la corrida
reportada). Si un documento no cupiera en una página, se paginaría, porque el VLM reduce la imagen
a su tamaño de entrada y la letra pequeña se perdería (los parches de ViT tienen un tamaño fijo);
en esta corrida los 15 documentos caben en una página cada uno.

- **Impreso:** DejaVu Sans en Colab (Arial si se genera en Windows).
- **A mano:** 16 caligrafías libres (SIL OFL 1.1 y una Apache 2.0), una por documento. Varían
  tamaño, posición y presión de cada letra, ondulación e inclinación de la línea, y deforman los
  trazos. Incluyen errores de escritura (tachaduras, palabras pegadas, tinta que falla) y papel
  blanco, rayado o cuadriculado. Ningún error cambia las palabras: una tachadura agrega una
  palabra extra y el texto original queda completo.

**Niveles.** Son 28, en dos grupos:

| Grupo | Niveles | Documentos | Para qué |
|---|---|---|---|
| Base | `limpio` (techo visual), `foto_realista` (nota impresa fotografiada con una mezcla aleatoria de efectos), `manuscrito_foto` (nota a mano fotografiada) | 15 | Delta principal frente al texto |
| Barridos | 8 familias × 3 severidades (`leve`, `moderado`, `severo`): `manuscrito`, `sombra`, `flash`, `perspectiva`, `desenfoque`, `rotacion_recorte`, `baja_resolucion`, `marcas`; más `orientacion` (90/180/270°) | 15 | Atribuir la caída del F1 a una causa concreta |

Todos los niveles usan los mismos 15 documentos del eval set, para que las severidades sean
comparables. Dentro de una familia, el mismo documento recibe los mismos sorteos (posición de la
sombra, caligrafía) en las tres severidades. Solo cambia la intensidad.

**Semillas.** La semilla global es 42 (`config.yaml`). Cada sorteo usa una semilla derivada con
SHA-256 de (semilla, documento, página, efecto), que no depende del orden de ejecución, de los
niveles pedidos ni del número de procesos. Lo sorteado en cada página queda en el manifiesto.

**Texto oculto.** Algunos efectos sacan texto de la imagen: el recorte, la sombra casi negra, el
flash que satura o la firma encima. Para no castigar al VLM por palabras que no estaban en la
imagen, cada palabra lleva 15 puntos de muestra que siguen todos los efectos geométricos. Una
palabra es visible si sobrevive al menos la mitad de sus puntos. El manifiesto trae, por
documento y nivel, el texto que quedó visible y el estado de cada entidad del gold (`visible`,
`parcial`, `oculta`). La legibilidad (desenfoque, baja resolución, letra difícil) no cuenta como
texto oculto, porque es justamente lo que se mide.

**Salidas** (en `<PROJECT_ROOT>/M4/data/imagenes/`):

- `<nivel>/<doc_id>_p<k>.png|jpg`: las imágenes.
- `manifest.json`: semilla, fuentes, versiones de las librerías, subconjuntos y un resumen por
  nivel. Por documento y nivel incluye el texto por página, el texto visible, las entidades por
  estado, la caligrafía si aplica y, por página, cada efecto con su semilla y sus parámetros.
  También guarda `texto_extra`: sellos, notas y tachaduras que están en la imagen pero no en el gold.
- `casos/<nivel>.csv`: una fila por página en el formato que lee `M4/vision`
  (`id = <doc_id>__p<k>`).

**Cifras medidas.** Fuente: `M4/ejecucion/start_eval_visual.ipynb` (celdas 10 y 12) del
26-10-2026 (Colab, DejaVu Sans, Python 3.13, 11 procesos) y su `manifest.json`, que no se versiona
y se regenera igual con la misma fuente.

- 15 documentos, 28 niveles, **420 imágenes** en 90 s.
- 1 página por documento en todos los niveles (15 páginas por nivel).
- 16 caligrafías disponibles para los niveles manuscritos.

Texto oculto por nivel (manifiesto, celda 12). En los niveles que no aparecen, ninguna palabra
quedó oculta y las 15 entidades siguen visibles:

| Nivel | Palabras ocultas | Entidades no visibles |
|---|---|---|
| `sombra_severo` | 18,57 % | 2 de 15 |
| `rotacion_recorte_severo` | 16,78 % | 2 de 15 |
| `rotacion_recorte_moderado` | 4,95 % | 1 de 15 |
| `perspectiva_severo` | 2,48 % | 1 de 15 |
| `marcas_severo` | 1,93 % | 0 de 15 |
| `foto_realista` | 0,28 % | 0 de 15 |
| `marcas_moderado` | 0,14 % | 0 de 15 |

En `sombra_severo`, cerca de una quinta parte del texto queda bajo zonas de baja luz. Si el F1
cae ahí, una parte de la caída se debe a que el texto no se ve, y así hay que reportarlo. Aun así,
solo 2 de 15 entidades del gold quedan no visibles: la mayoría del contenido clínico sigue
legible incluso en el peor nivel.

**Cómo se conecta con el harness** (tareas 1, 4 y 5a):

1. Las transcripciones de las páginas de un `doc_id` se unen en orden antes del NER (en esta
   corrida cada documento es una sola página, pero el paso se mantiene por si un documento se
   paginara).
2. Todos los niveles comparten los mismos 15 `doc_id` del eval set, que son los que exige
   `cached_system`. El techo de comparación es `limpio`.
3. Conviene reportar dos F1: contra todo el gold y contra lo visible, excluyendo las entidades `oculta`.

**Supuestos y limitaciones (no medidos).**

- Las degradaciones son sintéticas: modelos simples de luz, cámara y papel. No equivalen a
  escaneos ni fotos reales.
- La letra a mano sale de fuentes tipográficas. Es más regular que la de una persona y no tiene
  las abreviaturas ni las formas ambiguas de una nota médica real. La letra real se cubre en la
  sección 4 (tarea 3b).
- La etiqueta de dificultad de cada caligrafía (`facil`, `media`, `dificil`) se puso a ojo.
- Los umbrales de "ilegible" (flash con saturación de 0,9 o más, sombra con menos del 20 % de luz,
  palabra visible con el 50 % de sus puntos) son decisiones de diseño, no medidas.
- El set se generó en Colab con DejaVu Sans. Si se regenerara en Windows con Arial, las imágenes
  impresas cambiarían. El equipo debe usar un único set generado una vez.

### 3.2 Sistemas evaluados y cambios al harness

*Sistemas comparados (texto gold como techo, imagen → OCR → NER, imagen → VLM → NER), qué se
agregó al harness y cómo se corre.*

**Sistemas comparados.** Los tres usan el **mismo** NER (Clinical BERT + LoRA) y el mismo texto
base; solo cambia de dónde sale ese texto:

| Sistema | Fuente del texto | Notebook | Cache de predicciones |
|---|---|---|---|
| Texto gold (techo) | el texto anotado del gold set | `start_inference_harness.ipynb` | `M2/predictions/encoder_solo.json` |
| imagen → VLM → NER (limpio) | VLM sobre imágenes sin degradar | `start_inference_harness_vision.ipynb` | `M2/predictions/vision_vlm_limpio.json` |
| imagen → VLM → NER (degradado) | VLM sobre imágenes degradadas | `start_inference_harness_vision_realista.ipynb` | `M2/predictions/vision_vlm_<nivel>.json` |

No se evaluó el baseline OCR clásico (Docling/Tesseract): las tres corridas usan el backend `vlm`
(`Qwen/Qwen2-VL-2B-Instruct`). El OCR queda pendiente como fila adicional.

**Cambios al harness.** El harness de M2 **no se modificó**. La integración se hizo aguas arriba,
en `M2/run_inference/`:

- `run_inference.py` ganó un bloque `fuente:` en su config. Con `fuente.tipo: texto` (default) se
  comporta como siempre; con `fuente.tipo: vision` transcribe las imágenes del manifiesto de M4
  (reutilizando `M4/vision`) y usa ese texto como entrada del NER. El chunking, el modelo y el
  formato del cache `{doc_id: [entidades]}` son idénticos a los del baseline.
- Se agregaron dos configs dedicados: `M2/run_inference/config_vision.yaml` (produce el cache de
  visión y un sidecar `*.vision_meta.json` con backend, latencia, válvulas y errores) y
  `M2/harness/config_vision.yaml` (apunta a ese cache y escribe en `M2/outputs_vision/`).
- `scorecard.py` lee el sidecar si existe y agrega al scorecard una sección **Componente visual
  (M4)** con los hiperparámetros de la corrida. Si no hay sidecar, el scorecard es el de siempre.

El detalle de esta integración (código, pruebas, retrocompatibilidad) está en el README de M2.

**Cómo se corre.** Primero el baseline de texto (`python run_harness.py`), luego cada nivel de
visión (`python run_inference.py --config config_vision.yaml` y después
`python run_harness.py --config config_vision.yaml`). Los notebooks de `M4/ejecucion/harness/`
encapsulan esa secuencia.

### 3.3 Métricas de la transcripción

*CER y WER (normalización adoptada) y protocolo de auditoría de entidades inventadas por el
VLM.*

El F1 de la sección 3.5 dice cuánto cae el sistema con imágenes. Estas métricas dicen por qué:
porque la visión leyó mal, porque inventó texto, porque el texto no estaba en la imagen o porque
falló el NER. Se calculan con `python M4/metricas_transcripcion/run_metricas.py`, un cache de
transcripciones por nivel (de `M4/vision`) y, si existe, su cache NER.

**Preparación.** Las transcripciones de cada página se unen por `doc_id` en orden de página. El
módulo deja esa versión unida en `transcripciones_por_documento/`, con el formato que acepta
`M4/vision/ner_bridge.py`, para que el NER y el harness trabajen con los `doc_id` del gold. Una
página con válvula ("no se aprecia texto") o con error cuenta como transcripción vacía.

**CER y WER.**
- **Definición.** Distancia de edición (Levenshtein, con rapidfuzz) entre la transcripción y la
  referencia, dividida por el largo de la referencia. CER en caracteres, WER en palabras. Se
  reportan micro (errores totales sobre largo total), que es la cifra principal, y macro (promedio
  por documento).
- **Normalización**, igual en ambos lados: Unicode NFC, minúsculas, y espacios y saltos de línea
  colapsados a uno, porque el render corta las líneas en otros lugares que el original. Las tildes
  no se quitan, porque perderlas es un fallo que se quiere medir. El CER conserva la puntuación. El
  WER la quita de los bordes de cada palabra.
- **Dos referencias.** La principal es el texto **visible**, es decir, solo lo que quedó en la
  imagen según el manifiesto de la sección 3.1. La otra es el texto **completo**. La diferencia
  entre las dos es lo que la imagen ocultó, y no se le cobra al VLM.
- **Diagnósticos.**
  - CER sin diacríticos: cuánto del error son tildes y eñes.
  - Sustituciones, borrados e inserciones: son indicativos, porque hay varias alineaciones posibles.
  - Peso del texto extra (sellos, notas, tachaduras): cuánto podría inflar las inserciones.
  - CER de `limpio` sobre los mismos documentos: es la comparación justa para los barridos de 20
    documentos.
  - Desglose por dificultad de la caligrafía.

**Lectura de entidades (sin NER).** Mide, de las entidades del gold que se ven completas en la
imagen, cuántas aparecen escritas tal cual en la transcripción, y cuántas con un parecido de 85 o
más. Separa dos fallos distintos: que la visión perdió la entidad, o que el NER no la reconoció.

**Auditoría de entidades predichas.** Cada entidad que el NER sacó de una transcripción se
clasifica, comparando la cadena normalizada y completa:

| Categoría | Regla |
|---|---|
| Acierto | Está en el gold |
| Falso positivo del NER | No está en el gold, pero sí en el texto original. Se marca si el sistema sobre texto limpio comete el mismo error |
| Texto extra | No está en el original, pero sí en un sello, nota o tachadura de la imagen. No es alucinación |
| Mal leída | No está en el original y se parece (partial ratio ≥ 85) a un fragmento de él |
| Inventada | No está en el original ni se parece a nada de él. Candidata a alucinación |

Las tasas de inventadas y mal leídas se calculan sobre el total de entidades predichas. Las mal
leídas y las inventadas se exportan a `candidatos_juez.jsonl`, con una afirmación y las imágenes
del documento, para revisarlas con el juez visual.

**Validación del módulo.** Fuente: `--simular` con 12 documentos y 6 niveles, corrida local del
24-10-2026. Se inyectaron errores conocidos en el texto visible. La distancia medida dio entre el
99,6 % y el 99,9 % de las ediciones inyectadas, y las cinco categorías de entidades coincidieron
exactamente con lo inyectado. **Son datos simulados que solo validan el módulo. No son resultados
del sistema.**

**Limitaciones.**
- Mal leída o inventada depende de un umbral. En entidades de 6 caracteres o menos, un solo error
  ya cae como inventada.
- Si el VLM "corrige" algo que estaba mal escrito en el original, eso no se detecta comparando
  texto. Hace falta el juez visual.

**Juez visual.** El primitivo está en `M4/vision/juez.py`:
`juez_fidelidad_visual(afirmacion, imagen, preguntar_fn, max_new_tokens=3) -> float`. Recibe una
afirmación de `candidatos_juez.jsonl` (que genera `M4/metricas_transcripcion/run_metricas.py` para
las entidades `mal_leida` e `inventada`) y la imagen del documento; le pide al VLM:
`Afirmación: "...". ¿Se sostiene en la imagen? Responde solo SI o NO.` Devuelve 1.0 si responde
"SÍ" y 0.0 si responde "NO".

El modelo que juzga es el mismo VLM de transcripción, `Qwen2-VL-2B-Instruct`. La llamada se
inyecta como `preguntar_fn(imagen, prompt, max_new_tokens=3)`, que debe apuntar al backend `vlm`
(la app ya tiene esa función; en los tests se usa un mock), con `max_new_tokens=3` y
`do_sample=False`. No hay un modelo juez separado ni una llamada a Groq.

**Cómo se agrega.** La tasa de afirmaciones sostenidas es la suma de 1.0/0.0 sobre el total de
candidatos; se reporta por categoría (`mal_leida`, `inventada`) y por nivel. Esa agregación la hace
el módulo de métricas de Luis. En `M4/vision` está el primitivo, no el runner sobre el JSONL: la
integración completa con `candidatos_juez.jsonl` y la tabla por nivel quedan pendientes de que
existan las transcripciones reales del VLM. La prueba actual del primitivo es con un mock
(`TestJuez`), no con veredictos reales.

**Limitación.** Si el mismo Qwen2-VL transcribe y juzga, hay riesgo de auto-preferencia; la pregunta
se ancla a la imagen y no a la transcripción, y una muestra debe revisarse a mano. No se reportan
números de juez todavía.

**Resultados.** Fuente: `start_metricas.ipynb` (celda 18) del 26-10-2026,
`M4/outputs/metricas_transcripcion/resumen.md`. Backend `vlm`. Referencia: texto **visible**.
Hay transcripciones reales para dos niveles (`limpio` y `foto_realista`), 15 de 15 documentos:

| Sistema | Nivel | Docs | CER visible | WER visible | CER sin tildes | Entidades leídas | Inventadas | Mal leídas |
|---|---|---|---|---|---|---|---|---|
| vlm | limpio | 15/15 | 0.000 | 0.001 | 0.000 | 100.0 % | 0.0 % | 0.0 % |
| vlm | foto_realista | 15/15 | 0.003 | 0.004 | 0.003 | 100.0 % | 0.0 % | 0.0 % |

Origen de las entidades predichas por el NER (del mismo resumen):

| Sistema | Nivel | Predichas | Acierto | FP del NER | Texto extra | Mal leída | Inventada |
|---|---|---|---|---|---|---|---|
| vlm | limpio | 32 | 7 | 25 | 0 | 0 | 0 |
| vlm | foto_realista | 32 | 7 | 25 | 0 | 0 | 0 |

Lectura: en imágenes limpias y en la foto realista el VLM transcribe casi sin error de carácter
(CER ≤ 0.003) y **no inventa ni lee mal** ninguna entidad del gold (0 % en ambas). Las 25
"FP del NER" por nivel no son fallos de la visión: son entidades que el NER produce y que no están
en el gold, el mismo comportamiento que tendría sobre texto limpio (por eso la pérdida de F1 de la
sección 3.5 es de **recall**, no de lectura). Las cifras de CER/WER más altas que aparecen en el
notebook bajo el sistema `simulado` **no son del sistema**: son transcripciones con errores
inyectados que solo validan que el módulo de métricas mide bien (ver "Validación del módulo").

Los niveles reportados (`limpio` y `foto_realista`) cubren el uso realista esperado del sistema:
una nota impresa y una foto de cámara con iluminación y encuadre normales. El set incluye además
degradaciones más agresivas (sombra, desenfoque, manuscrito) que acotan el peor caso razonable; no
se evalúan imágenes ilegibles a simple vista, porque no se espera que un usuario suba al sistema
una foto donde ni una persona puede leer las letras.

### 3.4 Línea base de la última entrega

*Sistema de referencia, versión, valores por dimensión y archivo de resultados.*

**Sistema de referencia.** `encoder_solo`: el pipeline de M2 sobre el **texto gold** (sin visión).
Es el techo del delta de M4, porque es el límite del NER cuando la lectura de la nota es perfecta.
Modelo NER: Clinical BERT (`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`) + adaptador LoRA
`M1/saved_models/clinical_bert-distemist-lora`; chunking 277/50; semilla 42.

**Valores por dimensión.** Fuente: `start_inference_harness.ipynb` (celda 16),
`M2/outputs/scorecard_m2.{json,md}`, gold `gold_examples_20.jsonl`, 26-10-2026. Es el mismo texto
base de las corridas de visión (3.5), evaluado en formato de texto plano.

| Dimensión | Precision | Recall | F1 / score |
|---|---|---|---|
| Exact-match | 0.722 | 0.731 | 0.727 |
| Similitud semántica | 0.922 | 0.934 | 0.928 |
| LLM-as-judge (1–5) | — | — | 3.763 |
| Aciertos de dominio (sí/no) | — | — | 0.650 |

### 3.5 Resultados y delta

*Tabla por sistema, nivel de degradación y dimensión, con el delta respecto a la línea base.

Fuente: `start_inference_harness_vision.ipynb` y `start_inference_harness_vision_realista.ipynb`
(celda 16), `M2/outputs_vision/{limpio_vlm,foto_realista_vlm}/scorecard_vision.{json,md}`, gold
`gold_m4_limpio.jsonl` y `gold_m4_foto_realista.jsonl`, backend `vlm`, 26-10-2026. Son el mismo
texto base del baseline de 3.4, en formato de imagen, así que el Δ es una comparación directa.

| Sistema | Nivel | Exacto | Δ | Semántico | Δ | Juez | Δ | Fuente |
|---|---|---|---|---|---|---|---|---|
| imagen → VLM → NER | limpio | 0.617 | −0.110 | 0.660 | −0.268 | 4.417 | +0.654 | `limpio_vlm/scorecard_vision.json` |
| imagen → VLM → NER | foto_realista | 0.617 | −0.110 | 0.660 | −0.268 | 4.417 | +0.654 | `foto_realista_vlm/scorecard_vision.json` |

Detalle de la dimensión exacta (micro-PRF1, igual en ambos niveles): precision 0.906, recall
0.468, F1 0.617 (tp 29, fp 3, fn 33). El patrón es claro: **precisión muy alta, recall bajo**. El
VLM transcribe bien (3.3 confirma CER casi nulo), así que lo que el NER extrae casi siempre es
correcto (fp 3), pero se pierden entidades (fn 33). La caída de F1 frente al texto es un problema
de **recall**, no de lectura ni de alucinación.

**Por qué los dos niveles dan lo mismo.** `foto_realista` casi no degrada la transcripción respecto
a `limpio` (CER 0.000 vs. 0.003 en 3.3), así que el NER recibe prácticamente el mismo texto y
produce las mismas entidades. Las únicas diferencias reales entre las dos corridas son la latencia
(4.513 vs. 4.493 s/caso) y el test de posición del juez (7/14 vs. 6/14 empates); las métricas de
clasificación son idénticas. Es decir, la foto realista **no** añade un costo de lectura medible
sobre la imagen limpia.

**Costo de operar la visión** (sidecars `vision_vlm_*.vision_meta.json`): 15 casos, 15 OK, 0
errores, 0 válvulas en ambos niveles; latencia media **4.513 s/caso** (`limpio`) y **4.493 s/caso**
(`foto_realista`). Ninguna imagen quedó ilegible.

> **Lectura del juez.** El score sube a 4.417 (vs. 3.763 del baseline), pero los aciertos de
> dominio bajan a 0.333 (vs. 0.650) y el test de posición del juez quedó con 43–50 % de empates
> sobre estos 15 casos. Con tan pocos ejemplos el juez es inestable; el número exacto (F1 0.617)
> es el más confiable de los tres.

Los niveles `limpio` y `foto_realista` representan el uso realista esperado del sistema. El
backend OCR clásico (Docling/Tesseract) no se evaluó; sería una comparación adicional opcional,
pero el backend de producción es el VLM.

**Columna Juez (Agustín).** Se llena con los veredictos de `M4/vision/juez.py` sobre
`candidatos_juez.jsonl` (ver 3.3): proporción de afirmaciones que el VLM considera sostenidas en
la imagen, por sistema y nivel. Todavía no hay veredictos reales porque el VLM no se ha corrido
sobre el set completo; no se reportan números inventados. El primitivo está probado con un mock.

### 3.6 Hallazgos y supuestos

**Medido (con fuente):**
- Con imágenes **limpias** y con **foto realista**, el VLM transcribe casi sin error (CER 0.000 y
  0.003; WER 0.001 y 0.004; 3.3) y lee el 100 % de las entidades visibles, sin inventar ninguna.
- En ambos niveles el sistema imagen→VLM→NER baja a F1 exacto **0.617** sobre 15 documentos, con
  precisión 0.906 y recall 0.468 (`{limpio,foto_realista}_vlm/scorecard_vision.json`): la pérdida
  es de recall.
- `foto_realista` da el **mismo** resultado de clasificación que `limpio`: la foto no añade un
  costo de lectura medible sobre la imagen limpia (3.5).
- El costo de operar el VLM es **~4.5 s por documento** (4.513 limpio, 4.493 foto; sidecars),
  frente a la lectura instantánea del texto gold.

**Supuesto, no medido:**
- Que la caída de recall se debe a cómo el VLM reformatea el texto antes del NER (ver 3.7): es la
  hipótesis más consistente con los datos, pero falta localizar entidad por entidad para
  confirmarla.

### 3.7 Dónde el componente no aportó


Con los datos disponibles, el componente **no mejora** el NER respecto al texto en ninguna
dimensión de clasificación: en `limpio` y `foto_realista`, F1 exacto 0.617 vs. 0.727 del texto
(Δ −0.110) y F1 semántico 0.660 vs. 0.928 (Δ −0.268). Esto es lo esperado para el patrón *input*:
la visión no puede superar al texto, su techo **es** el texto; el objetivo es medir cuánto se
pierde al leer desde una imagen y a qué costo (~4.5 s/documento). Que `foto_realista` iguale a
`limpio` indica que el costo no viene de la fotografía en sí, sino de un paso posterior.

El hallazgo útil es **dónde** se pierde: no en la lectura (CER casi nulo, 0 inventadas) sino en el
**recall** del NER sobre el texto transcrito (fn 33 de 62 entidades). La hipótesis a verificar es
que el VLM, aunque transcribe bien, reordena o reformatea el texto lo suficiente como para que el
NER reconozca menos entidades que sobre el texto original.

**Decisión:** el componente se conserva. En el uso realista (nota impresa y foto de cámara) la
lectura es fiable (CER casi nulo, 0 entidades inventadas) y la pérdida frente al texto es acotada
y atribuible al recall del NER, no a la visión. Habilita la entrada por imagen —el objetivo del
patrón *input*— a un costo de ~4.5 s por documento. La vía de mejora es el paso de post-proceso
entre la transcripción y el NER, no la visión.

---

## 4. Casos de prueba

### 4.1 Entradas finales del sistema

Las entradas usadas en esta tanda son **imágenes sintéticas** generadas por
`M4/ejecucion/start_eval_visual.ipynb` (celdas 8 y 12) a partir de 15 casos clínicos tomados del
eval set de M3 (`eval_set_casos.json`). Ningún documento contiene datos de pacientes reales: el
texto de origen son casos redactados por el equipo y las imágenes se producen renderizando ese
texto y degradándolo de forma controlada. Fuente de las cifras: `manifest.json` del set (celda 12).

- **Volumen:** 15 documentos × 28 niveles = **420 imágenes** en `M4/data/imagenes/` (corrida del
  26-10-2026, 90 s; fuente impresa DejaVu Sans en Colab).
- **Tipos de entrada:** impreso (`limpio`), foto de nota impresa (`foto_realista`), foto de nota a
  mano (`manuscrito_foto`) y 25 niveles de barrido que aíslan una degradación (manuscrito, sombra,
  flash, perspectiva, desenfoque, rotación+recorte, orientación, baja resolución, marcas), cada
  uno en severidad leve/moderado/severo. El detalle del render y las semillas está en 3.1.
- **Gold.** El gold sigue siendo el texto original anotado; la imagen solo cambia la forma de
  entrada. Es el mismo contenido en los tres formatos (texto, imagen limpia, foto).

Estas degradaciones (sombra, desenfoque, foto de cámara, manuscrito) cubren el rango realista de
lo que un usuario subiría al sistema. No se incluyen imágenes ilegibles a simple vista, porque no
representan un uso esperado.

### 4.2 Casos propios del dominio

Casos del dominio evaluados sobre el set sintético, con el fallo esperado nombrado antes de medir
y el resultado observado después. Fuente: `metricas_transcripcion/resumen.md`, los scorecards de
visión y `manifest.json` (celda 12).

| ID | Descripción | Fallo esperado | Resultado observado | Evidencia |
|---|---|---|---|---|
| limpio | Nota impresa sin degradar (control) | Ninguno; es el techo visual | CER 0.000, WER 0.001, 0 inventadas; F1 exacto 0.617 (recall 0.468) | `resumen.md`, `limpio_vlm/scorecard_vision.json` |
| foto_realista | Nota impresa fotografiada (efectos de cámara leves) | Pequeña pérdida por iluminación/encuadre | CER 0.003, WER 0.004, 0 inventadas; F1 exacto 0.617 (idéntico a limpio); 0.28 % palabras ocultas | `resumen.md`, `foto_realista_vlm/scorecard_vision.json`, `manifest.json` |

Las dos entradas cubren el uso realista esperado: una nota impresa y una foto de cámara con
condiciones normales. El set generado incluye además niveles más agresivos (sombra, desenfoque,
manuscrito) que acotan el peor caso razonable; el manifiesto registra, por ejemplo, 18.57 % de
palabras ocultas en `sombra_severo`. No se evalúan imágenes ilegibles a simple vista, porque no
representan una entrada que un usuario subiría al sistema.

### 4.3 Análisis de los casos donde el componente falla

En los niveles evaluados (`limpio` y `foto_realista`, el uso realista del sistema), el fallo
dominante **no está en la lectura del VLM** sino en el **NER sobre el texto transcrito**:

- **Lectura del VLM (OK en limpio y foto):** CER ≤ 0.003 y 100 % de entidades leídas, 0 inventadas
  (3.3). La etapa de visión no es la que rompe en imágenes legibles; `foto_realista` iguala a
  `limpio` en todas las dimensiones de clasificación (3.5).
- **NER (aquí cae el F1):** en ambos niveles, recall 0.468 con precisión 0.906 (fn 33, fp 3;
  `*_vlm/resultado_dimension1.json`). El NER reconoce menos entidades sobre el texto que vino del
  VLM que sobre el texto gold, aun cuando el contenido es casi idéntico. Hipótesis: diferencias de
  formato/segmentación introducidas por el VLM afectan al reconocedor. **Por confirmar.**
- **Chunking / retrieval:** no hay evidencia de que rompan en estos niveles; documentos cortos
  (1 página) no activan el chunking por ventana.

Sobre si es corregible o inherente: la pérdida de recall parece **corregible**, porque la lectura
es correcta (CER casi nulo) y el problema está en el formato del texto que entra al NER, no en la
visión. La vía es trabajar el post-proceso de la transcripción antes del NER. En degradaciones con
**texto oculto** (sombra que tapa parte de la página) hay un componente **inherente**: si la
palabra no está en la imagen no hay forma de leerla; por eso 3.1 separa el F1 "contra todo el gold"
del F1 "contra lo visible".

> **Siguiente paso (Isabella, con el equipo):** localizar, caso por caso, la entidad perdida en el
> texto del VLM para confirmar que el formato de la transcripción es lo que baja el recall del NER.
**Aporte de visión (Agustín).**

| Etapa | Qué se observó | Evidencia | ¿Corregible? |
|---|---|---|---|
| Lectura OCR | Tesseract confundió `lpm` con `Ipm` en la nota sintética de humo; el resto del texto clínico salió correcto (tildes, ñ, DM2, HTA, EPOC, TFNA, dosis, HbA1c). | Corrida local del 2026-10-20, `--demo --backend ocr_tesseract`; transcripción en `M4/vision/predictions/transcripciones_ocr_tesseract.json` (no versionada). | Sí: post-proceso o un OCR/VLM mejor. No es un fallo del NER. |
| Integración | El CSV combinado de `ejemplos_finales` repite `id` entre niveles; sin `--nivel` el manifiesto es ambiguo y el cargador no puede separar niveles (antes se habría mezclado el cache). | `M4/vision/manifiesto.py` y `test_filtra_por_nivel_y_evita_duplicados`. | Sí: se eligió `--nivel`/`--grupo` y se dejó un error explícito. |
| Entorno | Si falta Docling/Tesseract, el backend devuelve `BackendError` y se registra en `error`; no hay fallback silencioso. | `test_error_del_backend_se_registra` y `run_vision.py` con `--backend ocr_docling`. | Sí: instalar la dependencia o cambiar de backend. |
| Lectura VLM | El modo de falla esperado es alucinación o corrección de texto; no se ha medido con el VLM real. | Prompt literal y válvula en `prompts.py`; juez pendiente (3.3). | Mitigable, no eliminable: prompt + válvula + juez + revisión humana. |

El VLM real no se ejecutó en la validación local; los fallos de lectura del VLM son **esperados,
no medidos**. El OCR sí se ejecutó y es la evidencia que se reporta.
