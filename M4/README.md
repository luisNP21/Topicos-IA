# M4 - Componente visual: notas clínicas escaneadas o fotografiadas → texto → entidades → guías

Este módulo agrega al sistema la capacidad de recibir una nota clínica como **imagen**
(escaneo o foto), transcribirla con un modelo de visión-lenguaje (VLM) y pasarla al pipeline
existente: chunking → encoder clínico → entidades de enfermedad → retrieval de fragmentos de
guía. Patrón de integración: **input**.

El README sigue los cuatro criterios de la rúbrica, una sección por criterio. Cada sección
indica quién la documenta.

> Convención: toda cifra que aparezca en este documento indica su fuente (archivo o run), y los
> hallazgos medidos se mantienen separados de los supuestos.

[COMPLETAR – Isabella: resumen de 4 a 6 líneas, escrito al final]

---

## 1. Componente visual implementado

*Responsables: Agustín (componente), Pau (punto de entrada), Isabella (integración con
harness y cache).*

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
  segundo baseline para comparar.

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
3. **Por cada entidad**, la orquestación de M3 (`resolver_query`): retrieval híbrido con reranker sobre las guías, normalización terminológica cuando el score es bajo y compuerta de evidencia.
4. **Generación** con el LLM externo de M3 (`generar_respuesta`), solo con los fragmentos que pasaron la compuerta.

**Generación final con LLM externo.**: Groq con `qwen/qwen3.8-27b`, temperatura 0, hasta 700 tokens y 3 s de pausa entre llamadas por los límites de la API (`M3/generacion/config.yaml`). El LLM solo se llama si al menos un fragmento supera el umbral de evidencia; si no, se devuelve ("No tengo esa información en mis fuentes.") sin llamar al modelo. Usamos los umbrales de la calibración final de M3.

**Salida mostrada.** La organizamos para que el médico vea primero lo útil:
 
- Un resumen: cuántos hallazgos se identificaron y cuántos tienen respaldo en las guías disponibles.
- El texto leído de la imagen, plegado, para que se pueda verificar la lectura.
- Por cada hallazgo con respaldo: "Qué dicen las guías" (la respuesta del LLM) y sus fuentes. Cada
  fuente muestra guía, sección y relevancia (alta, media o baja), y se despliega con el texto del
  fragmento. La relevancia sale del score del reranker con cortes en 0.5 y 0.3.
- Los hallazgos sin respaldo, en una sola línea, con las guías disponibles y una nota que explica
  que el sistema no responde para no inventar.
- "Detalles técnicos", plegado: motor y tiempo de lectura por página, scores, si se normalizó la
  consulta, `chunk_id` y tiempo por hallazgo.
Se consultan hasta 15 hallazgos por mensaje (configurable). Consultar uno tarda unos 0,35 s a hasta 75 s con consulta por medio de imagen (medido en Colab con T4).
 
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
 
[COMPLETAR – captura del caso con cobertura]
 
Caso desde una imagen sin cobertura: la nota es de cirugía maxilofacial. El VLM la leyó
correctamente (75,7 s para una página, en Colab con T4) y el NER detectó 7 hallazgos. Ninguno
pertenece a las guías del corpus, así que la app no responde en vez de inventar.
 
[COMPLETAR – captura del caso sin cobertura]


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

*Responsable de la sección: Isabella. Aportes: Luis (set de evaluación, métricas exactas y de
transcripción), Pau (métricas semánticas), Agustín (juez).*

### 3.1 Set de evaluación

*Cómo se generan las imágenes a partir de los documentos del test: niveles de degradación,
parámetros, semillas, script, ubicación y manifiesto. Aclarar que el gold sigue siendo el texto
original.*

**Origen y gold.** Las imágenes se generan a partir de los 59 documentos del gold set de M2
(`M2/examples/gold_examples.jsonl`, casos clínicos de DisTEMIST). Cada documento se dibuja como
páginas de imagen, impresas o escritas a mano, y luego se degrada. **El gold no cambia: sigue
siendo el texto original y sus entidades.** Cada imagen conserva el `doc_id` que asigna
`M2/harness/gold_loader.cargar_gold_set` (`ex_0` … `ex_58`), el mismo que usa el harness, así que
un sistema que lea las imágenes se evalúa con el harness de M2 sin cambiar el gold. Los
adversariales de M2 no se usan, porque su gold está pendiente de corrección (ver README raíz).

**Cómo se genera.** `python M4/eval_visual/generar_set.py` (en Colab,
`M4/eval_visual/start_eval_visual.ipynb`). La configuración completa está en
`M4/eval_visual/config.yaml`. Las imágenes y el manifiesto se escriben en
`<PROJECT_ROOT>/M4/data/imagenes/`, fuera de git. Versión de referencia: commit `e38ac45`.

**Render.** Página A4 a 150 dpi (1240 × 1754 px). Cada documento ocupa de 1 a 3 páginas impresas,
o de 1 a 5 si está escrito a mano. Se pagina en vez de usar una sola imagen alta porque el VLM la
reduciría a su tamaño de entrada y la letra pequeña se perdería (los parches de ViT tienen un
tamaño fijo).

- **Impreso:** Arial en Windows, DejaVu Sans en Linux/Colab.
- **A mano:** 16 caligrafías libres (SIL OFL 1.1 y una Apache 2.0), una por documento. Varían
  tamaño, posición y presión de cada letra, ondulación e inclinación de la línea, y deforman los
  trazos. Incluyen errores de escritura (tachaduras, palabras pegadas, tinta que falla) y papel
  blanco, rayado o cuadriculado. Ningún error cambia las palabras: una tachadura agrega una
  palabra extra y el texto original queda completo.

**Niveles.** Son 28, en dos grupos:

| Grupo | Niveles | Documentos | Para qué |
|---|---|---|---|
| Base | `limpio` (techo visual), `foto_realista` (nota impresa fotografiada con una mezcla aleatoria de efectos), `manuscrito_foto` (nota a mano fotografiada) | 59 | Delta principal frente al texto |
| Barridos | 8 familias × 3 severidades (`leve`, `moderado`, `severo`): `manuscrito`, `sombra`, `flash`, `perspectiva`, `desenfoque`, `rotacion_recorte`, `baja_resolucion`, `marcas`; más `orientacion` (90/180/270°) | 20, un subconjunto fijo | Atribuir la caída del F1 a una causa concreta |

Los barridos usan siempre los mismos 20 documentos (`subconjuntos.barrido` en el manifiesto)
para que las severidades sean comparables y para no multiplicar las llamadas al VLM. Dentro de
una familia, el mismo documento recibe los mismos sorteos (posición de la sombra, caligrafía) en
las tres severidades. Solo cambia la intensidad.

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

**Cifras medidas.** Fuente: corrida local de `generar_set.py` del 22-10-2026 (Windows, Arial,
Python 3.13, 11 procesos) y su `manifest.json`, que no se versiona y se regenera igual con la
misma fuente.

- 1.142 imágenes, 363 MB, 214 s.
- Páginas: 88 por nivel impreso con los 59 documentos y 31 con los 20 del barrido;
  `manuscrito_foto` 141 páginas; barridos manuscritos 46, 47 y 50.
- Entidades: 758 entidades distintas del gold en los 59 documentos y 284 en el barrido. Todas se
  localizaron en el texto.
- Reproducibilidad: dos corridas independientes dieron 27 de 27 archivos idénticos byte por byte,
  y una corrida en serie y otra en paralelo dieron 61 de 61. En ambos casos el manifiesto fue el mismo.

Texto oculto por nivel. En los niveles que no aparecen, ninguna palabra quedó oculta:

| Nivel | Palabras ocultas | Entidades visibles / parciales / ocultas |
|---|---|---|
| `sombra_severo` | 29,6 % | 164 / 68 / 52 de 284 |
| `rotacion_recorte_severo` | 13,2 % | 213 / 50 / 21 de 284 |
| `rotacion_recorte_moderado` | 2,1 % | 272 / 10 / 2 de 284 |
| `flash_severo` | 1,2 % | 280 / 4 / 0 de 284 |
| `marcas_severo` | 0,3 % | 279 / 5 / 0 de 284 |
| `marcas_moderado` | 0,3 % | 282 / 2 / 0 de 284 |
| `foto_realista` | 0,1 % | 757 / 1 / 0 de 758 |
| `perspectiva_severo` | 0,1 % | 284 / 0 / 0 de 284 |

En `sombra_severo`, casi un tercio del texto queda bajo zonas con menos del 20 % de luz. Si el
F1 cae ahí, una parte de la caída se debe a que el texto no se ve, y así hay que reportarlo.

**Cómo se conecta con el harness** (tareas 1, 4 y 5a):

1. Las transcripciones de las páginas de un `doc_id` se unen en orden antes del NER, porque una
   entidad puede quedar partida entre dos páginas.
2. En los barridos, el eval set se filtra a los 20 documentos de `subconjuntos.barrido`, porque
   `cached_system` exige los mismos `doc_id`. El techo de comparación es `limpio` restringido a
   esos mismos 20.
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
- Generado en Colab, el set usaría DejaVu Sans en vez de Arial. Las imágenes impresas cambiarían
  y el número de páginas podría variar. El equipo debe usar un único set generado una vez.
- Todavía no se ha medido si los niveles severos son demasiado fáciles o demasiado difíciles para
  el VLM. Eso sale de la sección 3.5.

### 3.2 Sistemas evaluados y cambios al harness

*Sistemas comparados (texto gold como techo, imagen → OCR → NER, imagen → VLM → NER), qué se
agregó al harness y cómo se corre.*

[COMPLETAR – Isabella]

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

**Resultados.** [COMPLETAR – Luis, cuando existan las transcripciones reales: tabla de CER, WER,
lectura de entidades e inventadas por sistema y nivel, con fuente.]

### 3.4 Línea base de la última entrega

*Sistema de referencia, versión, valores por dimensión y archivo de resultados.*

[COMPLETAR – Isabella]

### 3.5 Resultados y delta

*Tabla por sistema, nivel de degradación y dimensión, con el delta respecto a la línea base.
Cada dueño interpreta su dimensión: exacto (Luis), semántico (Pau), juez (Agustín).*

| Sistema | Nivel | Exacto | Δ | Semántico | Δ | Juez | Δ | Fuente |
|---|---|---|---|---|---|---|---|---|
| [COMPLETAR] | | | | | | | | |

**Columna Juez (Agustín).** Se llena con los veredictos de `M4/vision/juez.py` sobre
`candidatos_juez.jsonl` (ver 3.3): proporción de afirmaciones que el VLM considera sostenidas en
la imagen, por sistema y nivel. Todavía no hay veredictos reales porque el VLM no se ha corrido
sobre el set completo; no se reportan números inventados. El primitivo está probado con un mock.

### 3.6 Hallazgos y supuestos

**Medido (con fuente):** [COMPLETAR – Isabella]

**Supuesto, no medido:** [COMPLETAR – Isabella]

### 3.7 Dónde el componente no aportó

*Respuesta directa y con cifras por sistema y nivel. Incluye la decisión sobre cualquier variante
que no aporte (se retira del sistema o se conserva, y por qué).*

[COMPLETAR – Isabella, con el equipo]

---

## 4. Casos de prueba

*Responsables: Pau (casos y entradas finales), Luis (formato y estructura), Isabella (análisis
de fallos).*

### 4.1 Entradas finales del sistema

Las entradas finales son 56 imágenes seleccionadas del set sintético de la sección 3.1. Están en
`M4/eval_set/ejemplos_finales/`, con un manifiesto `ejemplos.csv` en el mismo formato que
`casos/<nivel>.csv` y dos columnas más: `grupo` y `motivo`. El script que las selecciona solo copia
del set completo; no modifica ni borra nada.
 
**Tipo.** Notas clínicas impresas y escritas a mano, "fotografiadas" o "escaneadas" con
degradaciones simuladas. Las organizamos en dos grupos que se reportan por separado, porque
mezclarlos distorsiona los promedios:
 
| Grupo | Imágenes | Qué contiene | Para qué |
|---|---|---|---|
| Núcleo comparable | 35 | Los mismos 3 documentos (`ex_45`, `ex_47`, `ex_53`) en 13 categorías: `limpio`, `foto_realista`, `manuscrito_foto`, `manuscrito_leve` y `manuscrito_severo`, `desenfoque_leve` y `desenfoque_severo` (2 documentos cada uno), y `sombra`, `flash`, `perspectiva`, `baja_resolucion` y `marcas` en nivel moderado, más `orientacion` | Comparar categorías: si el resultado cambia, es por la degradación y no porque la nota sea distinta |
| Retos | 21 | Casos difíciles que pueden pasar en la práctica: flash, perspectiva y baja resolución severos; sellos y firmas sobre el texto; sombras duras; recortes; una nota de 3 páginas con 28 entidades; letra cursiva difícil fotografiada | Ver cómo responde el sistema ante lo que no controlamos |
 
Elegimos `ex_45`, `ex_47` y `ex_53` para el núcleo porque, según el manifiesto del set, son los
únicos documentos del barrido que ocupan una sola página en las 13 categorías y no tienen ninguna
entidad oculta. Así el núcleo mide solo la legibilidad. Además, `ex_45` y `ex_47` usan una
caligrafía marcada como difícil, de modo que el núcleo también incluye letra a mano exigente.
 
Dentro de los retos hay dos con **texto oculto a propósito**: `sombra_severo` en `ex_4` y
`rotacion_recorte_severo` en `ex_22`. Sirven para ver si el modelo inventa lo que no puede leer o
activa la válvula de escape. En esos dos casos el F1 se calcula también contra el gold visible
(quitando las entidades marcadas como `oculta` en el manifiesto), y la transcripción se revisa a
mano.
 
**Procedencia.** Ninguna imagen es la foto de un documento real. Todas se generaron con
`M4/eval_visual/generar_set.py` (semilla 42, reproducible) a partir de los casos clínicos del gold
set de M2 (`M2/examples/gold_examples.jsonl`), que vienen de DisTEMIST, un corpus público de casos
clínicos publicados.
 
**Cómo se garantiza que no hay datos de pacientes reales**
 
- El texto sale de casos clínicos publicados en un corpus público, el mismo que el proyecto usa
  desde M1. No incorporamos notas de ninguna institución ni de ningún paciente.
- Los sellos usan nombres de instituciones ficticias ("HOSPITAL CLINICO SIMULADO", "CLINICA DE
  PRUEBA M4", "IPS SINTETICA"), y las firmas y notas al margen son trazos generados.
- La letra a mano sale de fuentes tipográficas libres, no de escritura de personas.
- La app muestra un aviso para que nadie ingrese datos reales de pacientes, porque el texto se envía
  a un servicio externo (Groq) para la generación.
Limitación: al ser sintéticas, estas entradas no tienen la letra, las abreviaturas ni el desorden
de una nota médica real. Lo que medimos con ellas es la robustez ante degradaciones controladas, no
el desempeño con notas reales.

### 4.2 Casos propios del dominio

*Casos clínicos propios (abreviaturas, entidades partidas entre líneas, tildes y ñ, dosis,
sellos o firmas, tablas de laboratorio, manuscrito real, entre otros). El fallo esperado se
documenta antes de ejecutar y el observado después.*

| ID | Descripción | Fallo esperado | Resultado observado | Evidencia |
|---|---|---|---|---|
| [COMPLETAR] | | | | |

### 4.3 Análisis de los casos donde el componente falla

*Etapa donde se rompe (lectura del VLM, chunking, NER, retrieval), evidencia y si el fallo es
corregible o inherente al enfoque.*

**Aporte de visión (Agustín).**

| Etapa | Qué se observó | Evidencia | ¿Corregible? |
|---|---|---|---|
| Lectura OCR | Tesseract confundió `lpm` con `Ipm` en la nota sintética de humo; el resto del texto clínico salió correcto (tildes, ñ, DM2, HTA, EPOC, TFNA, dosis, HbA1c). | Corrida local del 2026-10-20, `--demo --backend ocr_tesseract`; transcripción en `M4/vision/predictions/transcripciones_ocr_tesseract.json` (no versionada). | Sí: post-proceso o un OCR/VLM mejor. No es un fallo del NER. |
| Integración | El CSV combinado de `ejemplos_finales` repite `id` entre niveles; sin `--nivel` el manifiesto es ambiguo y el cargador no puede separar niveles (antes se habría mezclado el cache). | `M4/vision/manifiesto.py` y `test_filtra_por_nivel_y_evita_duplicados`. | Sí: se eligió `--nivel`/`--grupo` y se dejó un error explícito. |
| Entorno | Si falta Docling/Tesseract, el backend devuelve `BackendError` y se registra en `error`; no hay fallback silencioso. | `test_error_del_backend_se_registra` y `run_vision.py` con `--backend ocr_docling`. | Sí: instalar la dependencia o cambiar de backend. |
| Lectura VLM | El modo de falla esperado es alucinación o corrección de texto; no se ha medido con el VLM real. | Prompt literal y válvula en `prompts.py`; juez pendiente (3.3). | Mitigable, no eliminable: prompt + válvula + juez + revisión humana. |

El VLM real no se ejecutó en la validación local; los fallos de lectura del VLM son **esperados,
no medidos**. El OCR sí se ejecutó y es la evidencia que se reporta.

[COMPLETAR – Isabella, con aportes de cada integrante]
