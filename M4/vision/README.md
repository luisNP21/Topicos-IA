# M4 · Visión como *input* — transcripción de notas clínicas

**Responsable:** Agustín · **Tarea 1 del reparto de M4** · **Patrón:** `input`

Esta carpeta implementa el **componente visual** del proyecto: una imagen de una
nota clínica (foto o escaneo) se convierte en texto y ese texto entra al sistema
de M3 (NER de M1/M2 → retrieval → generación). El sistema de M3 **no se
reescribe**: la visión es un traductor en la puerta.

> La justificación formal del patrón (`input` vs. `tool` vs. `output`) es la
> Tarea 2 de Pau. Aquí se implementa el patrón, no se duplica esa justificación.

---

## 1. Qué hace y qué no hace

**Hace**

- `ver(imagen) -> (texto, None)`: transcribe con **VLM** o con **OCR clásico**
  y aplica una **válvula de escape** si no hay texto legible. Respeta la firma
  del lab S13 (`texto, json_o_None`); los metadatos están en `transcribir()`.
- `sistema_multimodal(pregunta, imagen)`: enchufa la transcripción al sistema de
  M3 mediante `responder_m3(pregunta, contexto_extra)`.
- `run(cfg, project_root) -> dict`: recorre un manifiesto de casos con imagen,
  escribe el **cache de transcripciones** y, si se pide, el **cache NER** en el
  formato que ya lee el harness de M2.
- Deja un **primitivo de juez visual** para que Luis lo use en la auditoría de
  alucinaciones (Tarea 5a).

**No hace (para no invadir tareas de otros)**

- No construye el eval set con imágenes ni las degradaciones sintéticas
  (Luis, Tarea 3a) ni los casos límite/fotos reales (Pau, Tarea 3b).
- No modifica el harness de M2 ni calcula el delta (Isabella, Tareas 4 y 5b).
- No calcula CER/WER ni agrega la tasa de alucinaciones (Luis, Tarea 5a).
- No construye la app Gradio/Streamlit (Pau, Tarea 6).
- No escribe el README de una página del equipo ni el informe de fallos
  (Isabella, Tareas 5c y entrega final).
- No inventa métricas: la corrida local de validación usa un backend falso y
  solo comprueba el cableado. Los números reales salen de correr los backends
  en Colab con el eval set del equipo.

---

## 2. Backends

| Backend | Qué es | Por qué |
|---|---|---|
| `vlm` | `Qwen/Qwen2-VL-2B-Instruct` (S11/S13) | Lee fotos y escaneos; sigue la receta encoder visual → proyector → LLM. Puede alucinar, por eso la instrucción es literal y con válvula. |
| `ocr_docling` | **Docling** (`DocumentConverter`) | Es la herramienta que Isabella ya usa en M3 para el corpus; conserva estructura/tablas y no requiere API key. Baseline OCR recomendado. Las versiones recientes incluyen OCR (RapidOCR) para imágenes sin capa de texto. |
| `ocr_tesseract` | Tesseract vía `pytesseract` | Baseline clásico de respaldo cuando Docling no está disponible o para comparar motores. |
| `auto` | Prueba `vision.fallback_order` | Solo hace fallback ante **error** del backend; si un motor funciona y no ve texto, esa es una respuesta válida (válvula). |
| `fake` | Texto fijo | Solo para tests y para validar el pipeline sin GPU. |

Docling se configura en español de forma *best-effort* (`ocr.docling.lang: "es"`).
Si la versión instalada no expone esa opción, se conserva su pipeline por
defecto y se registra la transcripción igual; para un control total del idioma,
usar `ocr_tesseract` con `lang: "spa+eng"`.

> **Nota de entorno:** Docling y Qwen2-VL pueden pedir versiones distintas de
> `transformers`. Si el entorno de Colab entra en conflicto, correr el baseline
> OCR en un runtime/entorno aparte o usar `ocr_tesseract`. El código maneja el
> fallo de forma explícita y no rompe el lote.

La comparación **VLM vs. OCR clásico** se ejecuta con:

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml --comparar --manifest ...
```

Eso produce un cache por backend y un resumen lado a lado en
`M4/vision/outputs/comparacion_backends.json` (casos, válvulas, errores y
latencia media). **No** calcula CER/WER ni el delta: eso es Tareas 5a y 5b.

---

## 3. Archivos

```
M4/vision/
├── prompts.py          # instrucción de transcripción literal + válvula
├── contratos.py        # Transcripcion, CasoVision, Protocols (sin imports pesados)
├── backends.py         # VLM Qwen2-VL, Docling, Tesseract, fake
├── transcripcion.py    # ver(), transcribir(), carga de imagen, limpieza
├── manifiesto.py       # CSV/JSON de casos con imagen (no genera casos)
├── cache.py            # cache de transcripciones y cache NER {doc_id: [entidades]}
├── ner_bridge.py       # visión -> Clinical BERT + LoRA (reutiliza M2)
├── integracion.py      # sistema_multimodal() + responder_m3 real/sustituto
├── juez.py             # primitivo de fidelidad visual (apoyo a Luis)
├── demo.py             # nota clínica sintética, marcada como sintética
├── run_vision.py       # run(cfg, project_root) + CLI
├── config.yaml
├── requirements.txt
├── test_vision.py      # 28 pruebas con backend falso (sin GPU/API)
└── README.md
```

---

## 4. Contratos

```python
# S13: imagen -> (texto, None)
texto, _ = ver(imagen, backend_nombre="ocr_docling", cfg=cfg)

# Interfaz rica para el scorecard: backend, latencia, válvula, error y hash
t = transcribir(imagen, backend_nombre="ocr_docling", cfg=cfg, doc_id="img_01")

# S13: pregunta + imagen -> respuesta del sistema
respuesta = sistema_multimodal(
    "¿Cuál es el tratamiento de la DM2?",
    imagen,
    entidad="DM2",                 # opcional si el eval set ya la trae
    responder_m3=responder_real,   # o se construye con crear_responder_m3(...)
    cfg=cfg,
)

# Interfaz uniforme del proyecto
resumen = run(cfg, project_root)
```

`ver()` devuelve la válvula literal si no hay texto:

```text
"No se aprecia texto legible en la imagen."
```

y marca `valvula=True`. Así el sistema no inventa una nota clínica.

### Enchufe real a M3

`crear_responder_m3(cfg, project_root, extractor=None)` reutiliza:

- `M3/retrieval` (Pau) para recuperar fragmentos,
- `M3/tools` (Luis) para normalizar siglas,
- `M3/generacion` (Agustín) para redactar con la válvula de escape.

Requiere `PROJECT_ROOT`, el índice de M3 y `GROQ_API_KEY`. Además de las
dependencias de visión, instalar las de M3 para el adaptador real:

```bash
pip install -r M3/corpus/requirements.txt \
            -r M3/generacion/requirements.txt \
            -r M3/tools/requirements.txt
```

Con `integracion.usar_mocks: true` devuelve un **sustituto marcado**
(`[SUSTITUTO M3]`), útil para demo/tests sin credenciales.

El **extractor de entidades** es inyectable (`extractor=...`). El sistema
completo de Pau puede pasar el NER de M1/M2; la visión no lo reimplementa.

---

## 5. Cómo correr

### 5.1 Colab (recomendado)

```bash
# Desde la raíz del repo clonado
pip install -r M4/vision/requirements.txt
```

**Demo sintética con VLM** (no usa datos personales):

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml --demo --backend vlm
```

**Demo sintética con Docling** (CPU; Docling reciente incluye OCR):

```bash
pip install "docling>=2.0.0"
python M4/vision/run_vision.py --config M4/vision/config.yaml --demo --backend ocr_docling
```

**Con el eval set del equipo** (cuando Luis/Pau lo dejen en
`M4/eval_set/casos_imagen.csv`):

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml \
  --manifest M4/eval_set/casos_imagen.csv --backend vlm
```

**Comparar VLM vs. OCR y generar el cache NER para el harness:**

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml \
  --manifest M4/eval_set/casos_imagen.csv --comparar --ner
```

**Tesseract** (requiere binario del sistema):

```bash
apt-get install -y tesseract-ocr tesseract-ocr-spa
pip install pytesseract
python M4/vision/run_vision.py --config M4/vision/config.yaml --demo --backend ocr_tesseract
```

### 5.2 Local / tests

```bash
python M4/vision/test_vision.py
# o: python -m unittest M4.vision.test_vision -v
```

Las 28 pruebas usan `BackendFake`: no descargan modelos, no llaman APIs y no
necesitan Docling/Tesseract.

### 5.3 Manifiesto de casos

CSV o JSON, una fila por caso. Columnas aceptadas:

| Columna | Obligatoria | Uso |
|---|---|---|
| `id` (o `doc_id`) | sí | clave del cache; debe coincidir con el eval set del harness |
| `imagen` (o `image`/`path`) | sí | ruta relativa a `imagenes_dir`, absoluta o URL |
| `pregunta` (o `question`) | no | pregunta clínica del caso |
| `entidad` | no | entidad gold; si viene, se pasa explícita a M3 |
| `esperado`, `criterio`, `tipo` | no | los usa el harness/eval set; la visión los ignora |

La visión **no genera** este archivo: solo lo consume.

---

## 6. Cachés y contrato con el harness (Isabella)

### 6.1 Cache de transcripciones

`M4/vision/predictions/transcripciones_<backend>.json`

```json
{
  "img_01": {
    "doc_id": "img_01",
    "texto": "Paciente con DM2 e HTA...",
    "backend": "vlm",
    "segundos": 1.23,
    "valvula": false,
    "error": null,
    "imagen": "M4/eval_set/imagenes/img_01.png",
    "sha256": "…"
  }
}
```

Es la fuente de verdad de la visión. Lo usa la auditoría de Luis (CER/WER) y
sirve para reproducir.

### 6.2 Cache NER para el harness

`M2/predictions/vision_<backend>.json`

```json
{"img_01": ["diabetes mellitus tipo 2", "hipertension"]}
```

Es exactamente el formato que espera `M2/harness/cached_system.py`
(`doc_id -> list[str]`). El puente reutiliza el modelo Clinical BERT + LoRA y
el chunking de `M2/run_inference`. **No se modifica el harness**: Isabella
apunta su configuración a este archivo cuando implemente la Tarea 4.

Si falta un `doc_id` o sobra uno, el cache falla explícitamente: un desajuste
silencioso se vería como una caída de F1 y sería deshonesto.

---

## 7. Reproducibilidad y manejo de fallos

- VLM con `do_sample=False` y `temperature` no aplica: salida determinista para
  la misma imagen/modelo.
- Cada transcripción guarda `sha256` del contenido visual y `segundos` de
  latencia.
- Si un backend falla (sin GPU, sin Docling, sin binario de Tesseract), se
  registra `error` y se devuelve la válvula; el lote **no se cae**. El resumen
  cuenta `n_ok`, `n_error` y `n_valvula`.
- Si `auto` no encuentra ningún backend, devuelve un error combinado con los
  motivos, no un fallback silencioso.

---

## 8. Ética y límites

- **Sin datos personales en el repo.** La demo es una nota **sintética** y lleva
  el rótulo visible `NOTA CLÍNICA SINTÉTICA · SIN DATOS PERSONALES · M4`. Las
  fotos/documentos reales del eval set deben anonimizarse (dueños: Luis/Pau).
- **Ver no es diagnosticar.** El `SYSTEM_VISION` y la instrucción prohíben
  diagnosticar; la salida es una transcripción. La decisión clínica es humana.
- **Alucinación del VLM.** La instrucción literal + válvula reduce el riesgo,
  pero no lo elimina. La auditoría de Luis (Tarea 5a) y el juez visual miden ese
  modo de falla. El VLM puede "corregir" una sigla o inventar una línea.
- **OCR clásico.** Falla con letra manuscrita, sellos encima del texto, tablas
  complejas, baja resolución y tildes/ñ perdidas. Docling está pensado para
  documentos; una foto de celular puede degradarlo.
- **Resolución y parches (ViT).** Una letra pequeña puede caer dentro de un
  parche y perderse. Es una limitación del encoder visual, no del prompt.
- **Sesgos de fábrica.** Qwen2-VL, Docling/Tesseract y sus datos de origen
  pueden sesgar la lectura de nombres, acentos o formatos no colombianos. El
  eval set debe incluir un caso donde eso importe.
- **Latencia.** Un VLM en la ruta multiplica el tiempo de respuesta; el resumen
  guarda la latencia por caso y el promedio. Ese número va a la tabla de delta.
- **Set sintético vs. escaneos reales.** Las degradaciones sintéticas son una
  aproximación de trabajo; no equivalen a fotos reales. Quien construya el eval
  set debe declararlo.

---

## 9. Estado de la Tarea 1

| Entregable | Estado |
|---|---|
| Backends VLM + OCR clásico + factory | Hecho |
| `ver()` y `sistema_multimodal()` (patrón input) | Hecho |
| `run(cfg, project_root)` | Hecho |
| `config.yaml` | Hecho |
| Cache de transcripciones | Hecho |
| Cache NER compatible con `run_inference/` | Hecho |
| Puente NER opcional (reutiliza M2) | Hecho |
| Primitivo de juez visual (apoyo a Luis) | Hecho |
| Pruebas sin GPU/API | **28/28 OK** (backend falso) |
| Corrida real VLM/Docling | Pendiente de Colab + dependencias |
| Eval set con imágenes y delta | Fuera de esta tarea (Luis/Pau/Isabella) |
| App Gradio/Streamlit | Fuera de esta tarea (Pau) |

### Cómo se cubre la rúbrica M4 desde esta pieza

- **Componente funcional e integrado:** `run()` + `sistema_multimodal()` +
  `responder_m3` real/sustituto; no es un notebook aislado.
- **Justificación del patrón:** la implementación es `input`; la argumentación
  formal la aporta Pau (Tarea 2).
- **Impacto medido:** esta pieza deja los caches y la latencia; el delta lo
  calcula Isabella (Tarea 5b) para no duplicar su trabajo.
- **Casos de prueba:** la visión consume el eval set de Luis/Pau; las pruebas
  propias cubren casos límite del componente (sin texto, backend caído, nota
  larga sin extractor, ids repetidos, caché inválido).
- **Límites y ética:** sección 8 y rótulo de imagen sintética.

---

## 10. Referencias del curso

- `profesora/sesiones/s13/README.md` y `S13_Lab_Integracion_Visual.ipynb`:
  patrón *input*, `ver()`, `sistema_multimodal()`, criterios y tabla de delta.
- `profesora/sesiones/s11/S11_Lab_Vision.ipynb`: Qwen2-VL-2B, válvula de escape
  y extracción estructurada.
- `M3/corpus/ingesta.py`: uso de Docling en el equipo.
- `M2/run_inference/`: modelo NER, chunking y formato del cache que consume el
  harness.
