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

*Responsables: Agustín (componente), Pau (punto de entrada), Isabella (integración con
harness y cache).*

### 1.1 Qué hace y dónde se conecta

*Flujo completo desde la imagen hasta las entidades y los fragmentos de guía, y en qué punto de
`run_inference/` se inserta.*

[COMPLETAR – Agustín]

### 1.2 Modelos y parámetros

*VLM (checkpoint y versión exactas), forma de acceso, baseline de OCR clásico, parámetros de
inferencia, latencia y costo por documento (medidos, con fuente).*

[COMPLETAR – Agustín]

### 1.3 Prompt de transcripción

*Texto literal y sus versiones.*

[COMPLETAR – Agustín]

### 1.4 Interfaz, configuración y estructura de archivos

*Interfaz `run(cfg, project_root) -> dict`, bloque en `config.yaml`, árbol de archivos nuevos o
modificados.*

[COMPLETAR – Agustín]

### 1.5 Integración con el cache de predicciones y el harness

*Cómo se registra cada sistema en `predictions/<sistema>.json` y cómo lo lee el harness.*

[COMPLETAR – Agustín e Isabella]

### 1.6 Punto de entrada del sistema

*App en Gradio o Streamlit que acepta imagen o texto y devuelve entidades más fragmentos de
guía: entradas aceptadas, salida mostrada, flujo interno, manejo de errores, generación final
con LLM externo, comando para ejecutarla y capturas.*

[COMPLETAR – Pau (app) y Agustín (generación con LLM externo)]

### 1.7 Cómo ejecutarlo

*Comandos exactos, dependencias con versiones, semillas y variables de entorno requeridas (sin
secretos).*

[COMPLETAR – Agustín]

---

## 2. Justificación del patrón de integración

*Responsable: Pau, con revisión de Isabella.*

### 2.1 Por qué input

*Argumento central, con evidencia del proyecto.*

[COMPLETAR – Pau]

### 2.2 Qué se habría perdido con tool y con output

| Patrón | Qué se habría perdido |
|---|---|
| Tool | [COMPLETAR – Pau] |
| Output | [COMPLETAR – Pau] |

### 2.3 Alternativa visual descartada y conexión con la teoría

*Por qué no CLIP sobre figuras de las guías, y cómo se aplican al diseño los temas vistos en
clase (parches de ViT y resolución, alucinación del VLM).*

[COMPLETAR – Pau]

---

## 3. Impacto medido en el harness

*Responsable de la sección: Isabella. Aportes: Luis (set de evaluación, métricas exactas y de
transcripción), Pau (métricas semánticas), Agustín (juez).*

### 3.1 Set de evaluación

*Cómo se generan las imágenes a partir de los documentos del test: niveles de degradación,
parámetros, semillas, script, ubicación y manifiesto. Aclarar que el gold sigue siendo el texto
original.*

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

**Juez visual.** [COMPLETAR – Agustín: cómo se corre `M4/vision/juez.py` sobre
`candidatos_juez.jsonl`, qué modelo juzga y cómo se agregan los veredictos.]

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
Cada dueño interpreta su dimensión: exacto (Luis), semántico (Pau), juez (Agustín).*

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

*Respuesta directa y con cifras por sistema y nivel. Incluye la decisión sobre cualquier variante
que no aporte (se retira del sistema o se conserva, y por qué).*

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

*Responsables: Pau (casos y entradas finales), Luis (formato y estructura), Isabella (análisis
de fallos).*

### 4.1 Entradas finales del sistema

*Documentos escaneados o fotografiados usados en las pruebas de M4: tipo, procedencia y cómo
se garantiza que no contienen datos de pacientes reales.*

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

*Casos clínicos propios (abreviaturas, entidades partidas entre líneas, tildes y ñ, dosis,
sellos o firmas, tablas de laboratorio, manuscrito real, entre otros). El fallo esperado se
documenta antes de ejecutar y el observado después.*

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

*Etapa donde se rompe (lectura del VLM, chunking, NER, retrieval), evidencia y si el fallo es
corregible o inherente al enfoque.*

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
