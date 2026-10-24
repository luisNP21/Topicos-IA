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

*Responsable: Luis. Detalle técnico completo en [`eval_visual/README.md`](eval_visual/README.md).*

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

[COMPLETAR – Luis y Agustín]

### 3.4 Línea base de la última entrega

*Sistema de referencia, versión, valores por dimensión y archivo de resultados.*

[COMPLETAR – Isabella]

### 3.5 Resultados y delta

*Tabla por sistema, nivel de degradación y dimensión, con el delta respecto a la línea base.
Cada dueño interpreta su dimensión: exacto (Luis), semántico (Pau), juez (Agustín).*

| Sistema | Nivel | Exacto | Δ | Semántico | Δ | Juez | Δ | Fuente |
|---|---|---|---|---|---|---|---|---|
| [COMPLETAR] | | | | | | | | |

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

*Documentos escaneados o fotografiados usados en las pruebas de M4: tipo, procedencia y cómo
se garantiza que no contienen datos de pacientes reales.*

[COMPLETAR – Pau]

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

[COMPLETAR – Isabella, con aportes de cada integrante]
