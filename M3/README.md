# M3 - RAG clínico sobre guías: documentación del módulo

Documento único del módulo M3. Cada responsable completa **su propia sección** con los
hallazgos y limitaciones encontrados al ejecutar su parte. Las secciones 1 y 2 son de contexto
común; las secciones 3 a 8 son una por área; la sección 9 se completa entre todos al final.

**Convenciones para completar el documento (léanlas antes de escribir):**

- Todo número reportado debe indicar de dónde sale (archivo, celda, comando o corrida). Si no se
  puede rastrear, no se reporta.
- Separen siempre **Hallazgos** (derivados de datos medidos) de **Supuestos** (cosas que
  asumimos sin haberlas medido). No mezclarlos en el mismo bullet.
- Una limitación se documenta con su impacto observado, no solo con su descripción teórica.
- Los bloques marcados con `[COMPLETAR]` son los que debe llenar cada responsable. Borren las
  instrucciones entre paréntesis al completar.

---

## 1. Contexto y decisiones de diseño

### 1.1 Objetivo del módulo

Conectar las entidades de enfermedad extraídas por el modelo de M1/M2 con fragmentos de guías
clínicas mediante un pipeline RAG, evaluado con RAGAS y con el harness propio de M2.

### 1.2 Decisiones de diseño tomadas

| Decisión | Alternativa descartada | Justificación |
|---|---|---|
| Arquitectura encoder para la extracción de entidades | Encoder-decoder (mT5) | El extractor identifica spans de enfermedad para construir las queries de retrieval, es decir, es un problema de clasificación de tokens. El encoder entrega spans exactos (start/end) sin parsear texto generado; un error de extracción se propaga directo a la query, y un decoder puede generar una entidad que no está en el texto. Es coherente con el modelo primario de M2 (clinical RoBERTa) y con el harness, construido alrededor de etiquetas BIO. Limitación aceptada: el encoder no genera texto, por lo que la síntesis de la respuesta la hace un LLM aparte (sección 8), que no forma parte del pool de arquitecturas comparado en M1. |
| Guías clínicas simuladas generadas con IA | Guías clínicas reales | Enfocar el esfuerzo en la parte técnica del sistema (retrieval, tools, evaluación). Consecuencia: las conclusiones clínicas no son extrapolables (ver 3.8 y 9.4). |
| Tool de tool-use: normalización terminológica (SNOMED CT / UMLS) | Búsqueda externa de respaldo (PubMed / Europe PMC); verificación de grounding con LLM | Resuelve la variación léxica de la entidad extraída ("DM2" vs "diabetes mellitus tipo 2") contra un vocabulario controlado, y retoma el linking a SNOMED que M1 dejó explícitamente para un módulo posterior. Limitación conocida: no resuelve el desajuste de intención de búsqueda (que un chunk mencione la enfermedad no implica que contenga tratamiento); eso se mitiga en retrieval con query transformation y filtro por `categoria_seccion` (ver 1.4). |
| LLM externo como generador de la respuesta final, no como tool | LLM como tool de verificación | La generación es el paso final del RAG, no una herramienta opcional; contarla como tool haría ambiguo el requisito de tool use. |
| Invocación de la tool decidida por regla determinista (umbral de score de retrieval) | Function-calling decidido por el LLM | Trazabilidad y reproducibilidad: cada invocación queda explicada por un valor medible y no por el razonamiento de un LLM. |
| Un `.py` por componente, orquestado desde un notebook delgado en Colab | Notebooks con toda la lógica | Contratos de datos claros, diffs revisables con 4 personas en paralelo, mismo patrón de M2. |
| Toda la configuración del corpus en un único `config.yaml` | Constantes en código y `cfg` dividido entre YAML y diccionario | Una sola fuente de verdad; evita que tokenizer y modelo de embeddings usen nombres de modelo distintos. |

### 1.3 Tareas realizadas

| Tarea | Sección |
|---|---|---|
| Corpus: procedencia y responsabilidad | 3 |
| Adaptación del harness de M2 | 4 |
| Retrieval avanzado + comparación contra RAG ingenuo | 5 |
| Tool de normalización terminológica | 6 |
| Evaluación con RAGAS | 7 |
| Generación final con LLM | 8 |
| Corrección del LLM-as-judge de M2 | 8 |

### 1.4 Hallazgo de diseño que condiciona el módulo

Al revisar guías clínicas reales durante la búsqueda de fuentes se observó (de forma
cualitativa, sin cifra medida) que el extractor entrega nombres de enfermedad, pero las guías
no están organizadas como "enfermedad -> tratamiento": un fragmento puede mencionar la
enfermedad sin contener ninguna recomendación terapéutica. Consecuencias de diseño:

- El corpus clasifica cada sección en `categoria_seccion` durante el chunking (sección 3).
- La generación debe poder responder "no se encontró evidencia suficiente" (`fallback_used`)
  en lugar de forzar una respuesta (sección 8).
- La frecuencia con que esto ocurre es un resultado a medir y reportar en la sección 7, no un
  supuesto.

---

## 2. Arquitectura y contratos entre módulos

```
Texto clínico
   |
   v
[Extractor NER (encoder, M1/M2)]  ->  entidades crudas: set[str]
   |
   v
[Retrieval] <-- corpus indexado 
   |   score bajo con entidad cruda?
   |        si -> [Tool de normalización] -> reintento de retrieval
   v
fragments recuperados: list[Fragment]
   |
   v
[Generación LLM]  ->  RespuestaRAG
   |
   v
[RAGAS + harness M2]  ->  scorecard cruzado
```

### 2.1 Contratos de datos

```python
Chunk = {
    "chunk_id": str, "doc_id": str, "texto": str,
    "seccion": str, "categoria_seccion": str,   # tratamiento | diagnostico | epidemiologia | otro
}

Fragment = {"chunk_id": str, "doc_id": str, "texto": str, "score": float}

NormalizationResult = {
    "entidad_original": str, "entidad_normalizada": str,
    "source_terminology": str | None, "normalization_failed": bool,
}

RespuestaRAG = {"answer": str, "sources_used": list[str], "fallback_used": bool}

Sistema = Callable[[str], set[str]]      # texto -> entidades detectadas
```

[COMPLETAR entre todos: si algún contrato cambió durante el desarrollo, actualizarlo aquí y
avisar a los módulos que dependen de él.]

---

## 3. Corpus: procedencia y responsabilidad


### 3.1 Qué se implementó

Pipeline de construcción del corpus de guías clínicas, con las etapas:

1. **Ingesta** con Docling: extracción estructural del PDF (encabezados y tablas; las tablas
   se exportan a markdown).
2. **Filtro de front matter**: descarta secciones por título (portada, resumen, abstract,
   índice, glosario, abreviaturas, etc.) y por densidad de texto (menos de `min_palabras`).
3. **Chunking** por sección, con corte por tokens y solapamiento, usando el tokenizer del
   mismo modelo de embeddings.
4. **Clasificación de sección** (`categoria_seccion`: tratamiento, diagnostico, epidemiologia,
   otro) por palabras clave sobre el título, para que retrieval pueda filtrar por intención.
5. **Embeddings** con `intfloat/multilingual-e5-base` (prefijo `passage: ` al indexar y
   `query: ` al buscar).
6. **Persistencia dual**: JSON por chunk como fuente de verdad versionable, y colección
   ChromaDB como índice de consulta.

Orquestado por `run(cfg, project_root) -> dict` y ejecutado desde Colab con `run_corpus.py`
(código en git, rama `corpus`; datos y salidas en Drive vía `PROJECT_ROOT` en `.env`). Las
fuentes ya procesadas se omiten por caché de `doc_id`: una nueva versión de una guía se sube
con otro nombre. Detalle técnico completo en `README_corpus.md`.

### 3.2 Entregables

| Archivo / carpeta | Descripción |
|---|---|
| `ingesta.py` | Carga de `fuentes.yaml`, parseo con Docling, filtro de front matter |
| `chunking.py` | Tokenizer, chunking por sección, renumeración de ids, clasificación de sección |
| `embeddings.py` | Carga del modelo, embeddings de chunks y de queries |
| `corpus_store.py` | `construir_indice_chroma`, `cargar_coleccion`, `guardar_chunks_json` (únicos que consume retrieval) |
| `config_utils.py` | Lectura de `config.yaml` y construcción de los patrones regex |
| `pipeline_corpus.py` | Orquestador con caché por `doc_id` y manifest fusionado |
| `run_corpus.py` | Entry point de línea de comandos (resuelve `.env` y rutas) |
| `config.yaml` | Modelo de embeddings, parámetros de chunking, patrones de front matter y de categorías, rutas de salida |
| `fuentes.yaml` | Registro de procedencia por guía (dato, no configuración) |
| `requirements.txt` | Dependencias del módulo |
| `data/guias_clinicas/*.pdf` | Guías simuladas de entrada (en Drive) |
| `data/guias_clinicas/chunks/*.json` | Chunks, fuente de verdad |
| `data/chroma_guias/` | Índice ChromaDB (colección `guias_clinicas`) |
| `data/guias_clinicas/corpus_manifest.json` | Procedencia por documento indexado |
| [COMPLETAR: ubicación en el repo] | Prompt usado para generar las guías simuladas |

**Estructura del chunk:**

```python
Chunk = {
    "chunk_id": str, "doc_id": str, "texto": str,
    "seccion": str,             # título crudo, para citar la fuente
    "categoria_seccion": str,   # tratamiento | diagnostico | epidemiologia | otro
}
```

### 3.3 Resultados medidos

Parámetros de configuración (confirmar contra el `config.yaml` final antes de reportar):

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo de embeddings | `intfloat/multilingual-e5-base` | `config.yaml` |
| `max_tokens` / `overlap_tokens` | 350 / 50 (valores propuestos inicialmente) | [COMPLETAR: confirmar en `config.yaml`] |
| `min_palabras` (front matter) | 25 (valor propuesto inicialmente) | [COMPLETAR: confirmar en `config.yaml`] |

Resultados de la corrida:

| Métrica | Valor | Fuente del número |
|---|---|---|
| Documentos indexados | [COMPLETAR] | [COMPLETAR: `stats` de `run()` y celda de verificación de fuentes] |
| Fuentes fallidas | [COMPLETAR] | [COMPLETAR: `stats["errores"]`] |
| Chunks totales en Chroma | [COMPLETAR] | [COMPLETAR: `coleccion.count()`] |
| Secciones filtradas como front matter | [COMPLETAR] | [COMPLETAR: `stats["n_secciones_filtradas"]`] |
| Chunks por `categoria_seccion` | [COMPLETAR: tratamiento / diagnostico / epidemiologia / otro] | [COMPLETAR: conteo sobre metadata de Chroma o JSON] |
| Chunks por documento (mín / media / máx) | [COMPLETAR] | [COMPLETAR] |
| Tamaño de la colección Chroma en Drive | ~18 MB en una corrida intermedia | Observado en Drive; [COMPLETAR: valor final] |

### 3.4 Procedencia, licencia y responsabilidad

| Aspecto | Detalle |
|---|---|
| Origen | Guías clínicas simuladas, generadas con IA por decisión del equipo. El contenido clínico (dosis, cifras, criterios) es inventado y no cita evidencia ni fuentes reales |
| Generación | Un prompt por enfermedad que fija la estructura de secciones (front matter: portada, resumen, índice, abreviaturas, glosario; cuerpo: epidemiología, diagnóstico, tratamiento farmacológico con tabla de dosificación, tratamiento no farmacológico, seguimiento). Enfermedades elegidas entre las especialidades cubiertas por DisTEMIST. [COMPLETAR: IA usada, fecha de generación, lista final de enfermedades] |
| Licencia | [COMPLETAR: términos de uso del contenido generado y del modelo que lo generó] |
| Vigencia | Sin vigencia clínica: las fechas de publicación son ficticias. No debe usarse para ninguna decisión clínica |
| Responsable | Isa (campo `responsable` en `fuentes.yaml`) |
| Trazabilidad | `corpus_manifest.json` registra, por documento: `doc_id`, `titulo`, `fuente_url`, `licencia`, `fecha_publicacion`, `fecha_indexado`, `responsable`. Cada chunk conserva `doc_id` y `seccion` para citar el origen |

### 3.5 Qué pasa si el corpus está mal

| Modo de falla | Efecto en el sistema | Estado | Detección / mitigación |
|---|---|---|---|
| Sección de tratamiento ausente o mal clasificada | El filtro por categoría deja el pool vacío o trae chunks de otra categoría | Anticipado | Fallback a búsqueda sin filtro; medir frecuencia en la evaluación |
| Enfermedad sin cobertura en el corpus | Recall bajo; riesgo de respuesta forzada con contexto pobre | Anticipado | `fallback_used` en generación; context recall de RAGAS |
| Guía desactualizada o con contenido erróneo | Respuesta fluida con evidencia obsoleta y score alto; el pipeline no lo detecta | Anticipado | Sin detección automática; con guías simuladas no aplica clínicamente, pero se documenta como límite del diseño |
| Mismo `doc_id` reutilizado con contenido editado | La caché sirve los chunks viejos sin aviso | Anticipado (regla aceptada) | Disciplina de nombres: nueva versión = nuevo nombre |
| Front matter mal filtrado (sección clínica descartada o boilerplate indexado) | Pérdida de información o ruido en el índice | [COMPLETAR: observado / anticipado] | Inspección manual de las secciones marcadas como filtradas |
| Desfase entre chunks JSON, manifest y Chroma | Documentos visibles en una capa pero no en otra | Anticipado | Celda de verificación de fuentes que compara las cuatro capas |

[COMPLETAR: reemplazar "Anticipado" por "Observado" y agregar la evidencia en los casos que
ocurran durante la ejecución final.]

### 3.6 Hallazgos (derivados de datos)

Observados al ejecutar el pipeline:

- Docling devuelve las tablas como `TableItem`, sin atributo `.text`; la primera versión de
  `ingerir_documento` fallaba con `AttributeError`. Se corrigió exportando la tabla a markdown.
- El primer indexado falló con `DuplicateIDError` con 64 ids duplicados: el contador de
  `chunk_id` se reiniciaba en cada sección. Además, `guardar_chunks_json` había sobrescrito
  archivos de chunks distintos bajo el mismo nombre. Se corrigió con `renumerar_chunks` (id
  secuencial por documento) y se regeneró la carpeta de chunks.
- Una ruta escrita como expresión de Python dentro de `fuentes.yaml` se leyó como texto
  literal y produjo `FileNotFoundError`. Las rutas del YAML son solo texto relativo; la
  concatenación con `PROJECT_ROOT` se hace en código.
- Un nombre de archivo mal escrito en `fuentes.yaml` interrumpió una corrida en la que ya se
  habían generado los chunks de 3 PDFs. Motivó la caché por `doc_id`, para no reprocesar
  guías ya indexadas.
- Al revisar guías reales, el nombre de la enfermedad aparecía en fragmentos sin recomendación
  de tratamiento (observación cualitativa, sin cifra; ver 1.4). Motivó `categoria_seccion`.

Cuantitativos: [COMPLETAR: conteos de la tabla 3.3 y hallazgos de la revisión manual del filtro
de front matter, indicando cuántas secciones revisadas y cuántas se descartaron por error.]

### 3.7 Supuestos (no medidos)

- `categoria_seccion`, asignada por palabras clave sobre el título, refleja el contenido real
  de la sección. No se validó contra una revisión manual de los chunks.
- `min_palabras` conserva las secciones clínicas con poco texto corrido (por ejemplo tablas de
  recomendaciones). [COMPLETAR: cuántas guías se inspeccionaron para calibrarlo.]
- Las guías simuladas reproducen la estructura de guías reales lo suficiente para representar
  el problema de retrieval. No se comparó contra guías reales.
- `multilingual-e5-base` es adecuado para retrieval en español clínico. No se comparó contra
  otros embedders en este módulo.
- Los valores de `max_tokens` y `overlap_tokens` son adecuados. No se hizo un barrido de
  parámetros.
- Chroma se puede leer directamente desde Drive montado. Si otro integrante ve errores de
  lectura, copiar la carpeta a disco local del runtime.

### 3.8 Limitaciones

- **Corpus simulado:** el contenido clínico es inventado, por lo que ninguna conclusión sobre
  calidad clínica de las respuestas es válida. Además, guías generadas con una plantilla común
  son más homogéneas que las reales, y esto puede hacer el retrieval más fácil de lo que sería
  en producción.
- **Clasificación por palabras clave:** títulos que no usen las raíces configuradas caen en
  `otro`, y la clasificación no mira el contenido del chunk.
- **Tamaño del corpus:** [COMPLETAR: número de documentos y enfermedades cubiertas]; con pocos
  documentos, las métricas de retrieval tienen poca variabilidad y son sensibles a casos
  individuales.
- **Tablas:** se exportan a markdown; tablas complejas pueden perder estructura o producir
  chunks largos. [COMPLETAR: resultado de la revisión de las tablas de dosificación.]
- **Caché por existencia:** el pipeline no detecta cambios de contenido bajo el mismo `doc_id`.
- **Sin detección de guías desactualizadas o contradictorias entre documentos.**
- **Dependencia de Docling** para el parseo de PDFs; un PDF escaneado o con maquetación
  atípica puede extraerse mal. [COMPLETAR: si ocurrió con alguna guía.]

---

## 4. Adaptación del harness de M2

### 4.1 Qué se implementó

[COMPLETAR: refactor a `harness(eval_set, sistema)` desacoplado del adaptador LoRA, y separación
de la inferencia en `run_inference/`.]

### 4.2 Entregables

| Archivo / carpeta | Descripción |
|---|---|
| `harness/` | [COMPLETAR] |
| `run_inference/` | [COMPLETAR] |

### 4.3 Contrato

```python
harness(eval_set: list[dict], sistema: Sistema) -> dict   # precision, recall, f1
```

[COMPLETAR: forma exacta del `eval_set` y del cache de predicciones.]

### 4.4 Resultados medidos

| Comparación | Precision | Recall | F1 | Fuente del número |
|---|---|---|---|---|
| Encoder solo | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| Encoder + normalización | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

### 4.5 Hallazgos (derivados de datos)

- [COMPLETAR]

### 4.6 Supuestos (no medidos)

- [COMPLETAR]

### 4.7 Limitaciones

- [COMPLETAR]

---

## 5. Retrieval avanzado y comparación contra RAG ingenuo

### 5.1 Qué se implementó

[COMPLETAR: técnicas implementadas (mínimo dos: hybrid search, reranking, query transformation),
con la justificación de cada una.]

### 5.2 Entregables

| Archivo | Descripción |
|---|---|
| [COMPLETAR] | `retrieve_naive` |
| [COMPLETAR] | `retrieve_advanced` |

### 5.3 Contrato

```python
retrieve_naive(query: str, k: int) -> list[Fragment]
retrieve_advanced(query: str, k: int) -> list[Fragment]
```

[COMPLETAR: parámetros relevantes (k, pesos de la fusión, modelo de reranking, umbrales).]

### 5.4 Gold set de relevancia

[COMPLETAR: cómo se construyó, cuántos pares (entidad, chunk esperado), quién lo armó y con qué
criterio. Incluir si se diseñó para probar el caso "la entidad se menciona pero no hay
tratamiento en ese chunk".]

### 5.5 Delta medido contra RAG ingenuo

| Métrica | Ingenuo | Avanzado | Delta | Fuente del número |
|---|---|---|---|---|
| [COMPLETAR: recall@k / precision@k / nDCG] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

Desglose por técnica (aporte individual de cada una):

| Configuración | Métrica | Valor |
|---|---|---|
| Ingenuo | [COMPLETAR] | [COMPLETAR] |
| + query transformation | [COMPLETAR] | [COMPLETAR] |
| + filtro por `categoria_seccion` | [COMPLETAR] | [COMPLETAR] |
| + hybrid search | [COMPLETAR] | [COMPLETAR] |
| + reranking | [COMPLETAR] | [COMPLETAR] |

### 5.6 Hallazgos (derivados de datos)

- [COMPLETAR]

### 5.7 Supuestos (no medidos)

- [COMPLETAR]

### 5.8 Limitaciones

- [COMPLETAR: por ejemplo, casos donde el filtro por categoría deja el pool vacío, sensibilidad
  a la query transformada, tamaño del gold set.]

---

## 6. Tool de normalización terminológica

### 6.1 Qué se implementó

[COMPLETAR: fuente terminológica usada (SNOMED CT, UMLS u otra), cómo se accede (API, archivo
local) y por qué.]

### 6.2 Entregables

| Archivo | Descripción |
|---|---|
| [COMPLETAR] | `normalizar_entidad(entidad) -> NormalizationResult` |

### 6.3 Criterio de invocación

[COMPLETAR: cuándo el sistema invoca la tool (umbral de score de retrieval con la entidad
cruda), con el valor del umbral y cómo se eligió.]

### 6.4 Comportamiento ante fallo

[COMPLETAR: qué hace el sistema si la tool falla (sin match, timeout, API caída), y cómo queda
registrado (`normalization_failed`).]

### 6.5 Resultados medidos

| Métrica | Valor | Fuente del número |
|---|---|---|
| Frecuencia de invocación (% de entidades) | [COMPLETAR] | [COMPLETAR] |
| Tasa de normalización exitosa | [COMPLETAR] | [COMPLETAR] |
| Delta en retrieval con vs. sin tool | [COMPLETAR] | [COMPLETAR] |
| Delta en F1 del harness con vs. sin tool | [COMPLETAR] | [COMPLETAR] |

### 6.6 Hallazgos (derivados de datos)

- [COMPLETAR]

### 6.7 Supuestos (no medidos)

- [COMPLETAR]

### 6.8 Limitaciones

- [COMPLETAR: cobertura de la terminología, dependencia de servicios externos, licencias de uso.]

---

## 7. Evaluación con RAGAS

### 7.1 Qué se implementó

[COMPLETAR: cómo se configuró RAGAS, qué LLM usa como evaluador internamente y sobre qué
eval set corre.]

### 7.2 Entregables

| Archivo | Descripción |
|---|---|
| [COMPLETAR] | `run_ragas(eval_set) -> RagasScores` |

### 7.3 Eval set

[COMPLETAR: tamaño, cómo se construyó, si tiene `ground_truth` y de dónde sale.]

### 7.4 Resultados medidos

| Métrica | Valor | Fuente del número |
|---|---|---|
| Faithfulness | [COMPLETAR] | [COMPLETAR] |
| Context precision | [COMPLETAR] | [COMPLETAR] |
| Context recall | [COMPLETAR] | [COMPLETAR] |
| Answer relevancy | [COMPLETAR] | [COMPLETAR] |

### 7.5 Cruce con el harness de M2

[COMPLETAR: tabla o análisis que relacione las métricas de RAGAS con las del harness (F1 de
extracción), indicando en qué etapa (extracción, retrieval o generación) falla el sistema.]

### 7.6 Análisis de fallos

| Caso | Etapa donde falla | Evidencia | Causa probable |
|---|---|---|---|
| [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

### 7.7 Hallazgos (derivados de datos)

- [COMPLETAR]

### 7.8 Supuestos (no medidos)

- [COMPLETAR]

### 7.9 Limitaciones

- [COMPLETAR: por ejemplo, sesgo del LLM evaluador, tamaño del eval set, ausencia de
  ground truth.]

---

## 8. Generación final con LLM y corrección del LLM-as-judge

### 8.1 Generación final

**Qué se implementó:** [COMPLETAR: modelo usado, prompt, cómo se restringe la respuesta a los
fragmentos recuperados.]

**Entregables:**

| Archivo | Descripción |
|---|---|
| [COMPLETAR] | `generar_respuesta(entidad, fragments) -> RespuestaRAG` |

**Comportamiento ante evidencia insuficiente:** [COMPLETAR: umbral y mensaje cuando
`fallback_used=true`.]

**Resultados medidos:**

| Métrica | Valor | Fuente del número |
|---|---|---|
| Frecuencia de `fallback_used=true` | [COMPLETAR] | [COMPLETAR] |
| [COMPLETAR: latencia, costo por consulta, etc.] | [COMPLETAR] | [COMPLETAR] |

### 8.2 Corrección del LLM-as-judge de M2

**Qué se corrigió:** [COMPLETAR: modelo reemplazado, motivo y validación de que las métricas
no cambiaron de forma inesperada.]

### 8.3 Hallazgos (derivados de datos)

- [COMPLETAR]

### 8.4 Supuestos (no medidos)

- [COMPLETAR]

### 8.5 Limitaciones

- [COMPLETAR: por ejemplo, riesgo de alucinación pese al contexto, dependencia de API externa,
  variabilidad entre corridas.]

---

## 9. Integración y conclusiones (se completa entre todos)

### 9.1 Estado de los checkpoints de integración

| Checkpoint | Contrato validado | Estado | Notas |
|---|---|---|---|
| Corpus -> Retrieval | Chunk / índice Chroma | [COMPLETAR] | [COMPLETAR] |
| Normalización -> Retrieval | NormalizationResult | [COMPLETAR] | [COMPLETAR] |
| Retrieval -> Generación | Fragment | [COMPLETAR] | [COMPLETAR] |
| Generación -> RAGAS | RagasExample | [COMPLETAR] | [COMPLETAR] |

### 9.2 Scorecard final del sistema

| Etapa | Métrica principal | Valor | Fuente |
|---|---|---|---|
| Extracción (harness M2) | F1 | [COMPLETAR] | [COMPLETAR] |
| Retrieval | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| Generación (RAGAS) | Faithfulness | [COMPLETAR] | [COMPLETAR] |

### 9.3 Dónde falla el sistema

[COMPLETAR: síntesis del análisis de fallos, con referencias a las secciones 5 a 8.]

### 9.4 Limitaciones globales del módulo

- [COMPLETAR: incluir el uso de guías simuladas y qué conclusiones no se pueden extrapolar a
  guías reales.]

### 9.5 Trabajo futuro

- [COMPLETAR]

### 9.6 Cómo reproducir todo el módulo

[COMPLETAR: referencia a .md con instrucciones]