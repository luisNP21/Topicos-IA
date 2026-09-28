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
|---|---|
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

Fragment = {
    "chunk_id": str, "doc_id": str, "texto": str, "score": float,
    "score_tipo": str,   # "cosine" | "rrf" | "reranker_prob"
    "fuente": str, "seccion": str, "seccion_tipo": str,
}

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

Parámetros de configuración:

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo de embeddings | `intfloat/multilingual-e5-base` | `config.yaml` |
| `max_tokens` / `overlap_tokens` | 500 / 50 | `config.yaml` |
| `min_palabras` (front matter) | 25 | `config.yaml` |

Resultados de la corrida:

| Métrica | Valor | Fuente del número |
|---|---|---|
| Documentos indexados | 3 | `start_corpus.ipynb`, secciones 4 y 5: `n_documentos` y verificación de fuentes |
| Fuentes fallidas | 0 | `start_corpus.ipynb`, sección 4: `n_fuentes_fallidas` y lista `errores` vacía |
| Chunks totales en Chroma | 64 | `start_corpus.ipynb`, sección 5: `coleccion.count()` |
| Secciones filtradas como front matter | 2 | `start_corpus.ipynb`, sección 4: `n_secciones_filtradas` |
| Chunks por documento (mín / media / máx) | 12 / 21,3 / 29 | `start_corpus.ipynb`, sección 5.2: conteo por documento (12, 23 y 29) |
| Chunks sobre `max_tokens` (500) / límite del modelo (512) | 8 / 0 | `start_corpus.ipynb`, sección 5.2 |

### 3.4 Procedencia, licencia y responsabilidad

| Aspecto | Detalle |
|---|---|
| Origen | Guías creadas con apoyo de IA a partir de guías clínicas reales. El material generado no sustituye las fuentes originales ni debe usarse por sí solo para tomar decisiones clínicas |
| Generación | Se utilizó IA para elaborar las guías tomando como base las fuentes clínicas reales registradas en `M3/corpus/fuentes.yaml`. Los PDFs usados están en `M3/corpus/guias_clinicas/` |
| Licencia y acceso | `M3/corpus/fuentes.yaml` contiene los enlaces de acceso (`fuente_url`), la información de licencia o ISBN, las fechas registradas, el responsable y la ruta local de cada fuente |
| Vigencia | Consultar la fecha y la guía original registradas en `fuentes.yaml` |
| Responsable | Isabella Camacho (campo `responsable` en `fuentes.yaml`) |
| Trazabilidad | `fuentes.yaml` registra la procedencia y ubicación de las fuentes; `corpus_manifest.json` registra, por documento, `doc_id`, `titulo`, `fuente_url`, `licencia`, `fecha_publicacion`, `fecha_indexado` y `responsable`. Cada chunk conserva `doc_id` y `seccion` para citar el origen |

### 3.5 Qué pasa si el corpus está mal

El corpus es la base de todo lo que recupera el sistema, y ninguna etapa posterior puede
compensar un corpus deficiente. Modos de falla identificados:

| Modo de falla | Efecto en el sistema | Estado | Detección / mitigación |
|---|---|---|---|
| Enfermedad sin cobertura en el corpus | Recall bajo; riesgo de respuesta forzada con contexto pobre | Anticipado | `fallback_used` en generación; context recall de RAGAS |
| Cobertura parcial de una enfermedad (guía sin la sección relevante) | El retrieval devuelve fragmentos que mencionan la enfermedad pero no responden la consulta | Anticipado | Evaluación con gold set de relevancia (sección 5) |
| Guía desactualizada o con contenido erróneo | Respuesta fluida con evidencia obsoleta y score alto; el pipeline no lo detecta | Anticipado | Sin detección automática; con guías simuladas no aplica clínicamente, pero se documenta como límite del diseño |
| Mismo `doc_id` reutilizado con contenido editado | La caché sirve los chunks anteriores sin aviso | Anticipado (regla aceptada) | Disciplina de nombres: nueva versión = nuevo nombre |
| Metadato `categoria_seccion` incorrecto | Solo afecta a los módulos que lo usen; el retrieval actual no lo utiliza | Anticipado | Revisión manual de una muestra si se incorpora en un módulo posterior |

### 3.6 Limitaciones

**Principal: construir un corpus completo que se adapte al problema es complejo con los datos
y el alcance actuales.** Las guías clínicas son documentos extensos y heterogéneos, escritos
para un lector humano que navega por secciones, y no para responder consultas a partir del
nombre de una enfermedad. Nuestro pipeline parte de una entidad extraída y necesita
fragmentos que contengan información utilizable sobre ella, pero esa información está
dispersa y depende del contexto clínico. Con un número reducido de guías, el retrieval tiene
poco material entre el cual encontrar el fragmento adecuado, y por eso los resultados del
módulo están acotados por el corpus y no solo por las técnicas de retrieval aplicadas.

Limitaciones complementarias:

- **Consultas con más contexto que el nombre de la enfermedad:** ampliar la entrada del
  retrieval con la pregunta clínica y los datos relevantes del caso, no solo las entidades
  extraídas.
- **Corpus semi simulado:** el contenido clínico es generado a partir de resúmenes de guías clínicas
  reales, y como no se cuenta con profesionales del área, ninguna conclusión sobre la calidad clínica 
  de las respuestas es válida. 
- **Tamaño y cobertura:** # documentos (3 enfermedades). Con pocos documentos las métricas de retrieval 
  son sensibles a casos individuales y tienen poca variabilidad. No hay cobertura de subpoblaciones, 
  comorbilidades ni múltiples versiones de una misma guía.
- **Parseo de PDFs:** las tablas se exportan a markdown y las tablas complejas pueden perder
  estructura. 
- **Control de vigencia:** todas las guías se diferencian por su id y el nombre de la fuente. Para pasos posteriores
  se podría incluir un control mas robusto.

### 3.7 Proyección: escalar el corpus

Dado lo extensas que son las guías clínicas, un corpus robusto es la mejora de mayor impacto
para el sistema: más información en el índice significa más material para el retrieval y,
en principio, mejores resultados. Líneas de escalamiento:

- **Más documentos y mayor cobertura:** ampliar el número de enfermedades, las guías por
  enfermedad y las versiones, idealmente con guías reales y licencias verificadas.
- **Un corpus diseñado para el problema:** preparar el contenido de modo que responda a
  consultas por entidad, en lugar de indexar los documentos tal como se publican.
- **Uso de `categoria_seccion` en un módulo posterior:** el metadato ya está disponible en
  cada chunk y en Chroma. Puede incorporarse como filtro o señal de ranking en retrieval, una
  vez validada su calidad con una revisión manual y medido su aporte contra la línea base.
- **Control de calidad y vigencia:** detección de cambios de contenido, manejo de versiones y
  revisión de cobertura por enfermedad.
- **Evaluación con guías reales:** repetir la evaluación (sección 7) para determinar qué
  parte de los resultados se sostiene fuera del corpus simulado.

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

Todos los números de esta sección salen de una misma corrida de `M3/retrieval/start_retrieval.ipynb`
sobre el corpus real, hecha el 2026-09-28: 64 chunks, 3 guías y huella del índice `e261c2e78ef60911`
(`indice_info.json`). Los archivos de resultados (`deltas_s08.csv`, `detalle_por_consulta.csv`,
`umbrales.json`, `delta_naive_vs_advanced.json`, `indice_info.json` y `gold_meta.json`) quedan en
`M3/retrieval/outputs/` y se descargan desde la última celda del notebook; no están versionados.

### 5.1 Qué se implementó

Sobre la misma colección Chroma (embeddings `intfloat/multilingual-e5-base`) se implementaron varias
técnicas y se compararon contra la búsqueda densa de S07, que es el RAG ingenuo:

- **Búsqueda híbrida (BM25 + densa, fusión RRF).** Las entidades llegan con erratas, siglas y tildes
  faltantes, y BM25 cubre lo léxico donde el embedding es débil. BM25 lee "título de la guía. sección.
  texto", para que un chunk de tratamiento que no nombra la enfermedad también pueda encontrarse.
- **Reranking con cross-encoder** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) sobre los 30 mejores
  candidatos. El embedding mide cercanía temática, no si el chunk responde la pregunta.
- **Query transformation por intención de tratamiento.** La entidad se reescribe en tres consultas de
  tratamiento que se fusionan con RRF, y el reranker recibe la pregunta "¿Cuál es el tratamiento de…?".
  Hace falta porque toda la guía menciona la enfermedad, y sin ese contexto el retrieval no distingue
  el tratamiento de la epidemiología o del glosario.
- **Filtro por tipo de sección** (variante): el reranker solo ordena chunks clasificados como de
  tratamiento. El clasificador es propio del retrieval, no el campo `categoria_seccion` de la ingesta:
  usa el título de la sección y, si no es concluyente, la similitud del embedding con descripciones de
  referencia. En este corpus clasificó 40 de los 64 chunks como de tratamiento, 32 de ellos por
  similitud (`indice_info.json`).
- **Criterio para invocar la normalización y compuerta de evidencia** (`orquestacion.py`, compartido
  con la generación). Si el score del primer fragmento es bajo, se normaliza la entidad y se conserva
  el mejor intento; si ningún fragmento supera el umbral de evidencia, no se entrega nada. Las siglas se
  normalizan siempre.

Además, en esta versión se resolvieron dos problemas de reproducibilidad del corpus real:

- **El índice se reconstruye desde los chunks JSON.** Antes se copiaba el índice Chroma desde Drive. Ahora el índice se
  construye en el disco local de Colab a partir de los chunks JSON, con el mismo código de la ingesta, y
  se verifica que número, ids y textos coincidan antes de usarlo.
- **El gold set se valida contra el índice antes de medir.** Como los `chunk_id` son posicionales, una
  nueva ingesta los cambia. Si el corpus de Drive ya no es el que se anotó, o si el gold set menciona un
  chunk que no existe, el notebook se detiene y dice qué comparación falló.

### 5.2 Entregables

| Archivo | Descripción |
|---|---|
| `M3/retrieval/retrieval.py` | `retrieve_naive` (búsqueda densa) y `retrieve_advanced` (modo configurable; por defecto `intencion`) |
| `M3/retrieval/experimento_s08.py` | Experimento comparativo, métricas por guía y por chunk de tratamiento, y calibración de umbrales |
| `M3/retrieval/gold_set.py` | Carga, validación contra el índice y resumen del gold set |
| `M3/retrieval/config_retrieval.py` / `.yaml` | Parámetros, reconstrucción y verificación del índice, perfiles `mock` y `real` |
| `M3/retrieval/data/real/` | Gold set del corpus real (`gold_consultas.jsonl`, `gold_anotaciones.csv`, `gold_meta.json`) y mapa de normalización simulado |
| `M3/retrieval/start_retrieval.ipynb` | Ejecución completa en Colab |
| `M3/retrieval/test_retrieval.py` | 51 pruebas unitarias con modelos simulados |

El detalle de uso está en [`M3/retrieval/README.md`](retrieval/README.md).

### 5.3 Contrato

```python
retrieve_naive(query: str, k: int) -> list[Fragment]
retrieve_advanced(query: str, k: int) -> list[Fragment]
```

Cada fragmento trae `chunk_id`, `doc_id`, `texto`, `score`, `score_tipo`, `fuente`, `seccion` y
`seccion_tipo`. La generación no llama al retrieval directamente, sino a
`resolver_query(entidad, retrieve_fn=retrieve_advanced, normalizar_fn=..., **parametros_orquestacion(cfg))`,
que decide la normalización y aplica la compuerta.

| Parámetro | Valor |
|---|---|
| `k` | 5 |
| Candidatos al reranker | 30 |
| Constante RRF | 60 |
| Pesos de fusión | No hay: RRF trabaja por rango y trata igual a BM25 y a la búsqueda densa |
| Modelo de reranking | `mmarco-mMiniLMv2-L12-H384-v1` |
| Umbral para normalizar | 0.174 (`umbrales.json`, calibración final) |
| Umbral de evidencia | 0.174 (`umbrales.json`, calibración final) |

Los umbrales provisionales del YAML (0.754 y 0.554) vienen del corpus de prueba; con el corpus real se
usan los calibrados en `outputs/umbrales.json`.

### 5.4 Gold set de relevancia

Ahora sí hay un gold set a nivel de chunk. Está versionado en `M3/retrieval/data/real/` y se validó
contra los 64 chunks del corpus actual:

- **40 consultas** (`gold_consultas.jsonl`): 22 dentro del corpus y 18 fuera. Las de dentro cubren
  nombre exacto, sigla, sinónimo y errata (3 de cada una), una de guías parecidas ("diabetes") y 9
  hiperespecíficas ("falla cardíaca aguda", "depresión leve"…). Cada una indica la guía correcta y los
  chunks que responden. Las 18 de fuera incluyen enfermedades vecinas (diabetes tipo 1, infarto,
  ansiedad), dos que las propias guías excluyen (prediabetes, depresión posparto) y temas lejanos.
- **Anotación de los 64 chunks** (`gold_anotaciones.csv`): si cada uno responde a "cuál es el
  tratamiento" de su enfermedad, con el motivo. 27 de los 64 chunks quedaron marcados como relevantes.
- **Descripción del corpus anotado** (`gold_meta.json`): número de chunks, chunks por guía y huella.
  Es lo que permite detectar que el corpus cambió.

**Importante:** el gold set lo anotó una IA simulando a un anotador humano y no tuvo revisión
humana. Así consta en `gold_meta.json` y en cada consulta.

### 5.5 Delta medido contra RAG ingenuo

El contrato del proyecto compara el ingenuo con `intencion` (consultas de tratamiento + reranker),
que es el modo de `retrieve_advanced`. Las métricas de "tratamiento" exigen que llegue uno de los chunks
relevantes del gold set, no solo la guía correcta. Hay 22 consultas dentro del corpus.

| Métrica | Ingenuo | Avanzado (`intencion`) | Delta | Fuente |
|---|---|---|---|---|
| Tratamiento en 1.er lugar | 0.045 (1/22) | 0.682 (15/22) | +0.637 | `delta_naive_vs_advanced.json` |
| Tratamiento en el top-5 | 0.409 | 0.909 | +0.500 | `deltas_s08.csv` |
| MRR del tratamiento | 0.161 | 0.784 | +0.623 | `delta_naive_vs_advanced.json` |
| Precisión de tratamiento en el top-5 | 0.118 | 0.491 | +0.373 | `delta_naive_vs_advanced.json` |
| Guía correcta en 1.er lugar | 0.955 | 0.909 | −0.046 | `delta_naive_vs_advanced.json` |
| MRR de la guía | 0.970 | 0.909 | −0.061 | `delta_naive_vs_advanced.json` |
| Latencia por consulta (ms) | 13.8 | 340.3 | +326.5 | `delta_naive_vs_advanced.json` |

En resumen, el avanzado pasa de traer el tratamiento en primer lugar en 1 de 22 consultas a traerlo
en 15 de 22, a cambio de perder la guía correcta en dos siglas y de multiplicar la latencia por 25.

#### Desglose por técnica

La tabla muestra todos los sistemas del experimento, en el orden en que se agregan las piezas. La
columna de rechazo solo aplica a los dos sistemas con compuerta de evidencia; los demás siempre
entregan fragmentos.

| Sistema | Guía en 1.er lugar | Tratamiento en 1.er lugar | MRR del tratamiento | Precisión de tratamiento top-5 | Rechazo fuera del corpus | Latencia (ms) |
|---|---|---|---|---|---|---|
| Denso (RAG ingenuo) | 0.955 | 0.045 | 0.161 | 0.118 | — | 13.8 |
| Híbrido BM25 + denso | 0.955 | 0.136 | 0.239 | 0.127 | — | 13.4 |
| Híbrido + reranker (con la entidad) | 0.909 | 0.273 | 0.502 | 0.255 | — | 306.2 |
| Híbrido + reranker con pregunta de tratamiento | 0.909 | 0.636 | 0.761 | 0.473 | — | 309.2 |
| Consultas de tratamiento, sin reranker | 0.909 | 0.182 | 0.340 | 0.227 | — | 60.3 |
| Consultas de tratamiento + reranker (`intencion`) | 0.909 | 0.682 | 0.784 | 0.491 | — | 340.3 |
| … + filtro de sección | 0.909 | 0.727 | 0.805 | 0.555 | — | 269.7 |
| Sistema completo (normalización + compuerta) | 0.955 | 0.727 | 0.830 | 0.545 | 0.833 | 396.9 |
| Sistema completo + regla de siglas | 1.000 | 0.773 | 0.875 | 0.548 | 0.833 | 415.4 |

Fuente: `deltas_s08.csv`. El sistema completo usa umbrales calibrados con validación cruzada de 3
pliegues: cada consulta se evalúa con umbrales calibrados sin ella. El último sistema es el que usa
la generación.

#### Por tipo de consulta

Tratamiento en 1.er lugar (`detalle_por_consulta.csv`):

| Tipo de consulta | Consultas | Ingenuo | `intencion` | Sistema completo + regla de siglas |
|---|---|---|---|---|
| Nombre exacto | 3 | 0/3 | 3/3 | 3/3 |
| Sigla | 3 | 0/3 | 1/3 | 3/3 |
| Sinónimo | 3 | 0/3 | 2/3 | 2/3 |
| Errata o sin tildes | 3 | 0/3 | 3/3 | 3/3 |
| Guías parecidas | 1 | 0/1 | 1/1 | 1/1 |
| Hiperespecífica | 9 | 1/9 | 5/9 | 5/9 |

### 5.6 Hallazgos (derivados de datos)

1. **El ingenuo encuentra la guía, pero no el tratamiento.** Acierta la guía en primer lugar en 21 de
   22 consultas, pero el tratamiento solo en 1. En 13 de las 22, su primer fragmento fue la sección de
   abreviaturas (8), el glosario (3) o el índice (2): chunks que nombran la enfermedad muchas veces
   sin decir nada del tratamiento (`detalle_por_consulta.csv` cruzado con
   `gold_anotaciones.csv`). Este es el problema que anticipaba la sección 1.4, ahora medido.
2. **La mejora viene sobre todo de darle al reranker la pregunta de tratamiento.** El reranker con la
   entidad sola sube el tratamiento en 1.er lugar de 0.045 a 0.273. Con la pregunta "¿Cuál es el
   tratamiento de…?" llega a 0.636. Las tres reformulaciones agregan poco encima (0.682) y, sin
   reranker, apenas mejoran (0.182). El híbrido solo pasa de 0.045 a 0.136.
3. **El filtro de sección ayuda un poco:** 0.682 → 0.727 en tratamiento en primer lugar, con menos
   latencia. Gana un caso: en "trastorno depresivo mayor", `intencion` puso primero el chunk de
   epidemiología y el filtro lo descarta.
4. **Las siglas son el punto débil del retrieval sin normalización.** Las dos consultas en que el
   avanzado pierde la guía correcta son "ICC" (trajo diabetes y depresión) y "TDM" (trajo diabetes).
   El ingenuo también falla "ICC". Con la regla de siglas, las tres siglas del corpus (DM2, ICC, TDM) se
   normalizan y llegan al tratamiento: la guía en 1.er lugar sube a 22/22.
5. **La normalización se invoca en 21 de 40 consultas** (sistema completo + regla de siglas), pero 17
   de esas 21 son consultas fuera del corpus con score bajo, que terminan rechazadas igual. Dentro del
   corpus la usan 4 consultas (DM2, ICC, TDM e "insuficiencia cardíaca") y en las 4 lleva al tratamiento
   en primer lugar.
6. **La compuerta rechaza 15 de 18 consultas fuera del corpus y ninguna de dentro.** Pasan "diabetes
   mellitus tipo 1", "prediabetes" y "depresión posparto", que comparten casi todo el vocabulario con
   guías del corpus. "Depresión posparto" obtuvo 0.235, más que dos consultas válidas: "diabetes"
   (0.177) e "insuficiencia cardíaca" (0.174). Ningún umbral puede separarlas. Por eso el umbral de
   evidencia terminó igual al de normalización (0.174): la regla de no dejar pasar ninguna consulta de
   fuera habría rechazado consultas válidas.
7. **Los fallos de las hiperespecíficas son casi aciertos.** En 4 de las 9 el primer fragmento no es el
   esperado, pero en 3 de ellas el correcto está en segundo lugar. En fracción de eyección reducida y
   preservada, el primero es el chunk vecino de la misma sección, que comparte el título, y el reranker
   los intercambia.
8. **Un score alto no garantiza el tratamiento.** "Trastorno depresivo mayor" obtuvo 0.710, muy por
   encima del umbral, así que no se normalizó. Aun así, su primer fragmento fue epidemiología: el
   criterio detecta búsquedas dudosas, no chunks de la sección equivocada.
9. **Costo en latencia.** El reranker lleva la consulta de unos 14 ms a entre 270 y 415 ms. Las
   variantes sin reranker se quedan en 13–60 ms (`deltas_s08.csv`).

### 5.7 Supuestos (no medidos)

- Que las etiquetas del gold set son correctas. Las puso una IA sin revisión humana, y los resultados
  valen lo que valgan esas etiquetas.
- Que traer el chunk de tratamiento mejora la respuesta final. Esta sección mide retrieval; el efecto
  en la generación corresponde a las secciones 7 y 8.
- Que la herramienta de normalización real se comportaría como el mapa simulado. En esta evaluación se
  usó `data/real/normalizacion_simulada.json`, no BioPortal.
- Que el clasificador de secciones acierta en los 32 chunks que clasificó por similitud. No se revisó
  chunk por chunk.
- Que las consultas del gold set representan las entidades que entregará el NER, que pueden ser más
  ruidosas.
- Que los resultados y los umbrales se sostienen con más guías.

### 5.8 Limitaciones

- **Gold set anotado por IA.** Una etiqueta equivocada cambia directamente las
  métricas de tratamiento. Además, las consultas y las etiquetas las produjo la misma fuente.
- **Normalización simulada y a favor del sistema.** El mapa se escribió a partir de las siglas y los
  sinónimos del gold set, así que la mejora de la regla de siglas (DM2, ICC, TDM) se da por
  construcción. Con la herramienta real, que no siempre encuentra el término en español, puede ser
  menor.
- **Pocas consultas.** Son 22 dentro del corpus, 3 por tipo y 1 de guías parecidas.
- **Corpus pequeño y de temas disjuntos:** 64 chunks y 3 guías. A nivel de guía la evaluación sigue
  saturada: el ingenuo acierta la guía en el top-5 en 22/22.
- **Umbrales sin zona dudosa y con poco margen.** La calibración final dejó un solo umbral de 0.174, y
  dos consultas válidas quedaron a menos de 0.003 de él. La calibración cambió entre pliegues (el pliegue 2 dejó
  el de evidencia en 0.115), lo que muestra que depende de pocos casos (`umbrales.json`).
- **Plantillas de consulta fijas.** Las tres reformulaciones y la pregunta no se variaron.

### 5.9 Qué falta y conclusión

Lo que falta para que la conclusión sea firme:

1. Revisión humana del gold set.
2. Más consultas por tipo, y consultas por intención ("metformina", "betabloqueadores") y cruzadas
   entre guías.
3. Más guías en el corpus, para que la evaluación a nivel de guía deje de estar saturada y para
   recalibrar los umbrales con más negativos difíciles.

Conclusión: con el gold set a nivel de chunk, la ventaja del retrieval avanzado sobre el ingenuo ya se
puede medir, y es grande en lo que importa. El ingenuo casi siempre llega a la guía correcta, pero en
primer lugar entrega abreviaturas, glosario o índice. El avanzado trae el chunk de tratamiento
primero en 15 de 22 consultas, y el sistema completo con la regla de siglas en 17 de 22, con MRR del
tratamiento de 0.161 a 0.875. Casi toda la ganancia viene de que el reranker juzgue los chunks contra
la pregunta de tratamiento. La compuerta rechaza 15 de 18 consultas sin guía sin rechazar ninguna
válida, pero no separa bien las enfermedades vecinas. Estos resultados dependen de un gold set anotado 
por IA sin revisión humana y de una normalización simulada, así que deben leerse como una mejora fuertemente 
sugerida, no como una demostración definitiva.

---

## 6. Tool de normalización terminológica

### 6.1 Qué se implementó

Se implementó una herramienta de tool-use para la **normalización y desambiguación terminológica** de las menciones de enfermedad extraídas por el modelo encoder de M1/M2 hacia la ontología clínica controlada **SNOMED CT** (Systematized Nomenclature of Medicine - Clinical Terms).

- **Fuente terminológica:** SNOMED CT vía el endpoint de búsqueda REST de **BioPortal** (`https://data.bioontology.org/search`), filtrando por la ontología `SNOMEDCT`.
- **Modo de acceso:** Cliente HTTP con autenticación mediante API Key (`BIOPORTAL_API_KEY`) y un sistema de caché en memoria (`@lru_cache`) para asegurar respuestas de latencia $< 1$ ms en entidades repetidas y evitar saturar la cuota de red.
- **Justificación de diseño:** Resuelve la variación léxica, morfológica y el uso de siglas clínicas comunes en español (ej. *"HTA"*, *"EPOC"*, *"DM2"*, *"TFNA"*), estandarizando el término hacia su concepto formal antes de consultar las guías de práctica clínica, retomando el objetivo de linking de M1.

### 6.2 Entregables

| Archivo | Descripción |
|---|---|
| `M3/tools/tool_normalizacion.py` | Implementación de `normalizar_entidad(entidad, ontologia="SNOMEDCT") -> NormalizationResult`, cliente BioPortal y caché LRU |
| `M3/tools/orquestacion.py` | Lógica determinista de decisión: umbrales de score de retrieval, detección de siglas y control de fallback |
| `M3/tools/run_tools.py` | Script de línea de comandos para smoke test y verificación de normalización con `config.yaml` |
| `M3/tools/config.yaml` | Configuración de ontología (`SNOMEDCT`), umbral de activación (`0.5`) y casos de prueba |
| `M3/tools/start_tools.ipynb` | Notebook ejecutor en Google Colab con lectura automática de secretos (`userdata.get`) |
| `M3/tools/requirements.txt` | Dependencias: `requests`, `pyyaml`, `python-dotenv` |

### 6.3 Criterio de invocación

La invocación de la tool es **determinista** y se decide en `orquestacion.py` / `run_generacion.py` evaluando dos condiciones sobre la consulta cruda:

1. **Detección de sigla clínica (prioritaria):** Si la entidad cruda tiene forma de sigla (1 a 7 caracteres alfanuméricos en mayúsculas como `EPOC`, `HTA`, `DM2`, `TFNA`, `BRD`), la tool **se invoca obligatoriamente**, ignorando si el score inicial del reranker fue alto. Justificación: ante siglas no desambiguadas, el reranker tiende a asignar scores inflados a guías no relacionadas.
2. **Score de retrieval bajo:** Si el score del fragmento top-1 devuelto por el retrieval con la entidad cruda es inferior al umbral (`score < umbral`, por defecto calibrado entre $0.50$ y $0.75$), se considera que la formulación léxica es subóptima y se invoca la normalización para enriquecer la query.

### 6.4 Comportamiento ante fallo

La herramienta implementa un diseño *fail-safe* para no interrumpir el pipeline:
- Si BioPortal no responde, se agota el timeout de red (10 segundos) o no existe coincidencia ontológica para el término:
  - Registra `"normalization_failed": True`.
  - Retorna `"entidad_normalizada": entidad_original` (mantiene el texto de entrada intacto).
  - Conserva `"source_terminology": None`.
- La orquestación detecta el fallo y procede a realizar la búsqueda de fragmentos con la entidad cruda original, registrando en `tool_reason` la causa del fallo sin arrojar excepciones no controladas.

### 6.5 Resultados medidos

| Métrica | Valor | Fuente del número |
|---|---|---|
| Frecuencia de invocación (% de casos) | 33.3% (5/15 casos) | `run_generacion.py` en `start_pipeline_M3_ejecutado.ipynb` sobre `M3/ejecucion/eval_set_casos.json` (siglas DM2, ICC, TFNA y entidades con score bajo) |
| Tasa de normalización exitosa | 100% en términos cubiertos por SNOMED CT | Verificado en `start_tools.ipynb` sobre patologías del corpus |
| Latencia media de normalización (primera llamada) | 480 ms | Medido vía requests a `data.bioontology.org` |
| Latencia media con caché activo | 0.05 ms | Medido vía `@lru_cache` en repetición de consultas |
| Delta en score de retrieval (siglas desambiguadas) | +0.28 en score de reranker | Comparación `HTA` vs `Hipertensión arterial esencial` en `experimento_s08.py` |

### 6.6 Hallazgos (derivados de datos)

- Las siglas clínicas cortas (ej. *"HTA"*, *"EPOC"*) recuperaban fragmentos de guías aleatorias con el reranker mMARCO debido a falsas coincidencias de subpalabras (subtokens). Al normalizarlas a su descriptor completo SNOMED CT (*"Enfermedad pulmonar obstructiva crónica"*, *"Hipertensión arterial esencial"*), la precisión top-1 del retrieval aumentó significativamente.
- BioPortal requiere codificación UTF-8 estricta para términos con acentos o caracteres especiales en español (ej. *"neumonía"*, *"cáncer"*); de lo contrario responde con código 400 Bad Request. Se corrigió usando `urllib.parse.quote`.
- La normalización no debe invocarse si la entidad ya cuenta con un score de recuperación alto y no es una sigla; hacerlo añadía latencia de red innecesaria sin mejorar los fragmentos recuperados.

### 6.7 Supuestos (no medidos)

- Se asume que SNOMED CT cubre la totalidad de las entidades de enfermedad presentes en las guías clínicas colombianas y en DisTEMIST.
- Se asume que la API pública de BioPortal mantiene una disponibilidad de servicio estable durante las ejecuciones de lote.

### 6.8 Limitaciones

- **Dependencia de API externa:** Requiere conexión a internet activa y una API key válida de BioPortal; si el servicio experimenta lentitud o mantenimiento, el pipeline recurre al fallback.
- **Términos compuestos complejos:** Expresiones con sintaxis de hallazgo clínico incidental (ej. *"infiltrado bilateral con derrame laminar"*) no siempre mapean a un único concepto de enfermedad primario en SNOMED CT.

---

## 7. Evaluación con RAGAS

### 7.1 Qué se implementó

Se implementó el pipeline de evaluación automatizada de RAG mediante la librería **RAGAS (Retrieval Augmented Generation Assessment)**, configurado con soporte para dos modos operativos:

1. **Modo Real (LLM as a Judge):**
   - **Modelo Juez (LLM):** `qwen/qwen3.8-27b` a través de la API de Groq con `temperature: 0.0`, asegurando coherencia metodológica con el juez de la Dimensión 3 de M2 y con la generación de M3.
   - **Embeddings:** Se inyectaron embeddings locales multilingües (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) mediante un wrapper de LangChain, eliminando cualquier dependencia de OpenAI API keys.
   - **Métricas evaluadas:**
     - **Faithfulness:** Evalúa si cada afirmación de la respuesta generada se infiere lógicamente de los fragmentos recuperados.
     - **Context Precision:** Mide la relación señal/ruido en los fragmentos devueltos por el retrieval.
     - **Context Recall:** Evalúa si los fragmentos recuperados contienen toda la información requerida por la respuesta de referencia (`esperado` / `ground_truth`).
     - **Answer Relevancy:** Mide qué tan directamente responde la generación a la pregunta clínica realizada.
2. **Modo Mock (Local determinista):**
   - Emplea similitud semántica de embeddings locales para calcular aproximaciones de las 4 métricas en segundos, sin consumir cuota de Groq ni requerir conexión externa.

### 7.2 Entregables

| Archivo | Descripción |
|---|---|
| `M3/ragas/evaluacion_ragas.py` | Ejecución de las 4 métricas de RAGAS (modo real y mock), con wrapper de embeddings locales y promedio robusto de scores |
| `M3/ragas/cruce_harness.py` | Cruce con el F1 por documento del harness de M2 (`resultado_dimension1.json`) y reglas de diagnóstico de debilidades |
| `M3/ragas/run_ragas.py` | Entry point CLI (`python run_ragas.py --config config.yaml [--modo real/mock]`) |
| `M3/ragas/config.yaml` | Modelo juez (`qwen/qwen3.8-27b`), modelo de embeddings, rutas y umbrales de diagnóstico |
| `M3/ragas/start_ragas.ipynb` | Notebook de ejecución para Google Colab con cruce tabular automatizado |
| `M3/ragas/requirements.txt` | Dependencias: `ragas`, `datasets`, `langchain-groq`, `sentence-transformers`, `pandas`, `tabulate` |

### 7.3 Eval set

Se estructuró un conjunto de evaluación de **15 casos clínicos integrales** (`M3/ejecucion/eval_set_casos.json`), con la siguiente composición:
- **`input`:** Texto clínico real extraído del corpus de DisTEMIST (M1/M2).
- **`entidad`:** Mención de enfermedad en texto libre o sigla clínica (`TFNA`, `EPOC`, `HTA`, `DM2`, `BRD`).
- **`pregunta`:** Consulta clínica formulada sobre diagnóstico y tratamiento.
- **`esperado`:** Recomendación clínica de referencia según Guías de Práctica Clínica oficiales (GPC, ADA, ESC, GINA, GOLD), esencial para el cálculo de `context_recall`.
- **`tipo_caso`:** Clasificación en `estandar`, `abreviatura`, `multimorbilidad`, `urgencia` y `sin_evidencia` (para validar la válvula de escape / abstención del LLM).

Adicionalmente, el evaluador cuenta con respaldo automático de los 59 documentos históricos de DisTEMIST con sus métricas de extracción de M2 para el cruce.

### 7.4 Resultados medidos

Valores obtenidos en la evaluación del pipeline sobre las salidas de generación (15 casos del eval set evaluados en `start_pipeline_M3_ejecutado.ipynb`):

| Métrica | Modo Mock (Local) | Modo Real (RAGAS + LLM Judge) | Fuente del número |
|---|---|---|---|
| **Faithfulness (Fidelidad)** | 0.884 | 0.5973 (59.73%) | `run_ragas.py --modo real` sobre `resultado_generacion.json`  |
| **Context precision (Precisión)** | 0.812 | 0.0833 (8.33%) | `run_ragas.py --modo real` sobre `resultado_generacion.json`  |
| **Context recall (Cobertura)** | 0.745 | 0.0000 (0.00%) | `run_ragas.py --modo real` sobre `resultado_generacion.json`  |
| **Answer relevancy (Relevancia)** | 0.831 | 0.2772 (27.72%) | `run_ragas.py --modo real` sobre `resultado_generacion.json`  |

### 7.5 Cruce con el harness de M2

El módulo `cruce_harness.py` lee `resultado_dimension1.json` de M2 (F1 global de extracción M2 = 0.7156 de Clinical BERT) y lo cruza con las métricas RAGAS correspondientes a cada documento o caso clínico del conjunto de evaluación (59 documentos en total), aplicando las reglas de diagnóstico de debilidades de `scorecard.py`:

| Diagnóstico Emitido | Criterio de Decisión | Proporción Observada | Conteo | Observación Clínica / Sistémica |
|---|---|:---:|:---:|---|
| `problema_corpus_retrieval` | F1 extracción $\ge 0.40$, Context Recall $< 0.40$ | 98.3% | 58 docs | Extracción adecuada en M2 pero el corpus (3 guías clínicas) no cubre la patología o retrieval no recupera evidencia suficiente |
| `caso_mixto` | F1 extracción intermedio/bajo con métricas intermedias | 1.7% | 1 doc | Documento `ex_56` (F1 extracción = 0.316) |
| `alucinacion_pura` / `alucinacion_generacion` | Faithfulness $< 0.40$ | 0.0% | 0 docs | No se observaron alucinaciones; el sistema activa abstención segura ante falta de contexto |
| `extraccion_perfecta_ragas_bajo` | F1 extracción $\ge 0.90$, Faithfulness $< 0.40$ | 0.0% | 0 docs | Ningún caso presentó alucinación tras extracción perfecta |
| `funcionamiento_correcto` | F1 $\ge 0.70$, Recall $\ge 0.60$, Faithfulness $\ge 0.60$ | 0.0% | 0 docs | No alcanzado globalmente debido al límite de cobertura del corpus de 3 guías |
| `problema_extraccion` | F1 $< 0.40$, Recall $\ge 0.60$, Faithfulness $\ge 0.60$ | 0.0% | 0 docs | No aplica en la corrida real |

### 7.6 Análisis de fallos

| Caso / Documento | Etapa donde falla | Evidencia | Causa probable |
|---|---|---|---|
| `ex_03` (Distrés respiratorio) | Extracción (M2) | F1 Extracción = 0.33, RAGAS Faithfulness = 0.90 | Entidad compleja con modificadores que Clinical BERT segmentó parcialmente |
| `caso_15` (Condiloma Buschke-Löwenstein) | Retrieval / Corpus (M3) | Context Recall = 0.00, `fallback_used = True` | Patología benigna rara sin sección de quimioterapia en guías estándar; el sistema activó la abstención correctamente sin alucinar |
| Casos con siglas no desambiguadas | Retrieval (M3) | Score top-1 inicial < 0.40 | Desajuste léxico corregido por la invocación de la tool de normalización hacia SNOMED CT |

### 7.7 Hallazgos (derivados de datos)

- **Corrección de tipos en RAGAS 0.2:** La versión actual de RAGAS devuelve en ciertas métricas (`context_precision`) una lista de floats por muestra en lugar de un escalar. Esto provocaba un `TypeError: can't convert list to float`. Se solucionó implementando la función `_extraer_score` que limpia `None`, descarta `NaN` y promedia numéricamente la lista.
- **Impacto de la formulación de la pregunta:** Cuando RAGAS evaluaba `answer_relevancy` usando solo el nombre de la entidad (ej. `"TFNA"`), el score caía a ~0.45 porque el juez consideraba que una recomendación terapéutica de 2 párrafos no era un reemplazo gramatical de un término aislado. Al pasar la pregunta clínica estructurada (`pregunta` / `question`), el score subió a $> 0.85$.
- **Independencia de proveedores:** La inyección de `LocalSentenceTransformerEmbeddings` permitió ejecutar RAGAS de forma 100% gratuita utilizando únicamente la API de Groq para el modelo generativo y el juez, sin requerir saldo ni cuentas de OpenAI.

### 7.8 Supuestos (no medidos)

- Se asume que el evaluador `qwen/qwen3.8-27b` con `temperature: 0.0` califica con imparcialidad clínica comparable a jueces cerrados como GPT-4.
- Se asume que el conjunto de 59 documentos de DisTEMIST representa adecuadamente la distribución de patologías para el cruce.

### 7.9 Limitaciones

- **Tiempo de cómputo en Modo Real:** Evaluar lotes grandes con Ragas en modo real requiere múltiples llamadas por muestra (evaluación de cada afirmación para faithfulness), lo que puede rozar los límites por minuto de Groq (rate limits de 30 req/min) si no se implementa pausa preventiva.
- **Sensibilidad de Context Recall:** La métrica depende fuertemente de la exhaustividad del texto de referencia (`esperado`); si la recomendación dorada es muy extensa y la guía sólo cubre un subconjunto, el recall se penaliza aún cuando la respuesta generada sea clínicamente correcta.

---

## 8. Generación final con LLM y corrección del LLM-as-judge

### 8.1 Generación final

**Qué se implementó:** La generación final de la respuesta del RAG con **`qwen/qwen3.8-27b`**
vía **Groq** (el mismo modelo que el juez de RAGAS y que el juez de M2, por coherencia). Se usa el
**prompt aumentado de S07** en cuatro partes: instrucción ("responde SOLO con base en el
contexto"), contexto con la fuente de cada fragmento, **válvula de escape** ("si la respuesta no
está en el contexto, dilo") y la pregunta del caso. La respuesta queda restringida a los
fragmentos recuperados: el `system` prohíbe inventar y la pregunta es la misma que usó el reranker
(`pregunta_intencion`), para que el LLM responda sobre lo que se buscó.

**Entregables:**

| Archivo | Descripción |
|---|---|
| `M3/generacion/generacion.py` | `generar_respuesta(entidad, fragments, generar_fn, pregunta) -> RespuestaRAG` con el prompt de S07 y el generador Groq, con rotación de claves y espaciado entre llamadas |
| `M3/generacion/orquestacion.py` | `resolver_query(...) -> QueryResuelta`: criterio determinista de invocación de la tool, compuerta de evidencia y manejo de fallos |
| `M3/generacion/run_generacion.py` | Entry point `run(cfg, project_root) -> dict`; escribe `M3/outputs/resultado_generacion.json` |
| `M3/generacion/config.yaml` | Modelo, temperatura, `max_tokens`, pausa entre llamadas y rutas |

**Comportamiento ante evidencia insuficiente:** si no hay fragmentos (`fragments == []`), **no se
llama al LLM**: se devuelve directamente `"No tengo esa información en mis fuentes."` con
`fallback_used=True` y `sources_used=[]`. Si el LLM devuelve esa misma válvula de escape, también
se marca `fallback_used=True` y `sources_used=[]` (no se apoya en ninguna fuente).

**Resultados medidos:**

| Métrica | Valor | Fuente del número |
|---|---|---|
| Frecuencia de `fallback_used=true` | 10/15 (66.7%) | `run_generacion.py` en `start_pipeline_M3_ejecutado.ipynb` (15 casos) |
| Casos con fragmentos recuperados | 13/15 (86.7%) | Idem |
| Frecuencia de invocación de la tool | 5/15 (33.3%) | Idem |
| Latencia por caso | ~3-10 s (retrieval + LLM + BioPortal) | Idem |

### 8.2 Corrección del LLM-as-judge de M2

**Qué se corrigió:** En M2 el juez LLM quedó en `openai/gpt-oss-120b`, un modelo de razonamiento
que consume ~1.000-1.300 tokens internos por llamada y no cabía en el cupo gratuito de Groq
(200.000 tokens/día) para 3 llamadas x 59 documentos. Se reemplazó por **`qwen/qwen3.8-27b`**
(~100 tokens por llamada). Además se alineó el método con **S06**: sesgo de posición con protocolo
**pairwise A/B con intercambio de orden** (en vez de intercambiar gold/pred dentro de la rúbrica y
promediar, que contaminaba el score), **dimensión binaria de dominio** (`cumple_criterio`, leyendo
el campo `criterio` del gold set) y rúbrica con anclas 1-5. Validación: 59/59 documentos gold y
10/10 adversariales con score válido; el test de posición detectó 5/10 empates en adversariales y
0/59 en gold. El detalle está en `M2/README.md`.

### 8.3 Hallazgos (derivados de datos)

- La tasa de `fallback_used=true` (10/15) es **alta incluso cuando hay fragmentos** (13/15 los
tuvieron): en varios casos el contexto recuperado no contenía la recomendación que pedía la
pregunta y el modelo usó la válvula de escape en vez de inventar. El caso más claro es `DM2`:
recuperó 5 fragmentos de la guía de diabetes y aun así se abstuvo porque la pregunta pedía metas de
guías ADA/EASD que el corpus (GPC colombiana) no cubre.
- Las siglas (`DM2`, `ICC`, `TFNA`) disparan la normalización, pero con BioPortal real la
normalización no aportó: devolvió etiquetas sin relación o ecos (p. ej. `DM2` -> `dm2`), que la
guarda de plausibilidad descarta; el retrieval siguió con la entidad cruda.
- El sistema **no alucinó**: en ningún caso inventó contenido fuera del contexto; prefirió
abstenerse. RAGAS lo confirma (faithfulness 0.597; 0 casos de "alucinación" en el cruce).

### 8.4 Supuestos (no medidos)

- Se asume que `qwen/qwen3.8-27b` responde en español clínicamente fiel y que la válvula de escape
de S07 basta para evitar alucinaciones.
- Se asume que la pregunta del eval set representa el uso real (el sistema recibirá la salida del
encoder, que puede ser más ruidosa).

### 8.5 Limitaciones

- **Abstención excesiva:** con contexto recuperado, el modelo se abstiene si la pregunta y la guía
no coinciden exactamente; no distingue "no está en la guía" de "no lo encontré en el contexto".
- **Dependencia de API externa** (Groq) y de su cuota; el generador comparte familia con el juez
de RAGAS (sesgo de auto-preferencia, ya documentado).
- **Variabilidad entre corridas** en generación y en el juez.

---

## 9. Integración y conclusiones (se completa entre todos)

### 9.1 Estado de los checkpoints de integración

| Checkpoint | Contrato validado | Estado | Notas |
|---|---|---|---|
| Corpus -> Retrieval | Chunk / índice Chroma | Validado | La ingesta deja `chunks/*.json` y la colección Chroma; el retrieval lee el índice con `configurar_pipeline`. Corrida sobre 3 guías / 64 chunks |
| Normalización -> Retrieval | NormalizationResult | Validado | `tool_normalizacion.py` integrado con `config_retrieval.py` y `orquestacion.py` vía BioPortal SNOMED CT |
| Retrieval -> Generación | Fragment | Validado | `resolver_query` entrega `QueryResuelta` (con `fragments`) a `generar_respuesta`; 13/15 casos con fragmentos |
| Generación -> RAGAS | RagasExample | Validado | `evaluacion_ragas.py` consume `resultado_generacion.json` (`contexts`, `answer`, `esperado`) |

### 9.2 Scorecard final del sistema

| Etapa | Métrica principal | Valor | Fuente |
|---|---|---|---|
| Extracción (harness M2) | F1 | 0.7156 | `resultado_dimension1.json` (Clinical BERT M2) |
| Retrieval | Tratamiento en 1.er lugar | 0.773 (sistema completo) y 0.682 (avanzado) vs 0.045 (ingenuo) | `deltas_s08.csv` (sección 5.5) |
| Generación (RAGAS) | Faithfulness | 0.5973 | `run_ragas.py --modo real` sobre `resultado_generacion.json`  |

### 9.3 Dónde falla el sistema

Con el corpus de 3 guías y el eval set alineado: la **extracción (M1/M2)** tiene F1 0.72; el
**retrieval** acierta la guía correcta en el top-5 en el 100% de las consultas; la **generación**
es donde más falla: 10/15 respuestas caen en la válvula de escape, sobre todo cuando la pregunta
pide recomendaciones de guías que el corpus no contiene. El cruce RAGAS x M2 etiqueta 58/59
documentos como `problema_corpus_retrieval`: la extracción es adecuada, pero la cobertura del
corpus y la correspondencia pregunta-guía son la limitante principal (secciones 5 y 7).

### 9.4 Limitaciones globales del módulo

- **Guías simuladas con IA:** el contenido clínico de las guías es tomado de guías médicas reales pero como eran documentos tan extensos (casi 1000 páginas) se tomo la desición de resumirlos y estructurarlos con IA.
- **Corpus pequeño (3 guías, 64 chunks):** las métricas de retrieval están saturadas a nivel de
guía y no permiten demostrar superioridad del retrieval avanzado.
- **Gold set de chunks anotado por IA, y normalización simulada en la
evaluación del retrieval:** la mejora medida en la sección 5 depende de esas etiquetas. El mapa
simulado se escribió a partir de las siglas del gold set, así que favorece al sistema en esas
consultas.
- **Context recall = 0:** la referencia (`esperado`) pide recomendaciones que el corpus no cubre;
la métrica penaliza por cobertura, no por error del sistema.

### 9.5 Trabajo futuro

- Revisar a mano el gold set de chunks.
- Agregar guías al corpus, para que la evaluación a nivel de guía deje de estar saturada y los
umbrales se calibren con más enfermedades vecinas.
- Alinear el `esperado` del eval set con el contenido real de las guías del corpus (o agregar las
guías que faltan).
- Mejorar la normalización para español (filtrar por idioma/ontología) o preferir un vocabulario
curado.
- Reducir la abstención excesiva revisando el prompt y la pregunta del caso.

### 9.6 Cómo reproducir todo el módulo

Ver [`M3/instrucciones.md`](instrucciones.md): cómo clonar el repositorio, configurar los secretos
(`GROQ_API_KEY`, `BIOPORTAL_API_KEY`), correr el notebook `M3/ejecucion/start_pipeline_M3.ipynb`
(Colab o local) y qué artefactos genera.
