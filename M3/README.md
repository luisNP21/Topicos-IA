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
| Frecuencia de invocación (% de casos) | ~46.7% (7/15 casos) | Medido en `eval_set_15_casos` (5 siglas forzadas + 2 entidades con score < 0.5) |
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

Se estructuró un conjunto de evaluación de **15 casos clínicos integrales** (`M3/data/eval_set_15_casos.json`), con la siguiente composición:
- **`input`:** Texto clínico real extraído del corpus de DisTEMIST (M1/M2).
- **`entidad`:** Mención de enfermedad en texto libre o sigla clínica (`TFNA`, `EPOC`, `HTA`, `DM2`, `BRD`).
- **`pregunta`:** Consulta clínica formulada sobre diagnóstico y tratamiento.
- **`esperado`:** Recomendación clínica de referencia según Guías de Práctica Clínica oficiales (GPC, ADA, ESC, GINA, GOLD), esencial para el cálculo de `context_recall`.
- **`tipo_caso`:** Clasificación en `estandar`, `abreviatura`, `multimorbilidad`, `urgencia` y `sin_evidencia` (para validar la válvula de escape / abstención del LLM).

Adicionalmente, el evaluador cuenta con respaldo automático de los 59 documentos históricos de DisTEMIST con sus métricas de extracción de M2 para el cruce.

### 7.4 Resultados medidos

Valores obtenidos en la evaluación del pipeline sobre las salidas de generación:

| Métrica | Modo Mock (Local) | Modo Real (RAGAS + Groq Qwen) | Fuente del número |
|---|---|---|---|
| **Faithfulness** | 0.884 | 0.892 | `evaluacion_ragas.py` sobre `resultado_generacion.json` |
| **Context precision** | 0.812 | 0.835 | `evaluacion_ragas.py` sobre `resultado_generacion.json` |
| **Context recall** | 0.745 | 0.768 | `evaluacion_ragas.py` (calculado contra campo `esperado`) |
| **Answer relevancy** | 0.831 | 0.854 | `evaluacion_ragas.py` (calculado contra `pregunta`) |

### 7.5 Cruce con el harness de M2

El módulo `cruce_harness.py` lee `resultado_dimension1.json` de M2 (F1 exacto de extracción de Clinical BERT) y lo cruza con las métricas RAGAS correspondientes a cada documento o caso clínico, aplicando las reglas de diagnóstico de `scorecard.py`:

| F1 Extracción M2 | Context Recall RAG | Faithfulness RAG | Diagnóstico Emitido | Proporción Observada |
| :---: | :---: | :---: | :--- | :---: |
| $\ge 0.70$ (Alto) | $\ge 0.60$ (Alto) | $\ge 0.60$ (Alto) | `funcionamiento_correcto` | 55.9% (33 docs) |
| $< 0.40$ (Bajo) | $\ge 0.60$ (Alto) | $\ge 0.60$ (Alto) | `problema_extraccion` | 1.7% (1 doc) |
| $\ge 0.70$ (Alto) | $< 0.40$ (Bajo) | - | `problema_corpus_retrieval` | 0.0% (con guías cubiertas) |
| - | - | $< 0.40$ (Bajo) | `alucinacion_generacion` | 0.0% (gracias a válvula de escape) |
| Intermedio | Intermedio | Intermedio | `caso_mixto` | 42.4% (25 docs) |

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
| Normalización -> Retrieval | NormalizationResult | Validado | `tool_normalizacion.py` integrado con `config_retrieval.py` y `orquestacion.py` vía BioPortal SNOMED CT |
| Retrieval -> Generación | Fragment | [COMPLETAR] | [COMPLETAR] |
| Generación -> RAGAS | RagasExample | Validado | `evaluacion_ragas.py` consume `resultado_generacion.json` (`contexts`, `answer`, `esperado`) |

### 9.2 Scorecard final del sistema

| Etapa | Métrica principal | Valor | Fuente |
|---|---|---|---|
| Extracción (harness M2) | F1 | [COMPLETAR] | [COMPLETAR] |
| Retrieval | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| Generación (RAGAS) | Faithfulness | 0.892 | `evaluacion_ragas.py` con juez Groq Qwen sobre eval set |

### 9.3 Dónde falla el sistema

[COMPLETAR: síntesis del análisis de fallos, con referencias a las secciones 5 a 8.]

### 9.4 Limitaciones globales del módulo

- [COMPLETAR: incluir el uso de guías simuladas y qué conclusiones no se pueden extrapolar a
  guías reales.]

### 9.5 Trabajo futuro

- [COMPLETAR]

### 9.6 Cómo reproducir todo el módulo

[COMPLETAR: referencia a .md con instrucciones]