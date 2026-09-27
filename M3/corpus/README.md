# M3 — Pipeline de Corpus: Procedencia y Responsabilidad

---

## 1. Qué hace este módulo

Toma PDFs de guías clínicas, extrae su contenido preservando la estructura del documento,
descarta secciones sin valor informativo (portadas, índices, resúmenes), las trocea en
fragmentos (`chunks`) categorizados por tipo de contenido, genera sus embeddings, y los deja
persistidos en dos formas: JSON versionable (fuente de verdad, auditable) y una colección
ChromaDB (índice de runtime para consultas de retrieval).

El resultado de este pipeline es el insumo de entrada de **retrieval**, y transitivamente
de **tool de normalización** y **generación final**.

---

## 2. Archivos creados

| Archivo | Responsabilidad | Quién lo importa |
|---|---|---|
| `ingesta.py` | `cargar_fuentes`, `ingerir_documento` (parseo con Docling, maneja texto y tablas), `filtrar_secciones`/`es_front_matter` | Solo el pipeline de corpus |
| `chunking.py` | `cargar_tokenizer`, `chunk_seccion`, `renumerar_chunks`, `clasificar_seccion` | Solo el pipeline de corpus |
| `embeddings.py` | `cargar_modelo_embeddings`, `embeber_chunks`, `embeber_query` | Pipeline de corpus **y** retrieval |
| `corpus_store.py` | `construir_indice_chroma`, `cargar_coleccion`, `guardar_chunks_json` | Pipeline de corpus **y** retrieval |
| `config_utils.py` | `cargar_config`, `construir_patron_front_matter`, `construir_patrones_categoria` | Solo el pipeline de corpus |
| `pipeline_corpus.py` | Orquestador: `run(cfg, project_root) -> dict`, encadena ingesta → chunking → embeddings → store | Solo `run_corpus.py` |
| `run_corpus.py` | Entry point de línea de comandos, resuelve `.env`/`PROJECT_ROOT`, llama a `run()` | Se ejecuta desde el notebook |
| `config.yaml` | Configuración centralizada: modelo de embeddings, parámetros de chunking, patrones de front matter y de categorización, paths de salida | Todo el equipo debería leerlo antes de tocar parámetros |
| `fuentes.yaml` | Registro de procedencia de cada guía clínica (dato, no configuración) | Cualquiera que audite el corpus |
| `requirements.txt` | Dependencias versionadas (`docling`, `sentence-transformers`, `chromadb`, `pyyaml`, `python-dotenv`, `transformers`, `torch`) | Cualquiera que corra el pipeline |

**Nota de organización de código:** `embeddings.py` y `corpus_store.py` se dejaron
deliberadamente sin dependencia de Docling — son los dos únicos archivos que el retrieval necesita
importar. No necesita instalar Docling ni tocar `ingesta.py`/`chunking.py`.

---

## 3. Técnicas y criterios de chunking

| Etapa | Técnica | Herramienta | Criterio |
|---|---|---|---|
| Ingesta | Extracción estructural (no solo texto plano) | **Docling** | Preserva jerarquía real del documento (encabezados, tablas exportadas a markdown) — corre local, sin API key ni límite de cuota |
| Filtro de front matter | Regex sobre título de sección + umbral de densidad de texto | Patrón construido desde `config.yaml → front_matter` | Descarta portada/resumen/abstract/índice/glosario/etc. por título, y cualquier sección con menos de `min_palabras` (default 25) por ser probablemente boilerplate |
| Chunking | Section-aware, corte real por tokens (no por palabras estimadas) con overlap | Tokenizer del mismo modelo de embeddings (`AutoTokenizer`) | Respeta los límites de sección real del documento antes de sub-dividir por `max_tokens`/`overlap_tokens` — evita cortar una recomendación clínica a la mitad |
| Categorización de sección | Regex por palabras clave sobre el título | Patrones construidos desde `config.yaml → categorias_seccion` | Clasifica cada sección en `tratamiento` \| `diagnostico` \| `epidemiologia` \| `otro`, para que retrieval pueda filtrar por intención sin depender de títulos crudos inconsistentes entre guías |
| Embeddings | Modelo multilingüe entrenado para retrieval (no similaridad genérica) | `intfloat/multilingual-e5-base` vía `sentence-transformers` | Modelo asimétrico: requiere prefijo `"passage: "` al indexar y `"query: "` al buscar — **ver sección 6, es crítico para quien consuma el índice** |

**Por qué se descartaron alternativas más simples:** un chunking puramente por palabras
(como en M1/M2) ignora la estructura semántica de una guía clínica. Un chunking semántico
completo (por similaridad de embeddings) sería más potente pero es complejidad innecesaria
para esta etapa — el punto medio elegido es cortar primero por sección real, y solo
sub-dividir si excede el tamaño máximo de tokens.

---

## 4. Estructura final del chunk

```python
Chunk = {
    "chunk_id": str,           # ej. "gpc_diabetes_2023_chunk14" — único a nivel de documento completo
    "doc_id": str,             # FK al manifest — de ahí sale la trazabilidad de licencia/fuente
    "texto": str,               # texto del chunk; incluye tablas exportadas a markdown si las hay
    "seccion": str,             # título crudo de la sección — para citar la fuente exacta en la respuesta final
    "categoria_seccion": str    # "tratamiento" | "diagnostico" | "epidemiologia" | "otro"
}
```

Este es el contrato que persiste tanto en los JSON (`data/guias_clinicas/chunks/*.json`) como
en la metadata de cada punto en ChromaDB.

---

## 5. Configuración

Toda la configuración del pipeline vive en un único archivo, `config.yaml`, para que no haya
que tocar código para cambiar un parámetro:

```yaml
embeddings:
  model_name: "intfloat/multilingual-e5-base"

chunking:
  max_tokens: 350
  overlap_tokens: 50

front_matter:
  min_palabras: 25
  titulos_excluidos:
    - portada
    - resumen
    - abstract
    - índice
    # ... (lista completa en el archivo)

categorias_seccion:
  tratamiento:
    - tratamiento
    - manejo
    - terapéutic
    - farmacológic
    - intervención
  diagnostico:
    - diagnóstico
    - criterios diagnósticos
    - evaluación clínica
  epidemiologia:
    - epidemiología
    - prevalencia
    - incidencia
    - factores de riesgo

paths:
  chunks_dir: "data/guias_clinicas/chunks"
  chroma_dir: "data/chroma_guias"
  manifest_path: "data/guias_clinicas/corpus_manifest.json"
```

`fuentes.yaml` es un archivo aparte (dato, no configuración): registra la procedencia de cada
guía clínica individual.

```yaml
fuentes:
  - doc_id: "gpc_diabetes_2023"
    titulo: "Guía de práctica clínica para diabetes tipo 2"
    fuente_url: "https://..."
    licencia: "CC BY-NC 4.0"
    fecha_publicacion: "2023-05-01"
    responsable: "Isabella Camacho"
    archivo_local: "data/guias_clinicas/GPC_diabetes_mellitus_tipo_2.pdf"  # relativo a PROJECT_ROOT, no absoluto
```

---

## 6. Uso — cómo correr el pipeline desde Colab

El código vive versionado en git (rama `corpus`); los datos (PDFs) y los outputs (chunks,
Chroma, manifest) viven en Drive. `PROJECT_ROOT` conecta ambos vía `.env`, nunca se hardcodea
dentro del código.

```python
# Celda 1 — clonar el repo
!git clone --branch corpus --single-branch https://github.com/luisNP21/Topicos-IA.git /content/corpus
%cd /content/corpus
!pip install -r requirements.txt -q
```

```python
# Celda 2 — montar Drive y generar .env
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
PROJECT_ROOT = Path("/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M3")

with open("/content/corpus/.env", "w") as f:
    f.write(f"PROJECT_ROOT={PROJECT_ROOT}\n")
```

```python
# Celda 3 — correr el pipeline
!python run_corpus.py --config config.yaml --fuentes fuentes.yaml
```

`run_corpus.py` resuelve `config.yaml`/`fuentes.yaml` **relativo al propio script** (no a
Drive), así que funciona igual sin importar en qué runtime se clone el repo. Los `archivo_local`
dentro de `fuentes.yaml` sí se resuelven contra `PROJECT_ROOT`, porque los PDFs viven en Drive.

**Salida esperada (`stats`):**
```python
{
    "n_fuentes_ok": int,
    "n_fuentes_fallidas": int,
    "n_documentos": int,
    "n_chunks": int,
    "n_secciones_filtradas": int,
    "errores": [{"doc_id": str, "error": str}]
}
```
Una fuente fallida no detiene el resto del lote — revisen siempre `stats["errores"]` antes de
dar el corpus por completo.

---

## 7. Funciones para retrieval 

### `cargar_coleccion(persist_dir, collection_name="guias_clinicas")`

Abre la colección de Chroma ya construida (no la reconstruye). Lanza error explícito si no
existe todavía — evita crear una colección vacía por accidente.

```python
from corpus_store import cargar_coleccion
coleccion = cargar_coleccion(persist_dir=f"{PROJECT_ROOT}/data/chroma_guias")
```

### `embeber_query(query, modelo_embed)`

Convierte una query (ya transformada con intención, ej. `"tratamiento y manejo farmacológico
de diabetes tipo 2"`) en vector, usando el prefijo `"query: "` requerido por el modelo
asimétrico.

```python
from embeddings import embeber_query, cargar_modelo_embeddings
modelo_embed = cargar_modelo_embeddings("intfloat/multilingual-e5-base")
query_emb = embeber_query("tratamiento y manejo farmacológico de diabetes tipo 2", modelo_embed)
```

### `construir_indice_chroma(chunks, embeddings, persist_dir, collection_name="guias_clinicas")`

Solo la usa el pipeline de corpus (o quien quiera reconstruir el índice desde los JSON). Pau
no debería llamarla — su punto de entrada es `cargar_coleccion`.

---

## 8. Detalles críticos que otros módulos deben respetar

- **Prefijo asimétrico del modelo de embeddings**: al indexar se usa `"passage: "`, al buscar
  `"query: "`. Si alguien busca sin el prefijo correcto, el ranking se degrada sin error
  visible — es el tipo de bug silencioso que no se detecta hasta comparar contra baseline.
- **Mismo `model_name` en ambos lados**: quien construya sus propios embeddings de query debe
  leer `config["embeddings"]["model_name"]` del mismo `config.yaml`, nunca hardcodear el
  nombre del modelo por separado.
- **Rutas y nombre de colección fijos**: `persist_dir` y `collection_name` deben coincidir
  exactamente con lo que generó el pipeline — se acuerdan como constantes, no se inventan.
- **`categoria_seccion` no es infalible**: es una clasificación por palabras clave sobre el
  título de sección, no una clasificación semántica del contenido. Si el filtro deja el pool
  vacío para una entidad/categoría, hay que caer a búsqueda sin filtro (ver función de
  fallback arriba) en vez de fallar.
- **Reproducibilidad del índice**: `construir_indice_chroma` usa `upsert` — correr el pipeline
  de nuevo sobre las mismas fuentes actualiza, no duplica. Si alguien corre su parte antes de
  que el corpus esté completo, solo verá el subconjunto indexado hasta ese momento — hay que
  avisar cuándo el corpus queda "congelado" para pruebas comparables.
- **Qué pasa si el corpus está mal** (cobertura insuficiente o guía desactualizada): no hay
  detección automática de esto en el pipeline actual — es un modo de falla que debe
  observarse en la evaluación (RAGAS — context recall bajo, o `fallback_used=true` con
  frecuencia alta en la generación) y documentarse con números, no solo señalarse como
  limitación teórica.

---

## 9. Integración con los siguientes módulos

```
Corpus
  data/guias_clinicas/chunks/*.json   (fuente de verdad, versionable)
  data/chroma_guias/                  (índice de runtime)
  data/guias_clinicas/corpus_manifest.json
        │
        ▼
Retrieval avanzado
  usa: cargar_coleccion, embeber_query, filtro por categoria_seccion
  produce: fragments recuperados (naive vs. híbrido+reranking) + delta medido
        │
        ▼
Tool de normalización + RAGAS
  normalización: se invoca cuando el retrieval con la entidad cruda da score bajo
  RAGAS: consume contexts + answer para faithfulness/
  context precision/recall/answer relevancy
        │
        ▼
Generación final
  usa: fragments ya resueltos (post normalización/reranking) + entidad
  produce: respuesta con sources_used (chunk_ids) trazables hasta el manifest,
  y fallback_used=true si no hay evidencia suficiente
```

Cualquier duda sobre el contrato de un módulo específico, revisar primero este documento antes
de asumir el formato de entrada/salida de otra parte del pipeline.