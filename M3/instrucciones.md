# M3 · Instrucciones de ejecución

Cómo reproducir el pipeline completo del Módulo 3 (corpus → retrieval → generación →
RAGAS + cruce con el harness de M2) con el notebook integrador
[`M3/ejecucion/start_pipeline_M3.ipynb`](ejecucion/start_pipeline_M3.ipynb).

El notebook funciona **igual en Google Colab y en local**: detecta el entorno solo.

---

## 1. Requisitos

- **Colab** (ruta principal): cuenta de Google y, preferiblemente, GPU T4
  (*Entorno de ejecución → Cambiar tipo → T4 GPU*) para el reranker del retrieval.
- **Local** (alternativa): Python 3.10+ en un venv. El notebook instala las dependencias solo.
- Cuenta gratuita en **Groq** (para el LLM generador y el juez de RAGAS).
- Cuenta gratuita en **BioPortal** (para la tool de normalización SNOMED CT). Opcional: si no hay
  key, el sistema usa un mapa de normalización simulado y la corrida continúa.

---

## 2. Secretos

| Variable | Para qué | Obligatoria |
|---|---|---|
| `GROQ_API_KEY` | Generación final y juez de RAGAS | Sí (modo real) |
| `BIOPORTAL_API_KEY` | Tool de normalización terminológica (BioPortal/SNOMED CT) | No (hay mapa simulado) |

- **Colab:** *panel lateral de la llave* → *Secrets* → agregar `GROQ_API_KEY` y
  `BIOPORTAL_API_KEY`, y activar *Acceso de notebook*. El notebook los lee solo.
- **Local:** exportarlas en el entorno o dejarlas en un `.env` (el notebook las recoge; el `.env`
  no se versiona).

---

## 3. Datos

- **Colab:** el corpus y el índice viven en Drive, en
  `MyDrive/TopicosIA/Proyecto-Salud/M3/data/`. El notebook copia las guías PDF desde el repo a
  `data/guias_clinicas/` si no están. Si prefieres otra ruta, edita `DATA_ROOT` en la celda 1.
- **Local:** por defecto usa `<raíz-del-repo>/../m3_data`. Se puede cambiar con la variable de
  entorno `M3_DATA_ROOT`.

---

## 4. Cómo correrlo

### En Google Colab

1. Abre el notebook desde GitHub:

   `https://colab.research.google.com/github/luisNP21/Topicos-IA/blob/main/M3/ejecucion/start_pipeline_M3.ipynb`

   (o *Archivo → Abrir notebook → GitHub → `luisNP21/Topicos-IA` → rama `main` →
   `M3/ejecucion/start_pipeline_M3.ipynb`).

2. Activa la GPU y agrega los secretos (sección 2).
3. Ejecuta todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*).

### En local

1. Ten el repo y un venv activo.
2. Configura los secretos (sección 2).
3. Abre el notebook con Jupyter/VS Code y ejecuta todas las celdas en orden.

> Las celdas son Python puro (sin `%pip`/`!`), así que también se pueden correr como script.

---

## 5. Qué hace, en orden

| Celda | Paso | Comando / entry point |
|---|---|---|
| 1 | Entorno y parámetros | — |
| 2 | Sincroniza el repo (Colab) o usa el local | `git` |
| 3 | Instala dependencias de cada pieza | `pip` |
| 4 | Drive/secretos y `.env` | — |
| 5 | Configuración del retrieval (Colab usa rutas de Drive; local arma un YAML propio) | — |
| 6 | Verifica recursos (guías, M2, eval set) | — |
| 7 | Pruebas unitarias | `unittest` |
| 8 | **Corpus** (ingesta + índice Chroma) | `M3/corpus/run_corpus.py` |
| 9 | **Retrieval** (calibra umbrales + delta vs. ingenuo) | `M3/retrieval/experimento_s08.py` |
| 10 | **Generación** de la respuesta final | `M3/generacion/run_generacion.py` |
| 11 | **RAGAS + cruce con M2** | `M3/ragas/run_ragas.py` |
| 12 | Persiste artefactos y arma un zip | — |

Parámetros de la corrida (celda 1): `RAMA`, `REPO_DIR`, `DATA_ROOT`, `EJEMPLOS`, `MODO_RAGAS`
(`real`/`mock`), `CORRER_CORPUS`, `CORRER_EXPERIMENTO`.

---

## 6. Artefactos que genera

| Ruta | Contenido |
|---|---|
| `M3/outputs/resultado_generacion.json` | Las 15 respuestas del RAG con su traza (`query_final`, `tool_invoked`, `tool_reason`, `contexts`, `answer`, `sources_used`, `fallback_used`, `ground_truth`) |
| `M3/retrieval/outputs/deltas_s08.csv` | Tabla del delta de cada técnica de retrieval vs. el RAG ingenuo |
| `M3/retrieval/outputs/umbrales.json` | Umbrales calibrados que usa la orquestación |
| `<DATA_ROOT>/resultados_pipeline/` y `resultados_M3.zip` | Copia de los resultados anteriores |

---

## 7. Correr una sola pieza (opcional)

Cada componente tiene su propio notebook en Colab:

- `M3/corpus/start_corpus.ipynb` — corpus e índice.
- `M3/retrieval/start_retrieval.ipynb` — retrieval y experimento.
- `M3/tools/start_tools.ipynb` — tool de normalización.
- `M3/ragas/start_ragas.ipynb` — RAGAS y cruce.

---

## 8. Notas

- **Umbrales:** los del corpus real se calibran con `experimento_s08.py`. Si las consultas reales
  no están etiquetadas, el experimento avisa y el pipeline sigue con los umbrales provisionales del
  YAML (no es un error fatal).
- **Normalización:** sin `BIOPORTAL_API_KEY` se usa el mapa simulado del equipo; con la key se
  consulta SNOMED CT real. La tool descarta etiquetas sin relación con la entidad.
- **RAGAS:** `MODO_RAGAS=real` usa Groq como juez (consume cuota); `mock` calcula local y sin API.
- **Cobertura:** el corpus tiene 3 guías clínicas (diabetes, falla cardíaca, depresión) y el eval
  set está alineado con ellas; las entidades fuera del corpus se responden con la válvula de escape
  (`fallback_used=true`), que es el comportamiento esperado.
