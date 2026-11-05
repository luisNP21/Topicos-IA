# M3 · Retrieval avanzado y orquestación de la normalización

Recupera fragmentos de guías clínicas para una entidad extraída (por ejemplo, "diabetes mellitus tipo 2")
y decide cuándo normalizar esa entidad con SNOMED CT / UMLS antes de buscar.

## Archivos

| Archivo | Contenido |
|---|---|
| `config_retrieval.yaml` | Rutas, modelos y parámetros del retrieval, la intención, la orquestación y la calibración |
| `config_retrieval.py` | Lectura del YAML y construcción del índice y de los parámetros de orquestación |
| `retrieval.py` | Técnicas de búsqueda sobre el índice Chroma y las funciones `retrieve_naive` y `retrieve_advanced` |
| `../generacion/orquestacion.py` | Criterio de invocación de la normalización y compuerta de evidencia (`resolver_query`) |
| `experimento_s08.py` | Experimento controlado, calibración de umbrales con validación cruzada y tablas de resultados |
| `gold_set.py` | Carga del gold set, validación contra el índice cargado y resumen |
| `construir_indice_mock.py` | Índice Chroma del corpus de prueba (`M3/retrieval/data/mock/corpus_mock.json`), con el código de la ingesta (`M3/corpus`) |
| `start_retrieval.ipynb` | Ejecución completa en Colab |
| `test_retrieval.py` | Pruebas unitarias con modelos simulados |
| `data/mock/` | Corpus de prueba, sus consultas y su mapa de normalización simulado |
| `data/real/` | Gold set del corpus real (`gold_consultas.jsonl`, `gold_anotaciones.csv`, `gold_meta.json`) y mapa de normalización simulado (`normalizacion_simulada.json`) |

## Gold set del corpus real

El corpus real (chunks JSON, manifiesto) vive en Drive y no se versiona. El gold set sí: 40 consultas
(22 dentro del corpus, 18 fuera) con las guías y los chunks relevantes, y la anotación de cada uno de
los 64 chunks. **Lo anotó una IA simulando a un anotador humano, sin revisión humana**; así consta en
`gold_meta.json` y en cada consulta (`anotador`).

Como los `chunk_id` son posicionales, una nueva ingesta puede cambiarlos. `gold_meta.json` guarda la
descripción del corpus anotado (número de chunks, chunks por guía y huella de contenido, calculada como
en `retrieval.info_indice`). Antes de medir, el notebook y `experimento_s08.py` comprueban que todos los
`chunk_id` existan en el índice, que cada uno pertenezca a una guía relevante de su consulta, que ninguna
consulta dentro del corpus quede sin chunks, que las anotaciones cubran el índice y que la huella
coincida. Si algo falla, se detienen indicando qué comparación falló. Si el corpus cambió, hay que volver
a la versión anotada o anotar de nuevo y actualizar `gold_meta.json`.

## Índice

El índice Chroma no se copia de Drive: se reconstruye en el disco local (`chroma_dir`) a partir de los
chunks JSON del perfil, con el mismo código de la ingesta (`M3/corpus/embeddings.py` y
`corpus_store.py`, que no dependen de Docling). Si el índice local ya existe y coincide con los chunks,
se reutiliza. Siempre se verifica que número, ids y textos del índice coincidan con los chunks JSON; si
no, se detiene con un mensaje que dice qué difiere. Copiar el índice ya construido no es seguro: la
ingesta actualiza la colección sin borrar ids anteriores y el índice puede provenir de otra versión de
Chroma.

## Técnicas

La base es el laboratorio S08: búsqueda densa (`intfloat/multilingual-e5-base`), BM25, fusión RRF con
k = 60 y reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. A eso se suma query transformation:
como toda la guía menciona la enfermedad, la entidad sola no permite distinguir la sección de
tratamiento. Las consultas se reformulan con intención ("tratamiento de {entidad}", entre otras) y el
reranker recibe la pregunta "¿Cuál es el tratamiento de {entidad}?". Opcionalmente, el reranker ordena
solo chunks clasificados como secciones de tratamiento.

## Criterio de la normalización

El score del primer fragmento (probabilidad del reranker) se compara con dos umbrales:

- por encima de `umbral`, la búsqueda es confiable y no se normaliza;
- entre `umbral_evidencia` y `umbral`, se normaliza la entidad y se conserva el mejor intento;
- por debajo de `umbral_evidencia`, no se entregan fragmentos y la generación responde que no tiene
  la información.

Las siglas se normalizan siempre (`forzar_por_sigla=True`), porque ante siglas el reranker asigna scores
altos a guías que no corresponden. Los umbrales se calibran con validación cruzada estratificada: el de
normalización con el índice de Youden y el de evidencia justo por encima del score más alto obtenido por
una consulta fuera del corpus.

## Configuración y uso

Todo se configura en `config_retrieval.yaml`. La clave `corpus` elige el conjunto de rutas (`mock` para
el corpus de prueba, `real` para el definitivo). Si `orquestacion.umbral` es `null`, se usan los umbrales
calibrados por el experimento en `<salida>/umbrales.json`.

```bash
python M3/retrieval/construir_indice_mock.py          # solo para el corpus de prueba
python M3/retrieval/experimento_s08.py                # usa config_retrieval.yaml
python M3/retrieval/experimento_s08.py --corpus real
```

Integración con el pipeline:

```python
from config_retrieval import cargar_config, configurar_pipeline, parametros_orquestacion
from retrieval import pregunta_intencion, retrieve_advanced

cfg = cargar_config("M3/retrieval/config_retrieval.yaml")
configurar_pipeline(cfg)
q = resolver_query(entidad, retrieve_fn=retrieve_advanced, normalizar_fn=normalizar_entidad,
                   **parametros_orquestacion(cfg))
respuesta = generar_respuesta(q["query_final"], q["fragments"], generar_fn=generar_fn,
                              pregunta=pregunta_intencion(q["query_final"]))
```

Alternativamente, basta con definir `M3_RETRIEVAL_CONFIG` con la ruta del YAML: `retrieve_advanced`
construye el índice con esa configuración la primera vez que se llama.

## Integración con las otras partes

| Parte | Cómo se conecta |
|---|---|
| Corpus (`M3/corpus`) | El perfil `real` lee de Drive los chunks JSON y el manifiesto que deja la ingesta (`corpus_json`, `manifest`) y reconstruye el índice en `/content/chroma_guias`. Del manifiesto solo se usan las guías indexadas (`solo_guias_indexadas`). Antes de construir el índice se verifica que `modelos.embeddings` coincida con `embeddings.model_name` de `M3/corpus/config.yaml`. |
| Herramienta de normalización (`M3/tools`) | Ambos perfiles usan hoy un mapa simulado (`mapa_normalizacion`). El del corpus real (`data/real/normalizacion_simulada.json`) cubre las siglas y los sinónimos del gold set con el vocabulario de las guías, y algunas siglas fuera del corpus; no incluye erratas ni términos ambiguos. Es una simulación: con `mapa_normalizacion: null` se usa `tool_normalizacion.normalizar_entidad` con la ontología de `M3/tools/config.yaml` (requiere `BIOPORTAL_API_KEY`). |
| Generación (`M3/generacion`) | `run_generacion.py` carga este YAML, llama a `configurar_pipeline` y toma los umbrales de `parametros_orquestacion`. |

Las rutas relativas del YAML se resuelven desde la raíz del repositorio, no desde el directorio de
trabajo, así que funcionan igual desde el notebook, desde un script o desde la generación. Si todavía
no hay umbrales calibrados en `<salida>/umbrales.json`, se usan `orquestacion.umbrales_provisionales`
(calibrados sobre el corpus de prueba) y se muestra un aviso.

## Salidas del experimento

`deltas_s08.csv` (resumen por sistema), `detalle_por_consulta.csv`, `umbrales.json` (por pliegue y
finales), `delta_naive_vs_advanced.json` e `indice_info.json` (tamaño y huella del índice evaluado).
