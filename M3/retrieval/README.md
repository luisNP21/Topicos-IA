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
| `construir_indice_mock.py` | Índice Chroma a partir del corpus de prueba (`data/mock/corpus_mock.json`) |
| `start_retrieval.ipynb` | Ejecución completa en Colab |
| `test_retrieval.py` | Pruebas unitarias con modelos simulados |
| `consultas_retrieval.jsonl` | Consultas para el corpus definitivo (pendientes de etiquetar) |

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

## Salidas del experimento

`deltas_s08.csv` (resumen por sistema), `detalle_por_consulta.csv`, `umbrales.json` (por pliegue y
finales), `delta_naive_vs_advanced.json` e `indice_info.json` (tamaño y huella del índice evaluado).
