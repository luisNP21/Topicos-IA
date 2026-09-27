# M3 · Generación RAG + orquestación de tool use — Agustín

Pieza del Módulo 3 encargada a **Agustín**: la **llamada al LLM para la
generación final de la respuesta** y la **lógica de cuándo se invoca la tool de
normalización y qué hace si falla**.

Sigue el material de la profesora (S07, S08, S10) y el patrón de M2
(`run(cfg, project_root) -> dict`, fallo explícito antes que silencioso).

---

## 1. Qué resuelve esta pieza

```
Entidad extraída (encoder de M1/M2)
        │
        ▼
[Orquestación] resolver_query()            ← esta carpeta
   ├─ retrieval con la entidad cruda      (pieza de Pau: M3/retrieval)
   ├─ si el score es bajo → normaliza     (tool de Luis: M3/tools)
   └─ decide query final + traza del porqué
        │
        ▼
[Generación] generar_respuesta()           ← esta carpeta
   └─ contexto + válvula de escape → RespuestaRAG
        │
        ▼
M3/outputs/resultado_generacion.json  → lo consume RAGAS (M3/ragas, Luis)
```

- **`resolver_query()`** cubre el criterio de la rúbrica *"cuándo el sistema
  invoca la tool y qué hace si falla"*.
- **`generar_respuesta()`** es el paso de *generation* del RAG; su salida
  (`contexts`, `answer`, `ground_truth`) alimenta `faithfulness`,
  `answer_relevancy` y el cruce con el harness de M2.

---

## 2. Integración con las piezas del equipo

`run_generacion.py` corre de dos formas:

| Modo | Cómo | Usa |
|---|---|---|
| **Real** (por defecto, `usar_mocks: false`) | `python run_generacion.py --config config.yaml` | `M3/retrieval/retrieval.py` (Pau) + `M3/tools/tool_normalizacion.py` (Luis) |
| **Mocks** (`usar_mocks: true` o `--mocks`) | `python run_generacion.py --config config.yaml --mocks` | Mocks propios; corre sin `chromadb` ni `corpus_utils` |

Detalles de la integración:

- Las **rutas y parámetros de cada pieza viven en su propio YAML** (como pidió
  Isabella). Este módulo los **referencia** en `config.yaml` (`modulos:`), no los
  duplica: lee `M3/retrieval/config_retrieval.yaml` y `M3/tools/config.yaml`.
- De `config_retrieval.yaml` toma `umbral`, `umbral_evidencia`, `forzar_por_sigla`
  y `k` (vía `parametros_orquestacion`), y **construye el índice** con
  `configurar_pipeline`.
- La **pregunta de generación es la misma que usa el reranker**
  (`retrieval.pregunta_intencion(entidad)`), para que el LLM responda sobre lo
  que se buscó (la sección de tratamiento), no sobre la enfermedad en general.
- `normalizar_entidad` de Luis recibe la ontología del YAML
  (`tool_normalizacion.ontologia`) mediante un adaptador.

---

## 3. Criterio de invocación de la tool (rúbrica: tool use)

`resolver_query(entidad, retrieve_fn, normalizar_fn, umbral, k, umbral_evidencia, forzar_por_sigla)`

El score del primer fragmento (probabilidad del reranker, en `[0, 1]`) se divide
en **tres zonas**:

| Zona | Acción |
|---|---|
| `score >= umbral` | Confiable: **no se invoca** la tool. |
| `umbral_evidencia <= score < umbral` | Dudoso: **se normaliza** y se vuelve a buscar; se conserva el intento con mejor score. |
| `score < umbral_evidencia` | **Sin evidencia**: no se entregan fragmentos y la generación responde que no tiene la información. |

- **Siglas** (`forzar_por_sigla`): se normalizan aunque el score sea alto,
  porque ante siglas el reranker da scores altos a guías que no corresponden.
- Guarda de seguridad: si el score es `rrf` (no comparable con un umbral fijo)
  se lanza un error explícito.
- La decisión es **determinista** (regla + umbral), no la decide el LLM: da
  trazabilidad total y está respaldada por la S10 *("no todo necesita un agente")*.
  El encoder, además, no razona; por eso la decisión vive en esta capa.
- Si el retrieval o la tool **lanzan una excepción**, se captura y se degrada de
  forma explícita (nunca en silencio).

Los umbrales los **calibra Pau** con `experimento_s08.py`.

---

## 4. Generación (rúbrica: RAGAS generation-side)

`generar_respuesta(entidad, fragments, generar_fn, pregunta)`

- Prompt aumentado de **S07**: instrucción → contexto con fuente → **válvula de
  escape** → pregunta. El `SYSTEM_RAG` es el literal del curso.
- **Fallback explícito**: sin fragmentos **no se llama al LLM** y se devuelve la
  válvula de escape con `fallback_used=True`. Si la respuesta es la válvula de
  escape, `sources_used` queda **vacío** (ningún fragmento la respalda).
- `sources_used` = `chunk_id` en que se apoya la respuesta; `contexts` = **todo**
  lo recuperado (lo que RAGAS mide en context precision/recall).

Modelo generador: **`qwen/qwen3.8-27b`** vía **Groq**, el mismo del juez de M2 en
`main`. El acceso reutiliza el patrón de `M2/harness/metrics_judge.py` (rotación
de claves `GROQ_API_KEY`, `_2`, … y espaciado entre llamadas).

> **Sesgo a documentar (auto-preferencia).** El generador y el juez de RAGAS
> comparten familia (Qwen). Si RAGAS se evalúa con el mismo modelo que genera,
> `faithfulness`/`answer_relevancy` pueden inflarse. Es una limitación a
> reportar, igual que en M2.

---

## 5. Estructura

```
M3/generacion/
├── contratos.py        # TypedDict + Protocol (interfaz del equipo)
├── generacion.py       # generar_respuesta() + GroqGenerator + contextos_para_ragas()
├── orquestacion.py     # resolver_query() (3 zonas, siglas, compuerta de evidencia)
├── mocks.py            # retrieval (Pau) y normalización (Luis) de juguete
├── run_generacion.py   # run(cfg, project_root) -> dict; modos real y mock
├── config.yaml         # referencia los YAML de cada pieza
├── entidades_ejemplo.json
├── requirements.txt
├── test_generacion.py  # 15 pruebas con mocks, sin llamar a la API
└── README.md
```

---

## 6. Cómo correrlo

```bash
pip install -r requirements.txt
# .env con PROJECT_ROOT, GROQ_API_KEY (y BIOPORTAL_API_KEY para la tool de Luis)
python run_generacion.py --config config.yaml            # piezas reales
python run_generacion.py --config config.yaml --mocks    # sin dependencias
python -m unittest test_generacion -v                    # pruebas (sin API)
```

Salida: `M3/outputs/resultado_generacion.json`, con la traza por entidad
(`query_final`, `tool_invoked`, `tool_reason`, `contexts`, `answer`,
`sources_used`, `fallback_used`, `ground_truth`).

---

## 7. Estado y pendientes de coordinación

**Hecho**
- [x] Contratos, orquestación (3 zonas), generación, mocks y tests (15/15).
- [x] Integración cableada a `M3/retrieval` (Pau) y `M3/tools` (Luis), con sus YAML.
- [x] Salida consumible por RAGAS (`contexts`, `answer`, `ground_truth`).

**Pendiente (equipo)**
- [ ] `corpus_utils.py` no existe en ninguna rama todavía; `M3/retrieval` lo
      importa para construir el índice. La corrida **real** espera ese detalle del
      corpus/refactor (Isabella).
- [ ] Instalar dependencias del retrieval (`chromadb`, `rank_bm25`) para la
      corrida real.
- [ ] **RAGAS (Luis)** quedó con `llm_model: "openai/gpt-oss-120b"`; el juez de M2
      es `qwen/qwen3.8-27b`. Conviene unificar (gpt-oss no cabe en el cupo gratuito).
- [ ] `M3/tools/orquestacion.py` (borrador de Luis) duplica este
      `M3/generacion/orquestacion.py`, que Paula documenta como el canónico.
      Conviene eliminar el borrador.
- [ ] `ground_truth` para `context_recall`: hoy es `null`; hay que decidir la
      referencia (respuesta esperada del eval set o el pasaje relevante).
