# M3 · Generación RAG + orquestación de tool use — Agustín

Pieza del Módulo 3 encargada a **Agustín**: la **llamada al LLM para la
generación final de la respuesta** y la **lógica explícita de cuándo se invoca
la tool de normalización y qué hace si falla**.

Sigue el material de la profesora (S07, S08, S10) y el patrón de M2
(`run(cfg, project_root) -> dict`, fallo explícito antes que silencioso).

---

## 1. Qué resuelve esta pieza

```
Entidad extraída (encoder de M1/M2)
        │
        ▼
[Orquestación — esta pieza]  resolver_query()
   ├─ retrieval con la entidad cruda      (pieza de Pau)
   ├─ si el score es bajo → normaliza     (tool de Luis)
   └─ decide query final + traza del porqué
        │
        ▼
[Generación — esta pieza]  generar_respuesta()
   └─ contexto + válvula de escape → RespuestaRAG
```

- **`resolver_query()`** cubre el criterio de la rúbrica *"cuándo el sistema
  invoca la tool y qué hace si falla"*.
- **`generar_respuesta()`** es el paso de *generation* del RAG, el que alimenta
  `faithfulness` y `answer relevancy` de RAGAS.

---

## 2. Contratos (interfaz con el resto del equipo)

Definidos en [`contratos.py`](contratos.py). Los de otras piezas se copian aquí
solo como referencia del acuerdo; **su dueño es quien los produce**.

| Contrato | Dueño | Nota |
|---|---|---|
| `Chunk`, `ManifestEntry` | Isa (corpus) | Se recomienda `Chunk.fuente` para poder citar (S07/S08 siempre la llevan) |
| `Fragment`, `RetrieveFn` | Pau (retrieval) | ⚠️ `Fragment.score` debe ser comparable con el umbral; declarar `score_tipo` |
| `NormalizationResult`, `NormalizeFn` | Luis (tool) | Fallo explícito vía `normalization_failed` |
| `RespuestaRAG`, `QueryResuelta`, `GenerateFn` | **Agustín** | Congelados por esta pieza |

---

## 3. Criterio de invocación de la tool (rúbrica: tool use)

`resolver_query(entidad, retrieve_fn, normalizar_fn, umbral, k)`:

1. Retrieval con la **entidad cruda** (la salida del encoder).
2. Si hay fragmentos y `score_top ≥ umbral` → **no se invoca la tool**.
3. Si el score es bajo o no hay resultados → **se invoca `normalizar_entidad`**:
   - **con match** → se reintenta el retrieval con la entidad normalizada;
   - **sin match** → se mantiene la entidad cruda y se registra el fallo.
4. Si el retrieval o la tool **lanzan una excepción**, se captura y se degrada de
   forma explícita (nunca se rompe el pipeline ni se falla en silencio).

La decisión es **determinista** (regla + umbral), no la decide el LLM. Esto da
**trazabilidad total** — se puede reproducir por qué se invocó la tool — y está
respaldado por la S10: *"no todo necesita un agente; el agente se justifica solo
si mejora el resultado"*. El encoder, además, no razona: solo clasifica tokens,
por eso la decisión vive en esta capa de orquestación.

---

## 4. Generación (rúbrica: RAGAS generation-side)

`generar_respuesta(entidad, fragments, generar_fn)`:

- Prompt aumentado de **S07**: instrucción → contexto con fuente → **válvula de
  escape** → pregunta. El `SYSTEM_RAG` es el literal del curso.
- **Fallback explícito**: si no hay fragmentos, **no se llama al LLM** y se
  devuelve la válvula de escape con `fallback_used=True`. Así el sistema no
  inventa cuando no tiene evidencia (lo que `faithfulness` penaliza).
- `sources_used` son los `chunk_id` que **se pasaron** como contexto
  (determinista y verificable), no lo que el LLM diga que citó.
- Salida: `RespuestaRAG = {answer, sources_used, fallback_used}`.

Modelo generador: **`qwen/qwen3.8-27b`** vía **Groq**, el mismo que el juez de
M2 en `main`, por coherencia y porque su cupo gratuito ya está validado. El
acceso reutiliza el patrón de `M2/harness/metrics_judge.py` (rotación de claves
`GROQ_API_KEY`, `_2`, … y espaciado entre llamadas).

> **Sesgo a documentar (auto-preferencia).** El generador y el juez de RAGAS
> comparten familia (Qwen). Si RAGAS se evalúa con el mismo modelo que genera,
> `faithfulness`/`answer relevancy` pueden inflarse. Es una limitación a
> reportar, igual que se hizo con el auto-preferencia en M2.

---

## 5. Estructura

```
M3/generacion/
├── contratos.py        # TypedDict + Protocol (interfaz del equipo)
├── generacion.py       # generar_respuesta() + GroqGenerator
├── orquestacion.py     # resolver_query() (criterio de tool + fallo)
├── mocks.py            # retrieval (Pau) y normalización (Luis) de juguete
├── run_generacion.py   # run(cfg, project_root) -> dict
├── config.yaml
├── entidades_ejemplo.json
├── requirements.txt
├── test_generacion.py  # pruebas con mocks, sin llamar a la API
└── README.md
```

---

## 6. Cómo correrlo

```bash
pip install -r requirements.txt
# .env con PROJECT_ROOT y GROQ_API_KEY
python run_generacion.py --config config.yaml
# pruebas (sin API):
python -m unittest test_generacion -v
```

Con `generacion.usar_mocks: true` corre de punta a punta **sin depender** de las
piezas de Pau ni de Luis (desarrollo en paralelo por contratos). Salida:
`M3/outputs/resultado_generacion.json` con la traza por entidad.

---

## 7. Estado y pendientes

- [x] Contratos + orquestación + generación + mocks + tests.
- [x] Corrida end-to-end con mocks.
- [ ] Conectar `retrieval.py` (Pau) y `normalizacion.py` (Luis) cuando estén.
- [ ] Exponer `contexts`/`answer` para que Luis los consuma en RAGAS.
- [ ] Corrida real sobre las entidades del encoder y cruce con el harness de M2.
