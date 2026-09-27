# M2 — Evaluacion del modelo de reconocimiento de entidades clinicas

Pipeline que evalua las predicciones NER del modelo Clinical BERT + LoRA (entrenado en M1)
contra el gold set de M2 con tres dimensiones: exact-match, similitud semantica y un juez
LLM que tambien emite un veredicto binario de dominio. Las dimensiones se combinan en un
scorecard de debilidades. La inferencia y la evaluacion se ejecutan por separado: primero
`run_inference.py` genera un cache por documento y despues el harness evalua ese cache.

## Formas de ejecucion

| | Launcher de Colab | Scripts Python |
|---|---|---|
| Entrada | `ejecucion/start_inference_harness.ipynb` | `run_inference.py` y `run_harness.py` |
| Entorno | Colab, monta Drive e instala ambos requirements | Python local o Colab con los mismos requirements |
| Configuracion | monta el Drive y escribe `.env` en el repo clonado | `.env` + YAML de cada fase |
| Ejecucion | corre inferencia y luego harness en celdas separadas | dos comandos, inferencia primero |
| Uso previsto | pipeline completo sobre los archivos de Drive | ejecucion reproducible o depuracion con `--solo` |

`harness.ipynb` se conserva como base conceptual y notebook exploratorio; no es el launcher
del pipeline modular y no se modifica al actualizar los scripts. `start_inference_harness.ipynb`
es el punto de entrada de Colab para ejecutar los scripts de `run_inference/` y `harness/`.

---

## Parte 1 — Los notebooks


| Notebook | Que mide | Depende de |
|---|---|---|
| `dimension1_exact_match.ipynb` | ¿La prediccion coincide caracter a caracter con el gold? | Nada (punto de partida) |
| `dimension1b_similitud_semantica.ipynb` | ¿La prediccion es semanticamente equivalente al gold, aunque el span no coincida exacto? | Salida anterior (reusa las mismas predicciones) |
| `llm_judge.ipynb` | ¿Un LLM juez califica la respuesta como buena, con controles de sesgo? | Nada (corre su propia inferencia) |
| `harness.ipynb` | Junta las tres dimensiones anteriores en un scorecard unico con diagnostico de debilidad | Salidas de los tres notebooks anteriores |
| `ejecucion/start_inference_harness.ipynb` | Ejecuta el pipeline modular de inferencia y evaluacion | Drive, los dos YAML y Secret `GROQ_API_KEY` |

**Nota importante:** `harness.ipynb` no es un cuarto analisis independiente — reimplementa
dentro de un solo notebook la logica completa de los tres notebooks individuales (exact-match,
similitud semantica y LLM-judge) y al final los combina. Sirve como version "todo en uno"
reproducible con una sola corrida, semilla fija y las mismas funciones que ya se validaron
por separado. Los notebooks individuales siguen siendo la referencia de cada dimension; el
harness es el que arma el scorecard final que se entrega.

### Orden de ejecucion

**Opcion A — correr cada dimension por separado (para revisar/depurar cada una):**

1. `dimension1_exact_match.ipynb` → genera `M2/outputs/resultado_dimension1.json`
2. `dimension1b_similitud_semantica.ipynb` → lee el paso 1, genera `resultado_dimension1b_similitud.json`
3. `llm_judge.ipynb` → corre su propia inferencia, genera `llm_judge_scorecard.csv` y `llm_judge_bias_summary.json`

**Opcion B — correr todo de una vez con el harness integrado:**

1. `harness.ipynb`, de arriba a abajo. Genera internamente `resultado_dimension1.json`,
   `resultado_dimension1b.json` y `resultado_dimension3.json` en `M2/outputs/`, y al final
   arma `scorecard_m2.json` y `scorecard_m2.md`.

Ambas opciones escriben en la misma carpeta de salida, asi que se pueden combinar: por
ejemplo, correr los notebooks individuales para revisar el detalle de cada dimension, y
despues correr solo la Seccion 8 del harness (el scorecard) apuntando a esos mismos
archivos, sin tener que re-ejecutar todo.

### Que necesita cada uno

| Notebook | Entrada | Salida | Requiere |
|---|---|---|---|
| `dimension1_exact_match` | `M1/saved_models/clinical_bert-distemist-lora/` + `M2/eval_harness/gold_examples.jsonl` | `resultado_dimension1.json` | GPU (T4 alcanza) |
| `dimension1b_similitud_semantica` | `resultado_dimension1.json` + `gold_examples.jsonl` | `resultado_dimension1b_similitud.json` | `pip install sentence-transformers`, CPU alcanza |
| `04_llm_judge` | mismo modelo + `gold_examples.jsonl` | `llm_judge_scorecard.csv`, `llm_judge_bias_summary.json` | GPU + `GROQ_API_KEY` (gratis en console.groq.com/keys) |
| `harness` | mismo modelo + `gold_examples.jsonl` | `resultado_dimension1.json`, `resultado_dimension1b.json`, `resultado_dimension3.json`, `scorecard_m2.json`, `scorecard_m2.md` | GPU + `sentence-transformers` + `GROQ_API_KEY` (necesita todo lo anterior junto) |

### Como correr

1. Abrir el notebook en Colab, correr la primera celda para montar Drive.
2. Verificar en la celda de rutas que `MODEL_DIR` y `GOLD_SET_PATH` existan (`.exists()` en
   `True`) antes de seguir. Si `MODEL_DIR` no existe, el modelo LoRA no se subio a Drive
   todavia o la ruta cambio.
3. Para la evaluación con LLM juez (`04_llm_judge.ipynb` y `harness.ipynb`), se requiere una API key de Groq (gratuita en console.groq.com/keys). En Google Colab, se debe agregar en la ventana de secretos / keys (icono de llave  en la barra lateral izquierda):
   - **Nombre:** `GROQ_API_KEY`
   - **Valor:** tu clave `gsk_...`
   - Activar la casilla **Acceso de Notebook** (*Notebook access*).
4. Correr todo de arriba a abajo. `harness.ipynb` es el mas largo (77 celdas, reproduce las
   tres dimensiones): puede tardar varios minutos, sobre todo en la Seccion 7 (llamadas al
   LLM juez).
5. Los resultados quedan en `M2/outputs/`, listos para citar en el informe.

---

## Parte 2 — Pipeline modular

El pipeline separa la generacion de predicciones de su evaluacion. `run_inference.py`
carga el modelo NER y guarda el cache; `run_harness.py` conserva el parser de argumentos,
carga el `eval_set` y crea `sistema` desde ese cache. La funcion `harness` recibe esos datos
y ejecuta exact-match, similitud semantica y LLM-as-judge. El harness no carga el modelo NER.

### Launcher de Colab

El notebook [`ejecucion/start_inference_harness.ipynb`](ejecucion/start_inference_harness.ipynb)
organiza la corrida en estas fases:

1. Clona la rama `refactor-harness` en `/content/corpus` y cambia al directorio del repo.
2. Instala `M2/run_inference/requirements.txt` y `M2/harness/requirements.txt`.
3. Monta Drive, define `PROJECT_ROOT`, revisa el adaptador y el gold set, y escribe
   `/content/corpus/.env` con `PROJECT_ROOT` y `GROQ_API_KEY` desde Colab Secrets.
4. Entra a `M2/run_inference/` y ejecuta `python run_inference.py`; las predicciones quedan
   en el `output_path` del YAML de inferencia.
5. Entra a `M2/harness/` y ejecuta el CLI con `--config config.yaml`.

En un runtime limpio, la celda `git pull` debe ejecutarse desde `/content/corpus`; si el
notebook aún está en `/content`, primero cambia al directorio clonado.

### Configuracion YAML

`M2/run_inference/config.yaml` controla exclusivamente la inferencia:

| Clave | Valor actual | Uso |
|---|---|---|
| `sistema` | `encoder_solo` | Identificador de esta corrida |
| `modelo.checkpoint` | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | Encoder base |
| `modelo.adapter_path` | `M1/saved_models/clinical_bert-distemist-lora` | Ruta relativa a `PROJECT_ROOT` |
| `chunking.window_words` / `overlap_words` | 277 / 50 | Ventana y solapamiento de inferencia |
| `eval_set_path` | `M2/eval_harness/gold_examples_20.jsonl` | Gold set usado para predecir |
| `normalizacion.activa` | `false` | Variante encoder sin normalizacion |
| `output_path` | `M2/predictions/encoder_solo.json` | Cache `{doc_id: [entidades]}` bajo `PROJECT_ROOT` |

`M2/harness/config.yaml` controla la evaluacion y las salidas:

| Clave | Valor actual | Uso |
|---|---|---|
| `rutas.gold_set` | `M2/eval_harness/gold_examples_20.jsonl` | Debe ser el mismo gold usado por inferencia |
| `rutas.outputs_dir` | `M2/outputs` | Directorio de resultados |
| `rutas.inference_config` | `../run_inference/config.yaml` | De ahi se lee el `output_path` del cache |
| `archivos_salida` | `resultado_dimension1.json`, `resultado_dimension1b.json`, `resultado_dimension3.json`, scorecards | Artefactos de las dimensiones y scorecard |
| `dimension1b_similitud_semantica` | embeddings `paraphrase-multilingual-MiniLM-L12-v2`, 4000 negativos, seed 42 | Matching semantico |
| `dimension3_llm_judge` | `qwen/qwen3.8-27b`, Groq, temperatura 0, max 600 tokens, pausa 3 s | Juez y controles de sesgo |
| `scorecard.umbrales_debilidad` | F1 bajo/alto 0.2/0.7; juez bajo/alto 2.5/3.5 | Diagnostico por documento |

Ambos YAML resuelven sus rutas de datos desde `PROJECT_ROOT`. El `.env` del runtime contiene
`PROJECT_ROOT` y `GROQ_API_KEY`; no se debe versionar la clave.

### Ejecucion desde terminal

Desde la raiz del repo, instala ambas listas de dependencias:

```bash
pip install -r M2/run_inference/requirements.txt
pip install -r M2/harness/requirements.txt
```

Luego ejecuta las dos fases en orden:

```bash
cd M2/run_inference
python run_inference.py
cd ../harness
python run_harness.py --config config.yaml
```

El comando sin `--solo` ejecuta las tres dimensiones y genera el scorecard. Para depurar,
`--solo semantica` ejecuta exact-match como prerequisito y luego semantica; `--solo judge`
ejecuta exact-match y juez. `--solo scorecard` solo integra resultados ya existentes.
Por ello, para preparar los tres artefactos antes de scorecard con comandos separados, corre
`--solo semantica` y `--solo judge` antes de `--solo scorecard`.

El notebook actual termina con `--solo judge` seguido de `--solo scorecard`, pero no incluye
un paso `--solo semantica`. Para generar el scorecard completo en esa corrida, ejecuta antes
la dimension semantica o usa el comando sin `--solo`.

Las salidas se guardan en `{PROJECT_ROOT}/M2/outputs/`: `resultado_dimension1.json`,
`resultado_dimension1b.json`, `resultado_dimension3.json`, `scorecard_m2.json` y
`scorecard_m2.md`. La dimension juez requiere `GROQ_API_KEY` en Colab Secrets. Consulta
[`ejecucion/README.md`](ejecucion/README.md) para los detalles de acceso a Drive y Secret.

---

## Que mide cada dimension (aplica a ambas formas de correrlo)

- **Exact-match:** compara el span predicho contra el gold caracter a caracter. Estricto:
  no reconoce boundary parcial ni sinonimia. Es el criterio que ya se uso para seleccionar
  el mejor checkpoint durante el fine-tuning en M1. En el script, ademas exporta
  `casos_boundary` (pares donde una entidad es substring de la otra), que la dimension
  semantica usa despues para calibrar su umbral.
- **Similitud semantica:** compara los mismos pares con similitud coseno de embeddings,
  resuelve la asignacion con el algoritmo hungaro (1:1, con penalizacion fuerte para pares
  bajo umbral) y calibra el umbral por indice de Youden sobre los `casos_boundary`
  (positivos) contra pares cruzados entre documentos distintos (negativos). Reconoce
  boundary, variantes de formato y, en menor medida, sinonimia. Clasifica cada acierto
  nuevo en categorias (exacto, puntuacion/formato, boundary, solapamiento parcial,
  sinonimia) y mide ambiguedad entre entidades gold del mismo documento.
- **LLM-as-judge:** un modelo de lenguaje (via Groq) califica la prediccion completa del
  sistema contra una rubrica 1-5 anclada y **lee el campo `criterio` del gold set** para
  emitir un veredicto binario de dominio (`cumple_criterio`: si/no), tal como pide S06
  (Dimensión 3). El sesgo de posicion se mide con el **protocolo pairwise A/B de S06 (Lab B)**:
  se comparan dos respuestas anonimas A/B y se intercambian de orden; solo se declara ganador
  si el veredicto coincide en ambos ordenes, si no -> empate. El test es **diagnostico y no
  se promedia con el score final** (el score del juez es el pointwise). El sesgo de longitud
  se mide con pares controlados de respuesta corta correcta vs. respuesta larga
  **parcialmente correcta** (no ruido sin relacion con el gold). La auto-preferencia queda
  documentada (no testeable directamente si el juez comparte familia con el modelo evaluado).

## Correccion tras el feedback (LLM-as-judge)

La entrega anterior se aparto de lo visto en S06 en tres puntos; esta version los corrige:

1. **Sesgo de posicion.** Antes se intercambiaban `gold` y `pred` dentro de la rubrica
   pointwise y se promediaban ambos scores. Eso medía la asimetria de la rubrica, no el orden,
   y contaminaba el score final. Ahora se usa el protocolo pairwise A/B de S06 (`comparar_robusto`)
   y el score final es el pointwise, sin promedio.
2. **Sesgo de longitud.** Antes la respuesta "incorrecta larga" era ruido sin relacion con el
   gold. Ahora es una respuesta **larga parcialmente correcta** (algunos aciertos + FP).
3. **Dimensión de dominio.** Antes el campo `criterio` no se leia y no existia la dimension
   si/no. Ahora el juez lo lee y emite `cumple_criterio`, que alimenta la dimension de
   "aciertos de dominio" del harness (S06).

> Los resultados versionados en `M2/ejecucion/outputs_*` provienen de la corrida anterior;
> deben **regenerarse** corriendo el harness para reflejar la metodologia corregida.

> **Pendiente de otra dimension (no LLM-as-judge):** los ejemplos adversariales actuales
> tienen un `esperado` con terminos que no son entidades ENFERMEDAD (`mocos`, `muerto`,
> `fiebre`) y prueban algo que un NER no hace por diseno. Segun el feedback, los
> adversariales utiles aqui son negaciones, siglas, abreviaturas de nota real y textos sin
> enfermedades. Esa correccion corresponde al dueno del eval set, no a la dimension del
> juez.
- **Scorecard integrador:** cruza las tres dimensiones por documento y aplica una regla de
  diagnostico (`diagnosticar_debilidad`) que interpreta el patron: si el exact-match es
  bajo pero la similitud semantica es alta, la debilidad es de boundary, no de comprension;
  si las tres son bajas, la debilidad es real; si exact-match y semantica son altos pero el
  juez objeta, la debilidad es de calidad clinica, no de extraccion.

Las tres dimensiones son complementarias, no intercambiables: ninguna sola cuenta la
historia completa de donde falla el sistema. El scorecard final es el que permite leerlas
juntas.

## Reproducibilidad

- El CLI fija la semilla configurada en `harness/config.yaml` mediante
   `common.fijar_seeds()` antes de calcular las dimensiones.
- El launcher modular y los scripts usan `gold_examples_20.jsonl`; la ruta del cache se
   comparte mediante `output_path` en `run_inference/config.yaml`. El notebook exploratorio
   `harness.ipynb` conserva su propio flujo y sus rutas originales.
