# M2 — Evaluacion del modelo de reconocimiento de entidades clinicas

Sistema que evalua el modelo Clinical BERT + LoRA (entrenado en M1) sobre el gold set de M2
en tres dimensiones independientes — coincidencia exacta de spans, similitud semantica y un
juez basado en LLM — y las combina en un scorecard con diagnostico de debilidad. Existe en
dos formas equivalentes: **cuatro notebooks** (exploratorios, uno por dimension mas el
harness integrado) y **un script de linea de comandos** (`run_harness.py`, produccion,
reproducible con un solo comando). Ambas formas leen y escriben en la misma carpeta del
proyecto.

## Las dos formas de correr esto

| | Notebooks | Script (`run_harness.py`) |
|---|---|---|
| Entorno | Google Colab, monta Drive | Cualquier maquina con Python, sin Colab obligatorio|
| Configuracion | variables hardcodeadas al inicio de cada notebook | `.env` + `config.yaml`, versionados |
| Ejecucion | celda por celda, manual | `python run_harness.py --config config.yaml` |
| Uso previsto | revisar/depurar una dimension en detalle, iterar | correr todo de una sola vez, reproducible, para entregar |

No son fuentes independientes de verdad — el script reimplementa la misma logica que ya se
valido en los notebooks, refactorizada en modulos importables (`common.py`,
`metrics_exact.py`, `metrics_semantic.py`, `metrics_judge.py`, `scorecard.py`) para poder
correr sin abrir Colab. Los notebooks siguen siendo la referencia para entender y depurar
cada dimension; el script es lo que se corre para generar el entregable final.

---

## Parte 1 — Los notebooks


| Notebook | Que mide | Depende de |
|---|---|---|
| `dimension1_exact_match.ipynb` | ¿La prediccion coincide caracter a caracter con el gold? | Nada (punto de partida) |
| `dimension1b_similitud_semantica.ipynb` | ¿La prediccion es semanticamente equivalente al gold, aunque el span no coincida exacto? | Salida anterior (reusa las mismas predicciones) |
| `04_llm_judge.ipynb` | ¿Un LLM juez califica la respuesta como buena, con controles de sesgo? | Nada (corre su propia inferencia) |
| `harness.ipynb` | Junta las tres dimensiones anteriores en un scorecard unico con diagnostico de debilidad | Salidas de los tres notebooks anteriores |

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
3. `04_llm_judge.ipynb` → corre su propia inferencia, genera `llm_judge_scorecard.csv` y `llm_judge_bias_summary.json`

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
3. En `04_llm_judge.ipynb` y en `harness.ipynb`, pegar la `GROQ_API_KEY` en la celda
   correspondiente, o guardarla en Colab Secrets con ese mismo nombre.
4. Correr todo de arriba a abajo. `harness.ipynb` es el mas largo (77 celdas, reproduce las
   tres dimensiones): puede tardar varios minutos, sobre todo en la Seccion 7 (llamadas al
   LLM juez).
5. Los resultados quedan en `M2/outputs/`, listos para citar en el informe.

---

## Parte 2 — El script (`run_harness.py`)

Version en modulos Python de la misma logica, pensada para correr con un solo comando fuera
de Colab, con configuracion versionada en vez de variables hardcodeadas.

```bash
python run_harness.py --config config.yaml
```

### Instalacion

```bash
pip install -r requirements.txt
```

| Paquete | Version | Para que |
|---|---|---|
| `transformers`, `accelerate`, `torchao` | 5.16.1, 1.14.0, 0.16.0 | Cargar Clinical BERT |
| `peft` | 0.20.0 | Adaptador LoRA |
| `datasets` | 4.0.0 | (no se ve uso directo en los modulos revisados) |
| `sentence-transformers` | 5.7.0 | Embeddings de la Dimension 1b |
| `scipy` | 1.16.3 | Asignacion hungara (`linear_sum_assignment`) |
| `numpy`, `pandas` | 2.1.3, 2.2.3 | Numerico / tabular |
| `PyYAML` | 6.0.3 | Leer `config.yaml` |
| `python-dotenv` | 1.2.3 | Leer `.env` |

`environment.asegurar_paquetes()` es un respaldo de auto-instalacion, usar `pip install -r requirements.txt` como mecanismo
principal, tal como dice el propio docstring del modulo.

### Configuracion

**`.env`** (no versionado, uno por maquina):
GROQ_API_KEY=gsk_...


**`config.yaml`** (versionado, compartido por el equipo):

| Clave | Valor actual | Notas |
|---|---|---|
| `proyecto.seed_global` | 42 | misma semilla en los 4 notebooks y el script |
| `rutas.gold_set` | `M2/eval_harness/gold_examples_20.jsonl` | **distinto al `gold_examples.jsonl` de los notebooks** |
| `rutas.model_dir` | `M1/saved_models/clinical_bert-distemist-lora` | requiere `adapter_config.json` + `adapter_model.safetensors` |
| `modelo.base_checkpoint` | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | |
| `chunking.window_words` / `overlap_words` | 277 / 50 |  |
| `dimension1b_similitud_semantica.modelo_embeddings` | `paraphrase-multilingual-MiniLM-L12-v2` | |
| `dimension1b...calibracion_umbral` | 4000 pares negativos, seed 42 | |
| `dimension3_llm_judge.judge_model` | `openai/gpt-oss-120b` (via Groq) | temperatura 0.0 |
| `dimension3_llm_judge.pares_longitud_path` | `length_bias_pairs.json` | 5 pares: 2 tipo A, 2 tipo B, 1 de FP puros |
| `scorecard.umbrales_debilidad` | f1_bajo=0.2, f1_alto=0.7, score_juez_bajo=2.5, score_juez_alto=3.5 | define las 4 categorias de `diagnosticar_debilidad` |

### Uso

```bash
# Las 4 piezas completas
python run_harness.py --config config.yaml

# Una dimension a la vez (debugging, sin re-correr todo)
python run_harness.py --config config.yaml --solo exact
python run_harness.py --config config.yaml --solo semantica   # requiere que 'exact' ya haya corrido
python run_harness.py --config config.yaml --solo judge
python run_harness.py --config config.yaml --solo scorecard   # requiere que las 3 anteriores hayan corrido
```

`verificar_prerequisitos()` corre antes que nada y falla rapido (gold set, adaptador LoRA,
`GROQ_API_KEY`) en vez de tronar a mitad de la dimension 3 despues de haber gastado tiempo
en las dos primeras.

### Ejecucion del script desde Google Colab

Para ejecutar el script en Colab, cargar en Google Colab alguno de los notebooks
`ejecucion/start_gold.ipynb` o `ejecucion/start_adversarial.ipynb`, segun el gold set que se
quiera evaluar, y correr sus celdas en orden. El notebook monta Google Drive, instala los
requerimientos, configura `PROJECT_ROOT` y `GROQ_API_KEY`, y ejecuta:

```bash
python run_harness.py --config config.yaml
```

Antes de ejecutarlo, verificar que el adaptador LoRA y el gold set correspondiente existan
en las rutas configuradas en `config.yaml`, y crear el secreto `GROQ_API_KEY` en Colab.
Las instrucciones detalladas estan en [`ejecucion/README.md`](ejecucion/README.md).

### Salidas del script

Todo en `{PROJECT_ROOT}/M2/outputs/`:

| Archivo | Generado por |
|---|---|
| `resultado_dimension1.json` | `metrics_exact.py` |
| `resultado_dimension1b.json` | `metrics_semantic.py` |
| `resultado_dimension3.json` | `metrics_judge.py` |
| `scorecard_m2.json` | `scorecard.py` |
| `scorecard_m2.md` | `scorecard.py` |

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
- **LLM-as-judge:** un modelo de lenguaje (via Groq) califica la respuesta completa del
  sistema contra una rubrica, con verificacion explicita de que el score no cambia solo por
  el orden en que se presentan las respuestas (sesgo de posicion) ni por su longitud
  (sesgo de longitud, con pares controlados en `length_bias_pairs.json`), y con el riesgo
  de auto-preferencia documentado (no testeable directamente si el juez comparte familia
  con el modelo evaluado).
- **Scorecard integrador:** cruza las tres dimensiones por documento y aplica una regla de
  diagnostico (`diagnosticar_debilidad`) que interpreta el patron: si el exact-match es
  bajo pero la similitud semantica es alta, la debilidad es de boundary, no de comprension;
  si las tres son bajas, la debilidad es real; si exact-match y semantica son altos pero el
  juez objeta, la debilidad es de calidad clinica, no de extraccion.

Las tres dimensiones son complementarias, no intercambiables: ninguna sola cuenta la
historia completa de donde falla el sistema. El scorecard final es el que permite leerlas
juntas.

## Reproducibilidad

- Todos los notebooks fijan `SEED = 42` (`random`, `numpy`, `torch`) al inicio; el script
  hace lo mismo via `common.fijar_seeds()`, llamado desde `run_harness.py` y de nuevo
  dentro de `environment.preparar_entorno()` (que solo corre como parte de la dimension
  exact-match — correr `--solo semantica` o `--solo judge` aislado no reimprime versiones
  de librerias, pero la seed ya quedo fijada antes por `run_harness.py`).
- Las rutas de entrada/salida son las mismas dentro de cada sistema (notebooks entre si,
  script consigo mismo) pero **no coinciden exactamente entre notebooks y script**.
