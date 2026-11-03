# Evaluación pareada M3 → M4

Herramientas pequeñas para que el delta de M4 compare **el mismo gold** con y sin imagen,
y para que cada caso adversarial tenga un criterio declarado y un resultado pasa/no pasa.
No cargan modelos: leen las salidas que ya producen `M2/run_inference`, `M2/harness` y `M4/vision`.

| Archivo | Para qué |
|---|---|
| `delta_pareado.py` | Escribe `M4/scorecard_m4.csv` (P/R/F1, Δ F1 y latencia por condición). Se niega si los golds de texto y visión no coinciden. |
| `verificar_gold.py` | `--literal`: entidades gold que no están literalmente en el texto del caso. `--comparar`: dos golds con los mismos doc_id y entidades. |
| `auditar_entidades.py` | Auditoría acierto / FP del NER / inventada con el **mismo** gold del harness. |
| `adversariales_m3.csv` + `evaluar_adversariales.py` | Retos sobre los 15 casos de M3 con criterio por caso y resultado pasa/no pasa. |
| `test_evaluacion_pareada.py` | Pruebas: `python M4/evaluacion_pareada/test_evaluacion_pareada.py`. |
| `resultados/` | Salidas vigentes: harness de texto, limpio y foto realista, cachés NER, sidecars de visión, adversariales y métricas de transcripción. |
| `resultados_revision/` | Salidas de la revisión docente con el gold anterior de 62 entidades (ver abajo). |
| `adversariales.csv` | Retos de la revisión sobre notas DisTEMIST (`ex_*`); se conservan como referencia. |

## Flujo

Todas las corridas usan el mismo gold: `M4/eval_set/eval_set_casos.jsonl` (15 casos, 57 entidades).

```bash
# 1. Gold: 0 entidades no literales
python M4/evaluacion_pareada/verificar_gold.py --literal M4/eval_set/eval_set_casos.jsonl

# 2. El mismo NER sobre el mismo gold, con texto y con imagen
#    (en Colab: notebooks de M4/ejecucion/harness/, que dejan un resultado_dimension1.json por condición)

# 3. Delta pareado (última celda de start_inference_harness_vision_realista.ipynb)
python M4/evaluacion_pareada/delta_pareado.py \
  --texto M4/evaluacion_pareada/resultados/resultado_dimension1_texto.json \
  --vision limpio=M4/evaluacion_pareada/resultados/resultado_dimension1_limpio.json \
           foto_realista=M4/evaluacion_pareada/resultados/resultado_dimension1_foto_realista.json \
  --meta limpio=M4/evaluacion_pareada/resultados/vision_meta_limpio.json \
         foto_realista=M4/evaluacion_pareada/resultados/vision_meta_foto_realista.json \
  --salida M4/scorecard_m4.csv

# 4. Auditoría de entidades con el mismo gold
python M4/evaluacion_pareada/auditar_entidades.py --gold M4/eval_set/eval_set_casos.jsonl \
  --ner limpio=M4/evaluacion_pareada/resultados/ner_limpio.json \
        foto_realista=M4/evaluacion_pareada/resultados/ner_foto_realista.json

# 5. Adversariales (defaults: adversariales_m3.csv y el gold de M4)
python M4/evaluacion_pareada/evaluar_adversariales.py \
  --transcripciones M4/evaluacion_pareada/resultados/transcripciones_vlm_sombra_severo.json \
                    M4/evaluacion_pareada/resultados/transcripciones_vlm_rotacion_recorte_severo.json \
  --salida M4/evaluacion_pareada/resultados/adversariales_resultado.csv
```

## Resultados vigentes

Fuente: `M4/scorecard_m4.csv` (15 casos, 57 entidades). Transcripción con Qwen2-VL-2B en T4.

| Condición | TP | FP | FN | Precision | Recall | F1 | Δ F1 vs. texto | Latencia (s/caso) |
|---|---|---|---|---|---|---|---|---|
| texto | 29 | 3 | 28 | 0.906 | 0.509 | 0.652 | — | — |
| limpio | 29 | 3 | 28 | 0.906 | 0.509 | 0.652 | 0.000 | 4.513 |
| foto_realista | 29 | 3 | 28 | 0.906 | 0.509 | 0.652 | 0.000 | 4.493 |

Auditoría de entidades (`resultados/resumen_metricas_transcripcion.md`): en texto, limpio y foto
realista, las 32 entidades predichas son 29 aciertos, 3 FP del NER y 0 inventadas.

Adversariales (`resultados/adversariales_resultado.csv`): 7 de 8 retos pasan. `caso_13` con
rotación y recorte severo no pasa: la lectura agrega «diarrea, pérdida de peso, fatiga», que no
está en la nota. Detalle y barrido de los niveles completos en `M4/README_detallado.md` (sección 4.2).

## `resultados_revision/`

Salidas de la revisión docente, generadas con Qwen2-VL-2B en Apple M4 Max (MPS fp16) y el mismo
Clinical BERT + LoRA. Usan el **gold anterior de 62 entidades**, que incluía 5 entidades no
literales en su nota (caso_03, caso_10, caso_11, caso_13 y caso_14). Por eso sus FN y su F1
(29/3/33, F1 0.617; manuscrito 0.568) no coinciden con los vigentes. La latencia (≈10 s por
página) es la de ese hardware. Se conservan como referencia; la tabla vigente es
`M4/scorecard_m4.csv`.

Sus adversariales (`resultados_revision/adversariales_resultado.csv`) son los de
`adversariales.csv` sobre notas DisTEMIST: 4 de 5 pasan; `ex_30_p2` con recorte agrega
«enfermedad pulmonar obstructiva crónica».
