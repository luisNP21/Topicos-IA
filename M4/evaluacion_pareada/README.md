# Evaluación pareada M3 → M4 (sugerencia de la revisión M4)

Herramientas pequeñas para que el delta de M4 compare **el mismo gold** con y sin imagen,
y para que el caso adversarial tenga un criterio declarado y un resultado pasa/no pasa.
No cargan modelos: leen las salidas que ya producen `M2/run_inference`, `M2/harness` y `M4/vision`.

| Archivo | Para qué |
|---|---|
| `delta_pareado.py` | Escribe `scorecard_m4.csv` (P/R/F1, Δ F1 y latencia por condición). Se niega si los golds de texto y visión no coinciden. |
| `verificar_gold.py` | `--literal`: entidades gold que no están literalmente en el texto del caso. `--comparar`: dos golds con los mismos doc_id y entidades. |
| `auditar_entidades.py` | Auditoría acierto / FP del NER / inventada con el **mismo** gold del harness. |
| `adversariales.csv` + `evaluar_adversariales.py` | Casos de reto con criterio por caso y resultado pasa/no pasa. |
| `test_evaluacion_pareada.py` | Pruebas: `python M4/evaluacion_pareada/test_evaluacion_pareada.py`. |
| `resultados_revision/` | Salidas de la revisión con sus componentes reales (ver abajo). |

## Flujo sugerido

```bash
# 1. El mismo NER sobre el mismo gold, con texto y con imagen
#    (eval_set_path: M4/eval_set/eval_set_casos.jsonl en ambos configs)
python M2/run_inference/run_inference.py --config M2/run_inference/config.yaml          # fuente.tipo: texto
python M2/run_inference/run_inference.py --config M2/run_inference/config_vision.yaml   # fuente.tipo: vision
python M2/harness/run_harness.py --config ...   # deja un resultado_dimension1.json por condición

# 2. Delta pareado
python M4/evaluacion_pareada/delta_pareado.py --texto DIM1_TEXTO.json \
  --vision limpio=DIM1_LIMPIO.json foto_realista=DIM1_FOTO.json \
  --meta limpio=VISION_META_LIMPIO.json foto_realista=VISION_META_FOTO.json \
  --salida M4/scorecard_m4.csv

# 3. Gold y adversariales
python M4/evaluacion_pareada/verificar_gold.py --literal M4/eval_set/eval_set_casos.jsonl
python M4/evaluacion_pareada/evaluar_adversariales.py \
  --transcripciones M4/vision/predictions/transcripciones_vlm_rotacion_recorte_severo.json \
                    M4/vision/predictions/transcripciones_vlm_sombra_severo.json
```

## `resultados_revision/`

Generados en la revisión con **sus** componentes: Qwen2-VL-2B (`M4/vision`, Apple M4 Max, MPS fp16,
no T4 4-bit) y Clinical BERT + LoRA versionado en `M1/clinical_BERT/clinical_bert-distemist-lora`
(CPU), con el chunking 277/50 y `micro_prf1_by_doc` de `M2`. La latencia es la del VLM por página en
ese hardware (en su T4 rondaba 3,4 s). Sirven como ejemplo del formato y como punto de comparación
cuando lo corran en su entorno.

| Condición | TP | FP | FN | F1 | Δ F1 vs. texto |
|---|---|---|---|---|---|
| texto (mismos 15 casos) | 29 | 3 | 33 | 0.617 | — |
| limpio | 29 | 3 | 33 | 0.617 | 0.000 |
| foto_realista | 29 | 3 | 33 | 0.617 | 0.000 |
| sombra_severo | 29 | 3 | 33 | 0.617 | 0.000 |
| manuscrito_foto | 27 | 6 | 35 | 0.568 | −0.049 |

Adversariales (`adversariales_resultado.csv`): 4 de 5 pasan con la válvula ajustada de
`M4/vision/prompts.py`. `ex_30_p2` con recorte no pasa: la lectura añade
«enfermedad pulmonar obstructiva crónica» y «elevada concentración», que no están en la nota.
