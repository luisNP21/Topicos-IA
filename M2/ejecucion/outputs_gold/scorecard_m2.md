# Scorecard -- M2 -- Evaluacion Clinical BERT (DisTEMIST)

**Ejemplos evaluados:** 20

## Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.722 | 0.731 | 0.727 |
| Similitud semantica | 0.922 | 0.934 | 0.928 |
| LLM-as-judge (score 1-5) | -- | -- | 3.731 |

## Mitigacion de sesgos del juez

- **Posicion:** delta medio = 0.403 (mitigado con promedio de score_normal y score_inverted)
- **Longitud:** el juez premio calidad en 5/5 pares (100.0%)
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)

## Distribucion de debilidades detectadas

| Debilidad | Documentos |
|---|---|
| funcionamiento correcto | 11 |
| caso mixto | 9 |

## Lectura del baseline

La debilidad mas frecuente es **funcionamiento correcto** (11/20 documentos). Comparando F1 exact-match (0.727) vs. F1 semantico (0.928), la brecha indica cuanto del error es de boundary/formato vs. comprension real de la entidad clinica.