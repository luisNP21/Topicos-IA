# Scorecard -- M2 -- Evaluacion Clinical BERT (DisTEMIST)

**Ejemplos evaluados:** 10

## Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.571 | 0.333 | 0.421 |
| Similitud semantica | 1.000 | 0.583 | 0.737 |
| LLM-as-judge (score 1-5) | -- | -- | 3.975 |

## Mitigacion de sesgos del juez

- **Posicion:** delta medio = 0.550 (mitigado con promedio de score_normal y score_inverted)
- **Longitud:** el juez premio calidad en 5/5 pares (100.0%)
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)

## Distribucion de debilidades detectadas

| Debilidad | Documentos |
|---|---|
| caso mixto | 7 |
| boundary/formato -- el modelo entiende, falla el limite del span | 2 |
| funcionamiento correcto | 1 |

## Lectura del baseline

La debilidad mas frecuente es **caso mixto** (7/10 documentos). Comparando F1 exact-match (0.421) vs. F1 semantico (0.737), la brecha indica cuanto del error es de boundary/formato vs. comprension real de la entidad clinica.