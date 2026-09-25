# Scorecard -- M2 -- Evaluacion Clinical BERT (DisTEMIST)

**Ejemplos evaluados:** 59

## Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.704 | 0.728 | 0.716 |
| Similitud semantica | 0.879 | 0.909 | 0.894 |
| LLM-as-judge (score 1-5) | -- | -- | 3.869 |
| Aciertos de dominio (si/no) | -- | -- | 0.593 |

## Mitigacion de sesgos del juez

- **Posicion (pairwise A/B):** empates = 0/59 (tasa = 0.000). veredicto robusto: solo se declara ganador si coincide en ambos ordenes; si no -> empate. El score final es pointwise y NO se promedia con este test.
- **Longitud:** el juez premio calidad en 6/6 pares (100.0%)
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)

## Distribucion de debilidades detectadas

| Debilidad | Documentos |
|---|---|
| funcionamiento correcto | 31 |
| caso mixto | 28 |

## Lectura del baseline

La debilidad mas frecuente es **funcionamiento correcto** (31/59 documentos). Comparando F1 exact-match (0.716) vs. F1 semantico (0.894), la brecha indica cuanto del error es de boundary/formato vs. comprension real de la entidad clinica.