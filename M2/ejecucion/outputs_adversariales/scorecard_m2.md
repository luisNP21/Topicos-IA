# Scorecard -- M2 -- Evaluacion Clinical BERT (DisTEMIST)

**Ejemplos evaluados:** 10

## Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.571 | 0.333 | 0.421 |
| Similitud semantica | 1.000 | 0.583 | 0.737 |
| LLM-as-judge (score 1-5) | -- | -- | 3.950 |
| Aciertos de dominio (si/no) | -- | -- | 0.300 |

> **Como leer la tabla.** Las columnas Precision/Recall solo aplican a las metricas de
> clasificacion (exact-match y similitud semantica), que cuentan aciertos y errores. Las dos
> ultimas filas no son de ese tipo y por eso se marcan con `--` (no aplica, no es un dato
> faltante): el **LLM-as-judge** es una escala 1-5 y su valor es el promedio de notas;
> **Aciertos de dominio** es una proporcion si/no y su valor es la tasa de cumplimiento del
> criterio (aciertos / total). Derivar precision/recall de esas filas exigiria un gold humano
> de "cumple / no cumple" que no existe.

## Mitigacion de sesgos del juez

- **Posicion (pairwise A/B):** empates = 5/10 (tasa = 0.500). veredicto robusto: solo se declara ganador si coincide en ambos ordenes; si no -> empate. El score final es pointwise y NO se promedia con este test.
- **Longitud:** el juez premio calidad en 6/6 pares (100.0%)
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)

## Distribucion de debilidades detectadas

| Debilidad | Documentos |
|---|---|
| caso mixto | 6 |
| boundary/formato -- el modelo entiende, falla el limite del span | 2 |
| fallo real -- ninguna metrica reconoce acierto | 1 |
| funcionamiento correcto | 1 |

## Lectura del baseline

La debilidad mas frecuente es **caso mixto** (6/10 documentos). Comparando F1 exact-match (0.421) vs. F1 semantico (0.737), la brecha indica cuanto del error es de boundary/formato vs. comprension real de la entidad clinica.