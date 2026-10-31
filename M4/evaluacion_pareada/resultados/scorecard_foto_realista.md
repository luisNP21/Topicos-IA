# Scorecard -- M4 -- Evaluacion con componente visual (imagen -> OCR/VLM -> NER)

**Ejemplos evaluados:** 15

## Componente visual (M4)

Las entidades de este sistema se obtuvieron transcribiendo imágenes de notas
clínicas antes del NER (patrón *input*). Hiperparámetros de la corrida de visión:

| Hiperparámetro | Valor |
|---|---|
| Backend | vlm |
| Modelo | Qwen/Qwen2-VL-2B-Instruct |
| Casos | 15 |
| Transcripciones OK | 15 |
| Errores de backend | 0 |
| Válvulas (sin texto legible) | 0 |
| Latencia media (s/caso) | 4.493 |
| Latencia total (s) | 67.401 |

> La latencia y la tasa de válvulas/errores son el costo de operar el componente
> visual; van junto al delta de F1 para decidir si la variante aporta.

## Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.906 | 0.509 | 0.652 |
| Similitud semantica | 0.969 | 0.544 | 0.697 |
| LLM-as-judge (score 1-5) | -- | -- | 4.483 |
| Aciertos de dominio (si/no) | -- | -- | 0.400 |

> **Como leer la tabla.** Las columnas Precision/Recall solo aplican a las metricas de
> clasificacion (exact-match y similitud semantica), que cuentan aciertos y errores. Las dos
> ultimas filas no son de ese tipo y por eso se marcan con `--` (no aplica, no es un dato
> faltante): el **LLM-as-judge** es una escala 1-5 y su valor es el promedio de notas;
> **Aciertos de dominio** es una proporcion si/no y su valor es la tasa de cumplimiento del
> criterio (aciertos / total). Derivar precision/recall de esas filas exigiria un gold humano
> de "cumple / no cumple" que no existe.

## Mitigacion de sesgos del juez

- **Posicion (pairwise A/B):** empates = 6/13 (tasa = 0.462). veredicto robusto: solo se declara ganador si coincide en ambos ordenes; si no -> empate. El score final es pointwise y NO se promedia con este test.
- **Longitud:** el juez premio calidad en 6/6 pares (100.0%)
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion)

## Distribucion de debilidades detectadas

| Debilidad | Documentos |
|---|---|
| caso mixto | 8 |
| funcionamiento correcto | 7 |

## Lectura del baseline

La debilidad mas frecuente es **caso mixto** (8/15 documentos). Comparando F1 exact-match (0.652) vs. F1 semantico (0.697), la brecha indica cuanto del error es de boundary/formato vs. comprension real de la entidad clinica.