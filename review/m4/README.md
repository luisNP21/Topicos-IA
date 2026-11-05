# Revisión M4: integración visual y evaluación · Asistente de guías clínicas

- **Repositorio:** `luisNP21/Topicos-IA`.
- **Commit revisado:** [`4faa880`](https://github.com/luisNP21/Topicos-IA/tree/4faa8809323f19ffa6a5ebdabbc3c667f523fd83) (rama `informes`, «Actualizar informe»). El código de la corrida reportada está en [`deb4b25`](https://github.com/luisNP21/Topicos-IA/tree/deb4b256881e677022de3d8740424200f6c6cc30) (`LuisM4`). Ambos están hoy en `dev`.
- La calificación se comunica por canal privado.
- Prototipo académico con datos sintéticos; no es un dispositivo clínico. Esta revisión no hace afirmaciones médicas.

## 1. Cómo se verificó

Niveles: **L** lectura del código y del README · **R** recálculo desde sus artefactos · **E** ejecución con sus componentes reales · **D** dobles o sustitutos.

| Qué | Cómo | Nivel | Resultado |
|---|---|---|---|
| Lector visual (`M4/vision`) | Su `run_vision.py`, `backends.py` y `prompts.py`, con Qwen2-VL-2B, sobre las 60 imágenes del set de 15 casos (regeneradas con su `generar_set.py`) y las 56 versionadas en `ejemplos_finales/`. Otro hardware: Apple M4 Max, MPS, fp16; solo `cuda → mps` y sin 4-bit | E | CER 0,0002 en limpio y 0,003 en foto realista, igual a su traza. 0 errores en 116 imágenes. ≈10 s por página en MPS (en su T4, ≈3,4 s) |
| NER (M1) sobre texto y sobre lo que lee el VLM | Clinical BERT + el LoRA versionado en `M1/clinical_BERT/clinical_bert-distemist-lora/`, en CPU, con su `cargar_modelo`, chunking 277/50 y `micro_prf1_by_doc` de M2, sobre los mismos 15 casos y 62 entidades | E | Texto plano 29/3/33 (TP/FP/FN), F1 0,617. Limpio, foto realista y sombra: idénticos. Manuscrito: F1 0,568 |
| Auditoría de entidades | Su `run_metricas.py` con esas predicciones | E | Reproduce su tabla exacta: 32 predichas = 7 aciertos + 25 «FP del NER» |
| Retos de recorte y sombra | Su VLM sobre `ejemplos_finales/retos/` | E | Un recorte completa una frase con un diagnóstico que no está en la imagen (M4-05) |
| Delta, scorecard y gold | Recálculo desde los JSON del harness guardados en los notebooks y desde `eval_set_casos.jsonl` | R | Ver M4-01, M4-03 y M4-04 |
| Integración imagen → M1 → M3 → respuesta | Lectura de `M4/app/` y `M4/vision/integracion.py`; `sistema_multimodal` con dobles | L, D | La lectura visual es la entrada del NER y define qué guía se consulta |
| Pruebas | `M4/vision/test_vision.py` en un entorno limpio | E | 31/31 en su rama |
| **No se re-ejecutaron** | Retrieval de M3 (el índice vive en Drive), la app completa, el LLM de Groq (el modelo no estaba disponible en la cuenta de la revisión) y la corrida en T4 4-bit | — | Pendientes; no se descuentan por eso |

No confirmamos que el LoRA versionado sea el mismo que el de Drive, aunque reproduce exactamente su TP 29 / FP 3.

## 2. Fortalezas

- **La justificación del patrón *input* es de las más sólidas del curso** ([README §2, L366–464](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L366-L464)): comparan input, tool y output con lo que se perdería en cada caso, el costo de la entrada y una alternativa descartada con argumento (CLIP y los parches ViT frente al texto fino).
- **El lector es fiel.** Lo comprobamos en otra máquina: CER 0,0002, y las 57 entidades legibles se transcriben literalmente.
- **Ya tienen un producto, no notebooks sueltos.** `M4/app/app.py` encadena foto → VLM → NER de M1 → recuperación con compuerta de M3 → respuesta con fuentes. Muestra el texto leído, los tiempos por etapa y se abstiene cuando las guías no cubren el caso.
- **Ingeniería cuidada:** backends intercambiables, contratos inyectables, `BackendError` sin respaldo silencioso, un manifiesto que impide mezclar niveles y 31 pruebas que pasan.
- **El generador de imágenes tiene control experimental:** semillas por documento, 28 degradaciones y la distinción entre «todo el gold» y «lo visible».
- **Honestidad:** declararon que el VLM real no se había medido en los retos, que las cifras vienen de una sola corrida en Colab y que los artefactos viven en Drive.

## 3. La noticia principal: la foto no le cuesta nada al sistema

Su README concluye que la imagen baja el F1 de 0,727 a 0,617 (Δ −0,110) y atribuye la caída al formato del VLM. Al correr el mismo NER sobre **los mismos 15 casos**, con texto y con imagen, el resultado es otro:

| Entrada (15 casos, 62 entidades) | TP | FP | FN | F1 | Δ F1 vs. texto |
|---|---|---|---|---|---|
| Texto plano | 29 | 3 | 33 | 0,617 | — |
| Imagen limpia | 29 | 3 | 33 | 0,617 | 0,000 |
| Foto realista | 29 | 3 | 33 | 0,617 | 0,000 |
| Sombra severa | 29 | 3 | 33 | 0,617 | 0,000 |
| Manuscrito fotografiado | 27 | 6 | 35 | 0,568 | −0,049 |

- La pérdida de recall ya está en el texto plano: es del NER sobre estos casos, no de la lectura.
- El 0,727 sale de otros 20 documentos de DisTEMIST, con otras 391 entidades (M4-01).
- Para un producto, «la foto rinde igual que el texto» es un mensaje más fuerte que el que presentaron.

## 4. Qué conviene corregir y cómo (rama de sugerencias, #CODE_PR)

- **[`SHA_C1`](https://github.com/manularrea/Topicos-IA/commit/SHA_C1) · Guardrail visible (M4-05):** `es_valvula` reconoce el eco «Se aprecia texto legible en la imagen.» como lectura vacía, y la app avisa, junto a la respuesta, que los hallazgos salen de una lectura automática y cuántas páginas no se leyeron. Con 2 pruebas nuevas.
- **[`SHA_C2`](https://github.com/manularrea/Topicos-IA/commit/SHA_C2) · Evaluación pareada (M4-01 a M4-04):** `M4/evaluacion_pareada/` con `delta_pareado.py` (se niega a calcular el delta si los gold no coinciden), `verificar_gold.py`, `auditar_entidades.py`, `adversariales.csv` con criterio por caso y `evaluar_adversariales.py`, más 12 pruebas y los resultados de la revisión.

No cambian prompts, modelos, parámetros de generación ni sus resultados históricos.

## 5. Observaciones

### M4-01 · El delta compara conjuntos distintos y se presenta como directo
- **Dónde:** [M4/README.md L25–33](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L25-L33), [L756](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L756), [L817](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L817) y [§4.3, L885](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L885); salidas de los tres notebooks del harness.
- **Qué se observa:**
  - El README dice que los tres gold «tienen el mismo contenido» y que el Δ «es una comparación directa».
  - El baseline usa `gold_examples_20.jsonl`: 20 documentos DisTEMIST y 391 entidades (F1 0,727). La visión usa los 15 casos de M3 con 62 entidades (F1 0,617).
  - Con el mismo gold, el delta real es 0,000 (§3). La hipótesis de §4.3 («el formato del VLM baja el recall») queda descartada.
- **Por qué importa:** S13 pregunta «¿mejoró lo nuevo? ¿no empeoró lo viejo?». Con conjuntos distintos, el −0,110 no responde ninguna de las dos, y lleva a buscar el problema en la visión cuando está en el NER.
- **Sugerencia:** C2. Correr el harness con `eval_set_path: M4/eval_set/eval_set_casos.jsonl` en las fuentes de texto y de visión, y escribir `M4/scorecard_m4.csv` con `delta_pareado.py`. Corregir L25–33, L756 y §4.3.
- **Criterio de cierre:** `scorecard_m4.csv` versionado, con filas texto / limpio / foto realista sobre los mismos 15 `doc_id`, latencia por condición y `delta_pareado.py` terminando sin error.

### M4-02 · El caso adversarial no tiene criterio ni resultado
- **Dónde:** [`ejemplos_finales/ejemplos.csv`](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/eval_set/ejemplos_finales/ejemplos.csv) (4faa880); [`M4/eval_set/eval_set_casos.jsonl`](https://github.com/luisNP21/Topicos-IA/blob/deb4b256881e677022de3d8740424200f6c6cc30/M4/eval_set/eval_set_casos.jsonl) (deb4b25); [README L906](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L906).
- **Qué se observa:**
  - Diseñaron 21 retos con un motivo que nombra el fallo esperado («texto oculto: recorte fuerte», «¿inventa o avisa?»). Ninguno tiene resultado: lo declaran en L906 («no se ha medido con el VLM real»).
  - El campo `criterio` está vacío en los 15 casos del gold.
  - El CSV versionado (`ex_45`, `ex_47`…, DisTEMIST) no es el set que se midió (los 15 casos de M3, cuyas imágenes viven en Drive).
- **Por qué importa:** en un asistente clínico, el comportamiento ante una imagen mala es lo que más importa. Al correr sus retos, apareció el caso de M4-05.
- **Sugerencia:** C2 (`adversariales.csv` con `tipo_criterio` y `criterio`, y `evaluar_adversariales.py` con resultado pasa / no pasa) y C1 para la válvula. Con C1, 4 de 5 retos pasan; `ex_30_p2` con recorte sigue sin pasar.
- **Criterio de cierre:** el CSV del eval set con una columna de criterio por caso, al menos un adversarial con resultado en el scorecard y el set medido versionado o regenerable con un comando.

### M4-03 · La auditoría de entidades usa otro gold
- **Dónde:** [README L712–713](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L712-L713); [`run_metricas.py` L78–96](https://github.com/luisNP21/Topicos-IA/blob/deb4b256881e677022de3d8740424200f6c6cc30/M4/metricas_transcripcion/run_metricas.py#L78-L96) (deb4b25).
- **Qué se observa:** con la fuente `m3`, la auditoría carga `M3/ejecucion/eval_set_casos.json`, que tiene una sola entidad por caso. El harness usa el gold de 62. Con sus predicciones reproducimos su tabla exacta (7 aciertos, 25 «FP del NER»); con el gold del harness, las mismas 32 predicciones son 29 aciertos, 3 FP y 0 inventadas.
- **Por qué importa:** el README dice a la vez «25 FP» y «FP 3» sobre las mismas 32 entidades.
- **Sugerencia:** C2 (`auditar_entidades.py` con el mismo gold que el harness) o apuntar `gold_set` del manifiesto a `M4/eval_set/eval_set_casos.jsonl`.
- **Criterio de cierre:** la tabla de origen de entidades suma lo mismo que el harness (29 aciertos, 3 FP).

### M4-04 · El gold de 62 entidades no dice de dónde sale
- **Dónde:** [`eval_set_casos.jsonl`, casos 03, 10, 11, 13 y 14](https://github.com/luisNP21/Topicos-IA/blob/deb4b256881e677022de3d8740424200f6c6cc30/M4/eval_set/eval_set_casos.jsonl) (añadido en `07ddb13`).
- **Qué se observa:** no se documenta quién anotó las 62 entidades ni con qué guía. Cinco no aparecen literalmente en el texto del caso: «diabetes mellitus tipo 2» (caso_03), «congestión pulmonar» (caso_10), «depresión en adultos» (caso_11 y caso_14) y «depresión» (caso_13).
- **Por qué importa:** el NER extrae tramos del texto; una entidad que no está en el texto es un falso negativo seguro y baja el recall de todas las condiciones por igual.
- **Sugerencia:** C2 (`verificar_gold.py --literal`). Corregir o justificar esas cinco y añadir dos líneas de procedencia del gold al README.
- **Criterio de cierre:** `verificar_gold.py --literal` sin hallazgos o con cada excepción justificada.

### M4-05 · Ante un recorte, el lector completa la frase
- **Dónde:** [`prompts.py` L28–45](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/vision/prompts.py#L28-L45); [`pipeline_app.py` L226](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/app/pipeline_app.py#L226); [`app.py` L172–174](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/app/app.py#L172-L174); [README L275](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L275).
- **Qué se observa** (su VLM, MPS fp16; en T4 4-bit puede variar):
  - En `retos/rotacion_recorte_severo/ex_30_p2.jpg`, la frase cortada «…en el seno de una [sepsis de origen] respiratorio» sale como «…en el seno de una **enfermedad pulmonar obstructiva crónica**», y «plaquetas = 51…» como «una **elevada concentración de plaquetas**». El NER extrae «enfermedad pulmonar obstructiva crónica» como hallazgo.
  - En `ex_22_p2.jpg` responde «Se aprecia texto legible en la imagen.». `es_valvula` no lo reconoce y la app lo pasaría como texto de la nota.
  - El texto leído se muestra plegado.
- **Por qué importa:** la entidad inventada llegaría a la recuperación, y la respuesta se apoyaría en una guía que no corresponde a la nota. Es el riesgo que su §2 identifica para el patrón *input*.
- **Sugerencia:** C1, y presentarlo en la demo como comportamiento de seguridad (PLAN_DEMO, P1). Si quieren ir más allá, avisar cuando el texto toca el borde de la imagen.
- **Criterio de cierre:** `ex_22_p2` termina como «sin texto legible»; con `ex_30_p2`, la app muestra el aviso de lectura automática y el texto leído.

### Límites que ustedes declararon (sugerencias, sin observación nueva)
- **Artefactos en Drive** ([L14](https://github.com/luisNP21/Topicos-IA/blob/4faa8809323f19ffa6a5ebdabbc3c667f523fd83/M4/README.md?plain=1#L14)): versionar al menos los `resultado_dimension1.json`, `resumen.md` y `scorecard_m4.csv`; son pequeños.
- **Imágenes ilegibles fuera del set** (L726, L855, L872): conviene incluir el peor caso, porque la válvula solo se prueba con él.
- **Manuscrito:** el NER extrae 3 entidades deformadas («mareomatismo», «artritis ml/min») de lecturas manuscritas. Vale como límite para la demo.
- **README de una página:** S13 pide una página; el actual tiene 909 líneas. Basta un resumen de una página arriba y el resto como anexo.
- **`ejemplos_finales/`** se borró de `main` después de la entrega (`fcc94ca`). Contiene los retos de `adversariales.csv`: conviene restaurarlo.

## 6. Seguimiento de M3

- **PR #23** (`revision/m3-evaluacion`): lo fusionaron. `M3/ragas/` en `4faa880` coincide con la propuesta.
- **PR #22** (`revision/m3`, borrador abierto): su sección «Antes de M4 y la Demo» pedía conservar casos textuales comparables y medir M4 con el mismo protocolo que M3, y M3-03 pedía no comparar promedios de documentos no pareados. Eso es justo M4-01; C2 lo resuelve. La orientación de la demo (separar lo sintético y no presentar la app como validación clínica) sí la siguieron.

## 7. Demo Day

La demo presenta el sistema completo como producto:
- la foto de la nota (M4);
- la extracción con Clinical BERT afinado (M1);
- la evaluación y el chunking (M2);
- la recuperación con compuerta y la respuesta con fuentes (M3).

[`PLAN_DEMO.md`](PLAN_DEMO.md) propone tres prioridades:

1. **[P0] El momento wow, estable en vivo.** Foto → respuesta con fuentes en segundos, con la versión etiquetada y un plan B.
2. **[P1] Confianza como característica.** Ante una foto recortada, el producto avisa y muestra lo que leyó (M4-05, M4-02; C1).
3. **[P2] Una prueba de valor con denominador.** «La foto rinde igual que el texto en 15 notas» (M4-01, M4-03, M4-04; C2).

Lo relativo a despliegue, URL, health check y eficiencia va como sugerencia para cuando desplieguen.

## 8. Cómo reproducir lo verificado

Desde la raíz del repositorio, en la rama de #CODE_PR. La rama parte de `4faa880`; el gold `M4/eval_set/eval_set_casos.jsonl` llegó por `LuisM4` y está en `dev`, así que el último comando necesita `dev` fusionado (la fusión es limpia y las pruebas pasan):

```bash
python M4/vision/test_vision.py                                # 33 pruebas (2 nuevas)
python M4/evaluacion_pareada/test_evaluacion_pareada.py        # 12 pruebas
python M4/evaluacion_pareada/delta_pareado.py \
  --texto M4/evaluacion_pareada/resultados_revision/resultado_dimension1_texto.json \
  --vision limpio=M4/evaluacion_pareada/resultados_revision/resultado_dimension1_limpio.json \
           foto_realista=M4/evaluacion_pareada/resultados_revision/resultado_dimension1_foto_realista.json \
  --salida scorecard_prueba.csv
python M4/evaluacion_pareada/verificar_gold.py --literal M4/eval_set/eval_set_casos.jsonl
```

- **Lector:** `M4/vision/run_vision.py` con el backend `vlm`. En otro dispositivo, cambiar `cuda` por el que corresponda.
- **NER pareado:** `M2/run_inference` con `eval_set_path: M4/eval_set/eval_set_casos.jsonl` en las fuentes `texto` y `vision`, y después `M2/harness`.
- **No se re-ejecutaron:** el retrieval de M3, la app completa, el LLM y la corrida en T4.
