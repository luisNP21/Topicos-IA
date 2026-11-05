# Definición del proyecto integrador

---

## 1. Dominio

<!-- Ejemplo: Atención al ciudadano en una alcaldía municipal. -->

> Salud - Clínico

---

## 2. Usuario + decisión

<!--
Ejemplo:
- Usuario: un funcionario de la ventanilla de atención.
- Decisión que cambia: a qué dependencia enrutar un trámite y qué documentos
  pedirle al ciudadano en el momento, en vez de mandarlo a averiguar y volver.
> **Si la respuesta a "qué decide el usuario con esto" es "se informa", el
> proyecto no está definido. Un sistema útil cambia una decisión concreta de
> una persona concreta.**
-->

> ¿Quién es el usuario concreto?

Un médico o residente que está documentando un caso real (escribe una nota clínica de un paciente que tiene enfrente) y usa el sistema como asistente en ese momento.

> ¿Qué decisión concreta toma distinto gracias a tu sistema?

Qué sección de guía clínica o qué opción de tratamiento revisar primero para ese paciente, en vez de buscarla manualmente en el documento completo de la guía. El flujo pasa de "leo toda la guía de manejo de neumonía para encontrar la parte que aplica a este caso" a "el sistema ya me trae el fragmento relevante, ligado a lo que mencioné en la nota, y yo decido si lo sigo o no". 

---

## 3. Tarea del modelo (M1)

<!-- Ejemplo: clasificar el tipo de trámite a partir de la descripción libre
del ciudadano. -->

Como decisión del equipo, desarrollamos el M1 con tres modelos diferentes con las arquitecturas  continuación: 

**Arquitectura Encoder:**
* Roberta Base Biomedical: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
* Bert Base Spanish: dccuchile/bert-base-spanish-wwm-cased

**Arquitectura Encoder-Deocder:**
* mT5 small: google/mt5-small

> _¿Qué tarea de ML resuelven los modelos ajustados del Módulo 1?_

Con los modelos prouestos resolvemos la tarea de NER (Named Entity Recognition o Reconocimiento de Entidades Nombradas) de nombres de enfermedades trabajando con texto clínico en español. Este es un buen primer paso para nuestro objetivo final de diagnóstico de enfermedades y sugerencia de tratameinto; en etapas posteriores se podrá conectar esta primera tarea con RAG sobre códigos de enfermedades para cerrar el ciclo completo, teniendo la ventaja de que DisTEMIST ya incluye las etiquetas Snomed-CT correspondientes a todas las enfermedades nombradas con su respectiva relación semántica.

En el documento [Entrega_M1.pdf](https://github.com/luisNP21/Topicos-IA/blob/main/M1/Entrega_M1.pdf) se encuentra la explicación a detalle del desarrollo de este módulo junto con los resultados obtenidos.

---

## 4. Dataset + licencia

<!-- Ejemplo: 1.200 solicitudes históricas anonimizadas; licencia de uso
interno con permiso de la entidad. -->

> _¿Con qué datos entrenas/evalúas?_

Nuestro dataset es DisTEMIST, un corpus de 1,000 casos clínicos en español de distintas especialidades médicas, anotados con menciones de enfermedad. Para este módulo usamos los 750 casos del training set, dividiéndolos en splits train/dev/test. Conectamos los textos clínicos con las entidades a través del campo doc_id, luego hicimos chunking con solapamiento de los documentos para no sobrepasar la venta de contexto de cada modelo (BERT, clinical-BERT y mT5), evitando así perder entidades por truncamiento. Como cada uno fragmenta el texto de forma distinta, calculamos una ventana de chunking específica por tokenizer, en vez de un único umbral fijo para los tres modelos. Cada fragmento conserva un doc_id derivado del documento original (caso_XXX_chunk0, caso_XXX_chunk1, etc.), lo que nos permite reagrupar las predicciones por caso clínico completo al momento de evaluar, evitando así inflar las métricas al tratar fragmentos de un mismo paciente como documentos independientes. El split train/dev/test se hizo sobre los documentos originales antes del chunking, para prevenir fuga de datos entre fragmentos de un mismo caso. Usamos únicamente el subtrack de reconocimiento de entidades (subtrack1_entities), no el de normalización a códigos SNOMED, ya que en este módulo trabajamos solo con los nombres de enfermedad mencionados en el texto. 

> _¿De dónde salen y bajo qué licencia?_

El corpus proviene de SPACCC (casos clínicos en español publicados en SciELO) y está disponible bajo licencia Creative Commons Attribution 4.0 (CC BY 4.0), de acceso abierto y sin necesidad de firmar un acuerdo de uso de datos.

Cita: Miranda-Escalada, A., Eulàlia Farré, Luis Gasco, Salvador Lima& Martin Krallinger. (2022). DisTEMIST corpus: detection and normalization of disease mentions in spanish clinical cases (Version 5.1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.7614764

---

## 5. Métrica de éxito

<!-- Ejemplo: F1 macro > 0.80 en enrutamiento; y que el funcionario acepte la
sugerencia en ≥ 70% de los casos en la prueba con usuarios. -->
A continuación se encuentran los scorecard del modelo para el dataset de ejemplos gold y el de ejemplos adversariales.

### Scorecard  M2 - Evaluacion Clinical BERT (DisTEMIST)

**Ejemplos evaluados:** 59 documentos gold tomados de DisTEMIST

 Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.704 | 0.728 | 0.716 |
| Similitud semantica | 0.879 | 0.909 | 0.894 |
| LLM-as-judge (score 1-5) | -- | -- | 3.869 |
| Aciertos de dominio (si/no) | -- | -- | 0.593 |

> **Como leer la tabla.** Las columnas Precision/Recall solo aplican a las metricas de clasificacion (exact-match y similitud semantica), que cuentan aciertos y errores. Las dos ultimas filas no son de ese tipo y por eso se marcan con `--` (no aplica, **no es un dato faltante**): el **LLM-as-judge** es una escala 1-5 y su valor es el promedio de notas; **Aciertos de dominio** es una proporcion si/no y su valor es la tasa de cumplimiento del criterio (aciertos / total). Derivar precision/recall de esas filas exigiria un gold humano de "cumple / no cumple" que no existe.

**Ejemplos evaluados:** 10 documentos adversariales creados por el equipo

Metricas globales

| Dimension | Precision | Recall | F1 |
|---|---|---|---|
| Exact-match | 0.571 | 0.333 | 0.421 |
| Similitud semantica | 1.000 | 0.583 | 0.737 |
| LLM-as-judge (score 1-5) | -- | -- | 3.950 |
| Aciertos de dominio (si/no) | -- | -- | 0.300 |

> La nota de "Como leer la tabla" de arriba aplica igual a esta tabla: `--` significa "no aplica" para Precision/Recall en las dimensiones de juez (escala) y de dominio (tasa).

## Mitigacion de sesgos del juez

- **Posicion:** protocolo pairwise A/B con intercambio de orden (S06, Lab B). El test es diagnostico y no se promedia con el score final. Resultado: 0/59 empates en gold (juez estable con documentos reales) y **5/10 empates en adversariales**: ante textos clinicamente incoherentes el veredicto depende del orden, senal de que el juez no es confiable en ese tipo de casos.
- **Longitud:** el juez premio calidad en 6/6 pares (100.0%), con pares de respuesta corta correcta vs. larga parcialmente correcta.
- **Auto-preferencia:** documentado, sin test cross-family disponible (limitacion).
- **Aciertos de dominio (si/no):** dimension de S06; el juez lee el `criterio` de cada caso y decide si la prediccion lo cumple (>= 75% de recall y sin ruido grave). Gold 59.3% vs. adversariales 30.0%.

### Lectura del baseline

El F1 de exact-match es 0.716 en el gold set de 59 documentos y la similitud semantica sube a 0.894: la brecha de 0.178 puntos es error de **boundary/formato**, no de comprension de la entidad clinica; el modelo identifica la enfermedad, falla el limite del span. El juez pointwise lo confirma con 3.869/5 y un 59.3% de aciertos de dominio: en la mayoria de los casos la prediccion es util para el criterio del caso.

En los adversariales el comportamiento cambia de forma informativa: el exact-match cae a 0.421 (recall 0.333), la similitud semantica se mantiene alta en precision (1.000) pero el recall baja a 0.583, y los aciertos de dominio caen a 30.0%. Ademas aparece un sesgo de posicion del 50% que no existe en el gold set. Esto es esperable: los adversariales actuales son textos clinicamente imposibles (un fallecido con sintomas, un recien nacido de 85 anos) y su `esperado` incluye terminos que no son entidades ENFERMEDAD ("muertos", "mocos", "fiebre"), algo que un NER no hace por diseno. Es decir, la caida mide en parte una tarea mal planteada en el eval set, no solo una debilidad del modelo. Los adversariales utiles para NER (negaciones, siglas, abreviaturas de nota real, textos sin enfermedades) estan pendientes de correccion por el dueno del eval set.

**Nota metodologica sobre el cambio de juez.** El modelo juez es `qwen/qwen3.8-27b` (Groq). Se cambio desde `openai/gpt-oss-120b` porque este ultimo consume ~1000-1300 tokens internos de razonamiento por llamada y el cupo gratuito de Groq (200.000 tokens/dia) no alcanzaba para 59 documentos x 3 llamadas. Qwen3.8-27b resuelve la misma tarea con ~100 tokens por llamada. Las dos corridas (gold y adversariales) usan el mismo juez, por lo que el scorecard es internamente comparable.

---

## 6. Componente visual (M4)

<!-- Ejemplo: leer el documento escaneado que adjunta el ciudadano y verificar
que corresponde al trámite. -->

> _¿Qué aporta el componente multimodal/visual del Módulo 4 al sistema?_

Permite que la nota clínica entre como **imagen** (foto o escaneo), no solo como texto escrito. Patrón de integración **input**: un VLM (`Qwen/Qwen2-VL-2B-Instruct`) transcribe la imagen a texto y ese texto entra al mismo pipeline de M1/M2/M3 (NER → retrieval → generación), sin reescribir nada aguas abajo. El harness de M2 no se modifica: `run_inference` gana una fuente `vision` que reutiliza el módulo `M4/vision`.

**Lo implementado:** backends de transcripción (VLM y OCR clásico), válvula de escape cuando la imagen no es legible, puente al NER que produce el cache `{doc_id: [entidades]}` que el harness ya consume, un set de evaluación con imágenes sintéticas (15 casos, 28 niveles de degradación) y una app de chat que acepta imagen o texto.

**Lo medido** (15 casos, backend VLM; detalle en `M4/README_detallado.md`):

- **Transcripción:** CER 0.000 en imagen limpia y 0.003 en foto realista; 0 entidades inventadas.
- **NER en el harness:** imagen limpia y foto realista dan **exactamente el mismo resultado que el texto gold** (15 casos, 57 entidades: 29 aciertos, F1 exacto 0.652, semántico 0.697, juez 4.483; `M4/scorecard_m4.csv`): **delta cero** en las tres dimensiones.
- **Costo:** ~4.5 s por documento de transcripción con el VLM; es el único costo frente a leer el texto directo.

Conclusión: el componente habilita la entrada por imagen sin pérdida de calidad frente al texto en el uso realista. El límite de F1 (recall 0.509) es del NER de M1/M2 y afecta por igual a texto e imagen, no lo introduce la visión.

---

## 7. Riesgos éticos

<!-- Ejemplo: sesgo contra solicitudes mal redactadas; riesgo de negar un
trámite por un error del modelo. Mitigación: el sistema sugiere, el funcionario
decide. -->

> _¿Qué puede salir mal para una persona real? ¿Cómo lo mitigas?_

El sistema sugiere qué sección de guía revisar; no diagnostica ni decide. La decisión clínica es
siempre del médico.

| Riesgo | Qué puede salir mal | Mitigación | Evidencia |
|---|---|---|---|
| Recomendación tomada como indicación | El médico sigue un fragmento que no aplica a su paciente | La respuesta cita guía y sección para que el médico la verifique. Sin evidencia suficiente, el sistema se abstiene en vez de responder | Abstención en 9 de 15 casos (`M3/ragas/resultados/resultado_generacion.json`). En los 2 casos respondidos donde RAGAS pudo medir faithfulness, 0.80 y 0.84: la mayoría de las afirmaciones se apoya en la guía, no todas |
| Enfermedad omitida | El NER no detecta una enfermedad de la nota y no se consulta su guía | La app muestra las entidades detectadas, así el médico ve qué se consultó y qué no | Recall del NER 0.51 (29 de 57) en los 15 casos (`M4/scorecard_m4.csv`). Limitación declarada |
| Lectura errónea de una foto | La transcripción agrega una enfermedad que no está en la nota | Válvula de escape ante imágenes ilegibles; la app muestra el texto leído | CER 0.003 en foto realista y 0 entidades inventadas; 7 de 8 retos adversariales pasan. El que falla (`caso_13`, recorte severo) agrega texto que no está en la nota |
| Privacidad | Datos de un paciente real salen a un servicio externo | La visión y el NER corren en local. Solo la redacción de la respuesta usa Groq. La app avisa que no se ingresen datos reales | Aviso visible en la app (`M4/app/app.py`) |
| Falsa validez clínica | Se interpretan los resultados como validación médica | Se declara que las guías y los casos se crearon con apoyo de IA y que no hubo revisión de profesionales | Ver "Procedencia de los datos" |

### Procedencia de los datos

- **Guías clínicas (3):** resumidas y estructuradas con IA a partir de guías reales (fuentes en
  `M3/corpus/fuentes.yaml`). No sustituyen las guías originales.
- **Casos clínicos de evaluación (15) y sus respuestas esperadas:** redactados con apoyo de IA.
- **Imágenes de M4:** sintéticas, generadas por código a partir del texto de esos casos (fuentes
  tipográficas y degradaciones simuladas). No son notas ni fotos de pacientes reales; los sellos
  usan nombres de hospitales ficticios.
- **DisTEMIST (M1/M2):** casos clínicos reales publicados en SciELO, con licencia CC BY 4.0.

### Qué se evaluó y qué no

Se evaluó el **comportamiento del sistema**: si extrae las enfermedades, si recupera la sección
correcta, si se abstiene sin evidencia y si no inventa contenido. **No se evaluó la corrección
clínica** de las respuestas: no participaron profesionales de la salud, y las guías y casos se
crearon con apoyo de IA. Las conclusiones clínicas no son extrapolables.

---

