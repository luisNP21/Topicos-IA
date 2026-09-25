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

---

## 7. Riesgos éticos

<!-- Ejemplo: sesgo contra solicitudes mal redactadas; riesgo de negar un
trámite por un error del modelo. Mitigación: el sistema sugiere, el funcionario
decide. -->

> _¿Qué puede salir mal para una persona real? ¿Cómo lo mitigas?_

---

## 8. Compromisos del equipo

<!-- Ejemplo: reuniones los martes; repositorio compartido; cada integrante es
dueño de un módulo pero todos revisan. -->

> _¿Cómo se organizan? ¿Quién responde por qué? ¿Cómo se comunican?_
