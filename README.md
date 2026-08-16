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

> _¿Cómo mides que el sistema sirve? Métrica técnica y señal de valor real._

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
