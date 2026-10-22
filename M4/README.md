# M4 - Componente visual: notas clínicas escaneadas o fotografiadas → texto → entidades → guías

Este módulo agrega al sistema la capacidad de recibir una nota clínica como **imagen**
(escaneo o foto), transcribirla con un modelo de visión-lenguaje (VLM) y pasarla al pipeline
existente: chunking → encoder clínico → entidades de enfermedad → retrieval de fragmentos de
guía. Patrón de integración: **input**.

El README sigue los cuatro criterios de la rúbrica, una sección por criterio. Cada sección
indica quién la documenta.

> Convención: toda cifra que aparezca en este documento indica su fuente (archivo o run), y los
> hallazgos medidos se mantienen separados de los supuestos.

[COMPLETAR – Isabella: resumen de 4 a 6 líneas, escrito al final]

---

## 1. Componente visual implementado

*Responsables: Agustín (componente), Pau (punto de entrada), Isabella (integración con
harness y cache).*

### 1.1 Qué hace y dónde se conecta

*Flujo completo desde la imagen hasta las entidades y los fragmentos de guía, y en qué punto de
`run_inference/` se inserta.*

[COMPLETAR – Agustín]

### 1.2 Modelos y parámetros

*VLM (checkpoint y versión exactas), forma de acceso, baseline de OCR clásico, parámetros de
inferencia, latencia y costo por documento (medidos, con fuente).*

[COMPLETAR – Agustín]

### 1.3 Prompt de transcripción

*Texto literal y sus versiones.*

[COMPLETAR – Agustín]

### 1.4 Interfaz, configuración y estructura de archivos

*Interfaz `run(cfg, project_root) -> dict`, bloque en `config.yaml`, árbol de archivos nuevos o
modificados.*

[COMPLETAR – Agustín]

### 1.5 Integración con el cache de predicciones y el harness

*Cómo se registra cada sistema en `predictions/<sistema>.json` y cómo lo lee el harness.*

[COMPLETAR – Agustín e Isabella]

### 1.6 Punto de entrada del sistema

*App en Gradio o Streamlit que acepta imagen o texto y devuelve entidades más fragmentos de
guía: entradas aceptadas, salida mostrada, flujo interno, manejo de errores, generación final
con LLM externo, comando para ejecutarla y capturas.*

[COMPLETAR – Pau (app) y Agustín (generación con LLM externo)]

### 1.7 Cómo ejecutarlo

*Comandos exactos, dependencias con versiones, semillas y variables de entorno requeridas (sin
secretos).*

[COMPLETAR – Agustín]

---

## 2. Justificación del patrón de integración

*Responsable: Pau, con revisión de Isabella.*

### 2.1 Por qué input

*Argumento central, con evidencia del proyecto.*

[COMPLETAR – Pau]

### 2.2 Qué se habría perdido con tool y con output

| Patrón | Qué se habría perdido |
|---|---|
| Tool | [COMPLETAR – Pau] |
| Output | [COMPLETAR – Pau] |

### 2.3 Alternativa visual descartada y conexión con la teoría

*Por qué no CLIP sobre figuras de las guías, y cómo se aplican al diseño los temas vistos en
clase (parches de ViT y resolución, alucinación del VLM).*

[COMPLETAR – Pau]

---

## 3. Impacto medido en el harness

*Responsable de la sección: Isabella. Aportes: Luis (set de evaluación, métricas exactas y de
transcripción), Pau (métricas semánticas), Agustín (juez).*

### 3.1 Set de evaluación

*Cómo se generan las imágenes a partir de los documentos del test: niveles de degradación,
parámetros, semillas, script, ubicación y manifiesto. Aclarar que el gold sigue siendo el texto
original.*

[COMPLETAR – Luis]

### 3.2 Sistemas evaluados y cambios al harness

*Sistemas comparados (texto gold como techo, imagen → OCR → NER, imagen → VLM → NER), qué se
agregó al harness y cómo se corre.*

[COMPLETAR – Isabella]

### 3.3 Métricas de la transcripción

*CER y WER (normalización adoptada) y protocolo de auditoría de entidades inventadas por el
VLM.*

[COMPLETAR – Luis y Agustín]

### 3.4 Línea base de la última entrega

*Sistema de referencia, versión, valores por dimensión y archivo de resultados.*

[COMPLETAR – Isabella]

### 3.5 Resultados y delta

*Tabla por sistema, nivel de degradación y dimensión, con el delta respecto a la línea base.
Cada dueño interpreta su dimensión: exacto (Luis), semántico (Pau), juez (Agustín).*

| Sistema | Nivel | Exacto | Δ | Semántico | Δ | Juez | Δ | Fuente |
|---|---|---|---|---|---|---|---|---|
| [COMPLETAR] | | | | | | | | |

### 3.6 Hallazgos y supuestos

**Medido (con fuente):** [COMPLETAR – Isabella]

**Supuesto, no medido:** [COMPLETAR – Isabella]

### 3.7 Dónde el componente no aportó

*Respuesta directa y con cifras por sistema y nivel. Incluye la decisión sobre cualquier variante
que no aporte (se retira del sistema o se conserva, y por qué).*

[COMPLETAR – Isabella, con el equipo]

---

## 4. Casos de prueba

*Responsables: Pau (casos y entradas finales), Luis (formato y estructura), Isabella (análisis
de fallos).*

### 4.1 Entradas finales del sistema

*Documentos escaneados o fotografiados usados en las pruebas de M4: tipo, procedencia y cómo
se garantiza que no contienen datos de pacientes reales.*

[COMPLETAR – Pau]

### 4.2 Casos propios del dominio

*Casos clínicos propios (abreviaturas, entidades partidas entre líneas, tildes y ñ, dosis,
sellos o firmas, tablas de laboratorio, manuscrito real, entre otros). El fallo esperado se
documenta antes de ejecutar y el observado después.*

| ID | Descripción | Fallo esperado | Resultado observado | Evidencia |
|---|---|---|---|---|
| [COMPLETAR] | | | | |

### 4.3 Análisis de los casos donde el componente falla

*Etapa donde se rompe (lectura del VLM, chunking, NER, retrieval), evidencia y si el fallo es
corregible o inherente al enfoque.*

[COMPLETAR – Isabella, con aportes de cada integrante]
