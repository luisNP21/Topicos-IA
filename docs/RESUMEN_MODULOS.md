# Resumen de los módulos — Sistema clínico NER + RAG

> Rama `main`, commit `2958935`.
> Explicación corta de cada módulo, con los términos técnicos entre paréntesis.

## Flujo general

```text
Nota clínica (texto o imagen)
   → detectar enfermedades (NER)
   → buscar en guías (retrieval)
   → redactar una respuesta (generación)
   → evaluar la calidad (harness y RAGAS)
```

---

## M1 — Detector de enfermedades

**En una frase:** convierte una nota clínica en una lista de enfermedades.

**Qué hace:** lee el texto y marca los fragmentos que son nombres de enfermedad. Eso se llama **reconocimiento de entidades nombradas (NER)**.

**Cómo:** usa casos clínicos anotados de DisTEMIST, parte los textos largos en pedazos (**chunks**) y entrena tres modelos con **ajuste fino liviano (LoRA)**: BERT español, RoBERTa clínico y mT5. Los encoders predicen una etiqueta por palabra (**BIO**); mT5 genera la lista como texto.

**Resultado:** el mejor fue **RoBERTa clínico + LoRA**, con **F1 0.708**. BERT quedó en 0.610 y mT5 en 0.119.

**Ojo:** el notebook versionado usa 30 documentos (24/3/3), no 750. El test de 3 documentos hace que el F1 sea poco estable.

---

## M2 — Evaluador del detector

**En una frase:** califica las enfermedades detectadas por M1 con tres exámenes distintos.

**Qué hace:** compara lo que predijo el NER contra las enfermedades correctas (**gold**).

**Cómo:**
- **Coincidencia exacta (exact-match):** ¿el texto coincide letra por letra?
- **Parecido de significado (similitud semántica):** usa **vectores (embeddings)** para aceptar variantes.
- **Juez automático (LLM-as-judge):** un modelo de lenguaje pone nota de 1 a 5 y decide si la predicción es útil.

Luego un **tablero (scorecard)** junta los tres resultados y diagnostica si el fallo es de límite del fragmento (**boundary**), de calidad clínica o real.

**Resultado:** con 59 documentos, F1 exacto **0.716**, F1 semántico **0.894** y juez **3.869/5**.

**Ojo:** el juez es `qwen/qwen3.8-27b` vía Groq. El harness no carga el NER: evalúa un cache de predicciones.

---

## M3 — Buscador y generador de respuestas

**En una frase:** toma las enfermedades de M1 y busca en las guías qué tratamiento les corresponde.

**Qué hace:** construye una biblioteca de guías, busca los fragmentos relevantes y redacta una respuesta con fuentes.

**Cómo:**
- **Corpus:** lee PDFs con **Docling**, los parte en **chunks** y guarda sus **embeddings** en una base vectorial (**Chroma**).
- **Búsqueda avanzada:** combina búsqueda por significado (**densa**), búsqueda por palabras (**BM25**) y unión de rankings (**RRF**). Luego reordena con un **reranker** y reescribe la consulta para pedir el tratamiento (**query transformation**).
- **Tool:** si la entidad es una sigla o el puntaje es bajo, la normaliza con **SNOMED CT** vía **BioPortal**.
- **Compuerta:** si no hay evidencia suficiente, no responde.
- **Generación:** un **LLM** (`qwen/qwen3.8-27b` vía Groq) redacta solo con los fragmentos recuperados. Si no hay contexto, usa una **válvula de escape**.
- **Evaluación:** **RAGAS** mide fidelidad, relevancia y cobertura del contexto.

**Resultado:** el buscador ingenuo ponía el tratamiento primero en **4.5%** de los casos; el avanzado llegó a **68.2%**, y con la regla de siglas a **77.3%**. En generación, **10/15** casos terminaron en válvula de escape porque el corpus de 3 guías no cubría la recomendación esperada.

**Ojo:** el corpus es pequeño (3 guías, 64 chunks), las guías son simuladas con IA y el gold de retrieval fue anotado por IA sin revisión humana.

---

## M4 — Entrada por imagen y app

**En una frase:** permite que la nota clínica entre como foto o escaneo y la convierte en texto para el mismo sistema.

**Qué hace:** lee la imagen, la transcribe y pasa el texto al NER de M1/M2 y al RAG de M3. No reescribe el pipeline: solo cambia la entrada.

**Cómo:**
- **Visión:** usa un **modelo que ve y escribe (VLM)**, Qwen2-VL, o un **lector clásico de texto en imágenes (OCR)**, Docling/Tesseract.
- **Prompt:** pide transcribir literalmente, sin interpretar ni inventar. Si no hay texto legible, devuelve una **válvula**.
- **Evaluación:** genera imágenes sintéticas (15 casos, 28 niveles, 420 imágenes) y mide el **error de lectura (CER/WER)**.
- **Harness:** el texto transcrito entra al mismo NER y al mismo evaluador de M2.
- **App:** una interfaz web (**Gradio**) recibe texto o imágenes, muestra la enfermedad principal, la respuesta, las fuentes y los detalles técnicos.

**Resultado:** en imagen limpia y foto realista el **CER fue 0.000 y 0.003**, sin entidades inventadas. El harness dio **exactamente el mismo resultado que con texto gold**: F1 exacto **0.617**, semántico **0.660** y juez **4.417**. La visión no mejora ni empeora el NER; solo habilita la entrada por imagen.

**Ojo:** las imágenes son sintéticas y la letra a mano sale de fuentes tipográficas; no reemplazan notas reales.

---

## Tabla resumen

| Módulo | En simple | Resultado principal |
|---|---|---|
| **M1** | Detecta enfermedades en una nota. | F1 **0.708** con RoBERTa clínico + LoRA |
| **M2** | Califica esas detecciones con tres exámenes. | F1 exacto **0.716**; semántico **0.894**; juez **3.869/5** |
| **M3** | Busca el tratamiento en las guías y redacta. | Tratamiento primero: **0.045 → 0.773**; RAGAS faithfulness **0.597** |
| **M4** | Permite que la nota entre como imagen. | Delta cero frente a texto: F1 **0.617**, juez **4.417** |

---

## La idea que resume todo

```text
M1 detecta.
M2 evalúa.
M3 busca y responde.
M4 permite que la nota entre como imagen.
```

**M1 no diagnostica: solo detecta menciones. Sin esas menciones, M3 no sabe qué buscar.**
