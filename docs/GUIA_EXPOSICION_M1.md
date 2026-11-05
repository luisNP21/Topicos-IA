# Módulo 1 — Guía completa para exponer

> **Tema:** entrenamiento y evaluación del modelo que detecta enfermedades en notas clínicas.
> **Idea central:** M1 enseña a un modelo a **subrayar los nombres de enfermedades** dentro de un texto clínico. Esa es la primera pieza del sistema: sin entidades no hay nada que buscar en las guías.
> **Cómo usar esta guía:** está escrita para que puedas explicar el módulo de principio a fin, incluso si tu público no conoce todos los términos. Cada vez que aparece una idea sencilla, se introduce entre paréntesis el nombre técnico.

---

## 1. Ficha rápida: qué decir en 30 segundos

M1 resuelve una tarea de **reconocimiento de entidades nombradas (NER)** sobre texto clínico en español. Recibe una nota y devuelve una lista de enfermedades mencionadas. Se entrenaron tres modelos con **ajuste fino (fine-tuning)** y **adaptadores de bajo rango (LoRA)**: BERT español, RoBERTa biomedical-clinical y mT5-small. Se evaluaron con **F1** comparando las enfermedades predichas contra las anotadas. El mejor fue **RoBERTa biomedical-clinical + LoRA**, con **F1 0.708**.

---

## 2. Dónde encaja M1 dentro del sistema

El sistema completo hace esto:

```text
Nota clínica
   │
   ▼
[M1] Detectar enfermedades        ← este módulo
   │
   ▼
[M3] Buscar en guías clínicas
   │
   ▼
[M3] Redactar una respuesta
```

**Qué entrega M1:** una lista de entidades, por ejemplo:

```text
["diabetes mellitus tipo 2", "hipertensión arterial"]
```

**Qué NO hace M1:**

- No busca en las guías.
- No redacta la respuesta.
- No asigna códigos de SNOMED CT.
- No decide el tratamiento.

M1 solo resuelve la **detección de menciones**. La normalización a SNOMED CT se deja para el módulo 3.

### La tarea, en una frase

> “Dado un texto clínico, encontrar los fragmentos que son nombres de enfermedad.”

Eso es **NER** (*Named Entity Recognition* o reconocimiento de entidades nombradas). En este proyecto solo hay una categoría de entidad: `ENFERMEDAD`.

---

## 3. El dataset: DisTEMIST

### Qué es

**DisTEMIST** es un corpus de aproximadamente **1.000 casos clínicos en español** anotados con menciones de enfermedad. Proviene de casos publicados en SciELO / SPACCC y está disponible en Zenodo con licencia **CC BY 4.0**.

### Qué contiene cada caso

- Un texto clínico.
- Una lista de anotaciones.
- Cada anotación tiene:
  - posición inicial (`off0`);
  - posición final (`off1`);
  - texto de la mención (`span`);
  - etiqueta (`ENFERMEDAD`).

Ejemplo conceptual:

```text
Texto: "Paciente con diabetes mellitus tipo 2 e hipertensión arterial."
Anotaciones:
  - "diabetes mellitus tipo 2"  → ENFERMEDAD
  - "hipertensión arterial"     → ENFERMEDAD
```

### Los dos subtracks de DisTEMIST

DisTEMIST tiene dos tareas:

1. **`subtrack1_entities`:** detectar los fragmentos de texto que son enfermedades. Es la tarea **NER**.
2. **Normalización / linking:** asignar a cada mención un código de **SNOMED CT**, una ontología médica.

**Este proyecto usa solo el subtrack 1.** Es decir, M1 aprende a detectar el nombre de la enfermedad, pero no a decir “esto es el código 44054006 de SNOMED CT”. Eso se hace después, en M3, con una herramienta aparte.

> **Frase para la exposición:** “Usamos el subtrack de entidades porque M1 es un detector de menciones, no un normalizador de conceptos. La normalización a SNOMED CT se resuelve más adelante con una tool.”

### Dato importante sobre el tamaño

El notebook versionado en `main` carga **30 documentos** y hace el split **24 / 3 / 3** (train / dev / test). El `README.md` raíz menciona 750 casos, pero el código versionado no usa esa cantidad.

> **Recomendación para exponer:** di la cifra que realmente corriste. Si usas las métricas del notebook, di “30 documentos, con 24 de entrenamiento, 3 de validación y 3 de prueba”. Si dices 750, aclara que es el corpus completo disponible, no la corrida versionada.

---

## 4. Recorrido completo: los tres notebooks

El pipeline está dividido en tres notebooks. El orden importa:

```text
01_dataset_preparation.ipynb
        │  genera splits crudos
        ▼
02_dataset_chunking_and_splits_base.ipynb
        │  genera chunks + formato BIO + formato mT5
        ▼
03_finetuning_base.ipynb
           entrena y evalúa
```

---

### Notebook 1 — Preparación de datos

**Archivo:** `M1/01_dataset_preparation.ipynb`

**Objetivo:** cargar DisTEMIST, entender el problema de longitud y crear los splits.

#### Paso 1: cargar el corpus

El notebook lee:

- los textos en `text_files/*.txt`;
- las anotaciones en `subtrack1_entities/*.tsv`.

Cada fila del TSV tiene:

```text
filename | mark | label | off0 | off1 | span | ...
```

El código agrupa las anotaciones por `filename`, de modo que cada documento queda con su texto y su lista de entidades.

También corrige secuencias Unicode escapadas, por ejemplo `\u00f1` → `ñ`, con `fix_unicode_escapes()`.

#### Paso 2: entender el problema de longitud

Los modelos tipo BERT tienen un límite de **512 tokens** de entrada. Un token no es lo mismo que una palabra: el tokenizador puede partir una palabra en varios tokens.

Por ejemplo:

```text
"hipertensión" → ["hiper", "tensión"]   (puede variar según el tokenizador)
```

Si un documento tiene más de 512 tokens y se corta sin cuidado:

- se pierde el final de la nota;
- se pueden perder enfermedades;
- el modelo no ve el contexto completo.

Por eso el notebook mide:

- cuántos tokens tiene cada documento;
- cuántos tokens hay por palabra;
- qué ventana de palabras cabe de forma segura en 512 tokens.

La función `compute_safe_window_words()` usa el **percentil 90** del ratio tokens/palabra y un margen de seguridad de 0.9. La idea es:

```text
ventana segura ≈ (480 tokens / peor ratio tokens-palabra) × 0.9
```

Se apunta a 480, no a 512, para dejar espacio a los tokens especiales `[CLS]` y `[SEP]`.

#### Paso 3: dividir en train / dev / test

La función `make_split()`:

- mezcla los documentos con semilla 42;
- usa 80% para entrenamiento (`train`);
- 10% para validación (`dev`);
- 10% para prueba (`test`).

**Detalle importante:** el split se hace sobre **documentos completos**, antes de partirlos en chunks. Así se evita que fragmentos del mismo paciente queden en train y en test a la vez. Eso sería **fuga de datos** (*data leakage*) y haría que las métricas se vieran mejor de lo que son.

#### Paso 4: guardar

Guarda los splits crudos en formato Apache Arrow de HuggingFace `datasets`:

```text
distemist_raw_splits/
```

Cada registro conserva:

```python
{
  "doc_id": str,
  "text": str,
  "entities": [{"start": int, "end": int, "text": str, "label": str}]
}
```

---

### Notebook 2 — Chunking y formatos

**Archivo:** `M1/02_dataset_chunking_and_splits_base.ipynb`

**Objetivo:** partir los documentos largos y convertir cada ejemplo al formato que necesita cada arquitectura.

#### Paso 1: elegir tokenizador y ventana

Cada modelo tiene su propio tokenizador, y cada tokenizador parte el texto de forma distinta. Por eso la ventana de palabras no es la misma para todos:

| Modelo | Tokenizador | Ventana de palabras | Solapamiento |
|---|---|---:|---:|
| Base / RoBERTa clínico | `Roberta-clinico-es` | 277 | 50 |
| BERT español | `bert` | 246 | 50 |
| Clinical BERT / RoBERTa biomedical | `clinical_bert` | 237 | 50 |
| mT5 | `mT5` | 180 | 50 |

La ventana se eligió con el mismo criterio del Notebook 1: que el chunk quepa en 512 tokens sin truncar.

#### Paso 2: chunking con solapamiento

La función `chunk_record_by_words()`:

1. busca las palabras del texto con `re.finditer(r"\S+", text)`;
2. si el documento cabe en la ventana, no lo parte;
3. si no, toma una ventana de palabras;
4. avanza con un **solapamiento** (`overlap`) de 50 palabras;
5. reajusta las posiciones de las entidades al nuevo fragmento;
6. crea un `doc_id` derivado: `caso_XXX_chunk0`, `caso_XXX_chunk1`, etc.

**Por qué hay solapamiento:** para que una enfermedad que cae justo en el borde no quede cortada entre dos chunks. Es como leer un párrafo y volver a leer las últimas líneas antes de seguir.

**Ejemplo simplificado:**

```text
Documento: A B C D E F G H I J
Ventana 4, overlap 2:

chunk 0: A B C D
chunk 1:     C D E F
chunk 2:         E F G H
chunk 3:             G H I J
```

#### Paso 3: formato para encoders (BIO)

Los modelos BERT y RoBERTa no generan texto: clasifican cada palabra. Para eso se usan las **etiquetas BIO**:

- `B-ENFERMEDAD`: primera palabra de una enfermedad (*Begin*).
- `I-ENFERMEDAD`: palabra que continúa una enfermedad (*Inside*).
- `O`: palabra que no es enfermedad (*Outside*).

Ejemplo:

```text
Paciente   con   diabetes   mellitus   tipo   2   e   hipertensión   arterial
O          O     B-ENF      I-ENF      I-ENF  I-ENF O   B-ENF          I-ENF
```

La función `char_spans_to_bio()` convierte posiciones de carácter a etiquetas por palabra. Se tokeniza por espacios y se guardan los offsets de cada palabra.

El resultado se guarda como:

```python
{"doc_id": str, "tokens": list[str], "ner_tags": list[str]}
```

#### Paso 4: formato para mT5 (texto a texto)

mT5 no clasifica palabras: **genera texto**. La tarea se convierte en:

**Entrada:**

```text
Identifica las enfermedades mencionadas en el siguiente texto clínico:
<PEGAR TEXTO>
```

**Salida esperada:**

```text
diabetes mellitus tipo 2 [SEP] hipertensión arterial
```

Si no hay enfermedades:

```text
ninguna
```

La función `build_mt5_target()`:

- toma el término normalizado si existe, o el texto de la entidad;
- elimina duplicados;
- los une con `" [SEP] "`.

El resultado se guarda como:

```python
{"doc_id": str, "input_text": str, "target_text": str}
```

#### Paso 5: guardar

Se generan dos carpetas:

```text
distemist_bert_format/    # para BERT y RoBERTa
distemist_mt5_format/     # para mT5
```

---

### Notebook 3 — Fine-tuning

**Archivo:** `M1/03_finetuning_base.ipynb`

**Objetivo:** entrenar y comparar los tres modelos.

#### Paso 1: elegir el modelo

El notebook tiene un registro (`MODEL_REGISTRY`) con la configuración de cada modelo:

```python
MODEL_KEY = "bert"  # "bert" | "clinical_bert" | "mt5"
```

| Clave | Modelo base | Arquitectura |
|---|---|---|
| `bert` | `dccuchile/bert-base-spanish-wwm-cased` | Encoder |
| `clinical_bert` | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | Encoder |
| `mt5` | `google/mt5-small` | Encoder-decoder |

> **Aclaración importante:** la carpeta se llama `clinical_BERT`, pero el modelo real es un **RoBERTa biomedical-clinical**. No es un BERT clásico. Para exponer, puedes decir: “el modelo clínico, que técnicamente es un RoBERTa entrenado con texto biomédico”.

#### Paso 2: cargar el modelo base

Para los encoders:

```python
AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=3,
    id2label={0: "O", 1: "B-ENFERMEDAD", 2: "I-ENFERMEDAD"},
)
```

Para mT5:

```python
AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

#### Paso 3: baseline

Antes de entrenar, se evalúa el modelo base **sin LoRA**.

**Matiz importante:** al cargar un modelo de lenguaje para clasificación de tokens, la cabeza de clasificación (`classifier`) es nueva y está sin entrenar. Por eso el baseline da F1 casi cero. No significa que el modelo base “no sepa español”; significa que todavía no sabe qué etiqueta poner en cada palabra.

> **Frase para exponer:** “El baseline no es un modelo que ya sabe NER; es el mismo modelo con una cabeza nueva. Sirve para mostrar cuánto aporta el fine-tuning.”

#### Paso 4: aplicar LoRA

**LoRA** (*Low-Rank Adaptation*) es una técnica de **ajuste fino eficiente** (*parameter-efficient fine-tuning*). En lugar de modificar todos los pesos del modelo, se congelan los pesos originales y se entrenan unas matrices pequeñas adicionales.

En el proyecto:

```python
LoraConfig(
    r=8,                    # rango de las matrices nuevas
    lora_alpha=16,          # escala (regla común: alpha = 2 × r)
    lora_dropout=0.05,
    target_modules=["query", "value"],  # en encoders
    task_type="TOKEN_CLS",
)
```

Para mT5:

```python
target_modules=["q", "v"],
task_type="SEQ_2_SEQ_LM",
```

**Por qué se usa LoRA:**

- entrena muchos menos parámetros;
- ocupa menos memoria;
- es más rápido;
- permite guardar solo el adaptador, no el modelo completo.

#### Paso 5: entrenamiento

Hiperparámetros comunes:

| Parámetro | Valor |
|---|---:|
| Learning rate | `2e-4` |
| Épocas | 10 |
| Batch de entrenamiento | 2 |
| Batch de evaluación | 4 |
| Evaluación | por época |
| Guardado | por época |
| Mejor modelo | por F1 |
| Precisión mixta (`fp16`) | Sí en GPU para encoders |
| `fp16` en mT5 | **No** |

**¿Por qué mT5 usa `fp16=False`?** Porque T5/mT5 puede producir valores `NaN` en sus capas de normalización (`layer_norm`) con precisión de 16 bits. Se desactiva para que el entrenamiento sea estable.

El seguimiento de experimentos se hace con **Weights & Biases (W&B)**.

#### Paso 6: guardar

Con LoRA no se guarda el modelo completo, sino el adaptador:

```text
M1/saved_models/clinical_bert-distemist-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

Para usarlo después:

```python
base = AutoModelForTokenClassification.from_pretrained(checkpoint)
modelo = PeftModel.from_pretrained(base, adapter_path)
```

Ese adaptador es el que usan M2, M3 y M4.

---

## 5. Los tres modelos, explicados

### BERT español

- Modelo base: `dccuchile/bert-base-spanish-wwm-cased`.
- Es un **encoder** entrenado con texto general en español.
- Se le agrega una cabeza de clasificación de tokens.
- Aprende a poner `B-ENFERMEDAD`, `I-ENFERMEDAD` u `O` en cada palabra.

### RoBERTa biomedical-clinical

- Modelo base: `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`.
- También es un **encoder**, pero fue preentrenado con texto biomédico y clínico en español.
- Tiene ventaja con vocabulario médico: nombres de enfermedades, abreviaturas, términos técnicos.
- En el proyecto se guarda en la carpeta `clinical_BERT`, aunque técnicamente sea RoBERTa.
- Fue el ganador.

### mT5-small

- Modelo base: `google/mt5-small`.
- Es un **encoder-decoder** (*seq2seq*): recibe texto y genera texto.
- En vez de etiquetar palabras, genera una lista de enfermedades separadas por `[SEP]`.
- Es más difícil de entrenar para esta tarea y en la corrida generó repeticiones y texto extra.
- Quedó muy por debajo de los encoders.

---

## 6. Cómo se evalúa el modelo

### Para encoders

1. El modelo predice una etiqueta por palabra.
2. Se agrupan las etiquetas `B-` e `I-` en entidades.
3. Se obtiene una lista de enfermedades predichas.
4. Se compara con la lista correcta.

Ejemplo:

```text
Predicción: ["diabetes mellitus tipo 2", "hipertensión"]
Gold:       ["diabetes mellitus tipo 2", "hipertensión arterial"]
```

Aquí hay un acierto y un error de **boundary** (el límite de la segunda entidad quedó corto).

### Para mT5

1. El modelo genera texto.
2. Se parte por `[SEP]`.
3. Se obtiene una lista de enfermedades.
4. Se compara con la lista correcta.

Ejemplo:

```text
Salida generada: "shock séptico [SEP] CID"
Lista: ["shock séptico", "CID"]
```

### La métrica: F1

Primero se calculan dos medidas:

- **Precisión:** de todo lo que marqué como enfermedad, ¿cuánto era correcto?
- **Recall:** de todas las enfermedades que había, ¿cuántas encontré?

Luego se combinan:

```text
F1 = 2 × (precisión × recall) / (precisión + recall)
```

**Ejemplo sencillo:**

- Gold: 4 enfermedades.
- El modelo encuentra 3 correctas e inventa 1.

```text
Precisión = 3 / 4 = 0.75
Recall    = 3 / 4 = 0.75
F1        = 0.75
```

Si el modelo encuentra las 4 pero inventa 4 extra:

```text
Precisión = 4 / 8 = 0.50
Recall    = 4 / 4 = 1.00
F1        = 0.67
```

Por eso el F1 castiga tanto olvidar enfermedades como inventarlas.

### Evaluación por documento

El proyecto no cuenta entidades sueltas en todo el corpus. Hace esto:

1. Cada chunk tiene un `doc_id` como `caso_001_chunk3`.
2. Se quita el sufijo `_chunk3` para recuperar `caso_001`.
3. Se juntan las entidades de todos los chunks de ese caso.
4. Se calcula TP, FP y FN **por documento**.
5. Al final se suman todos y se calcula el F1 global.

Eso es **micro-F1 por documento**.

**Por qué se hace así:**

- Evita que una enfermedad repetida en varios chunks se cuente varias veces.
- Evita que un documento largo pese más que uno corto solo por tener más chunks.
- Refleja mejor el rendimiento sobre casos clínicos completos.

**TP / FP / FN:**

- **TP** (*True Positive*): enfermedad correcta que sí se detectó.
- **FP** (*False Positive*): enfermedad inventada.
- **FN** (*False Negative*): enfermedad real que no se detectó.

---

## 7. Resultados

| Modelo | Baseline F1 | Precisión | Recall | F1 final |
|---|---:|---:|---:|---:|
| BERT español | 0.0047 | 0.623 | 0.597 | **0.610** |
| RoBERTa biomedical-clinical | 0.0125 | 0.697 | 0.719 | **0.708** |
| mT5-small | 0.0000 | 0.161 | 0.094 | **0.119** |

### Cómo leer la tabla

- **BERT** funciona bien, pero el modelo clínico lo supera.
- **RoBERTa clínico** tiene el mejor equilibrio entre precisión y recall: encuentra más enfermedades y comete menos errores.
- **mT5** tiene un F1 muy bajo porque generar texto libre es más difícil que etiquetar palabras, y además produjo repeticiones y formato inconsistente en la prueba.

### Modelo seleccionado

Se eligió **RoBERTa biomedical-clinical + LoRA**:

- mejor F1;
- modelo de dominio clínico;
- encoder, que entrega spans exactos;
- no genera texto inventado;
- es el mismo tipo de modelo que usa el harness de M2.

---

## 8. Cuándo sube y cuándo baja el F1

### Sube cuando

- El modelo encuentra la enfermedad completa.
- El límite del fragmento coincide con el gold.
- No inventa enfermedades.
- No omite enfermedades.
- El chunk no corta la entidad.
- El vocabulario se parece a lo visto en entrenamiento.
- El modelo tiene buen equilibrio precisión/recall.

### Baja cuando

- **Omite una enfermedad** → baja el recall.
- **Inventa una enfermedad** → baja la precisión.
- **Corta el span:** “diabetes” en vez de “diabetes mellitus tipo 2” → error de boundary.
- **Alarga el span:** incluye palabras que no son parte de la enfermedad → error de boundary.
- **La entidad queda partida entre dos chunks** → se pierde o se detecta mal.
- **No reconoce siglas, erratas o variantes** → omisión.
- **mT5 genera texto repetido, incompleto o con formato distinto** → muchos FP/FN.
- **El test es muy pequeño** → un solo caso puede mover mucho la métrica.

---

## 9. Limitaciones y transparencia

Menciona estas limitaciones si te preguntan por la validez de los resultados:

1. **Tamaño de la corrida:** el notebook versionado usa 30 documentos (24/3/3). El README raíz menciona 750. Las métricas de la tabla corresponden a la corrida de 30.
2. **Test pequeño:** 3 documentos. El F1 tiene alta varianza; no es una estimación robusta.
3. **Una sola categoría:** solo `ENFERMEDAD`; no se distinguen tipos de enfermedad.
4. **Subtrack 1 solamente:** no se entrena la normalización a SNOMED CT.
5. **El baseline es una cabeza nueva:** su F1 casi cero no significa que el modelo base sea malo, sino que no estaba entrenado para NER.
6. **Desbalance de clases:** la mayoría de palabras son `O`; el notebook no usa ponderación de clases. Eso puede sesgar al modelo hacia “no es enfermedad”.
7. **Datos clínicos reales pero acotados:** el corpus no cubre todos los estilos de nota hospitalaria.
8. **No hay revisión manual de errores:** las métricas son automáticas; no se hizo una auditoría clínica de cada fallo.

---

## 10. Guion de exposición (10–12 minutos)

### Minuto 0–1: contexto

> “M1 es la primera etapa del sistema. Su trabajo es detectar nombres de enfermedades en una nota clínica. Sin esta lista, el resto del pipeline no sabe qué buscar en las guías.”

### Minuto 1–2: dataset

> “Usamos DisTEMIST, un corpus de casos clínicos en español con enfermedades anotadas. Usamos solo el subtrack de entidades, no el de normalización a SNOMED CT.”

### Minuto 2–4: preparación

> “El problema principal es que las notas son más largas que el límite de 512 tokens de los modelos. Por eso medimos tokens, elegimos una ventana segura por tokenizador y partimos el texto en chunks con solapamiento.”

### Minuto 4–6: formatos

> “Los encoders reciben etiquetas BIO por palabra. mT5 recibe texto y genera una lista separada por [SEP]. Ambos formatos salen del mismo split, así que comparamos sobre los mismos casos.”

### Minuto 6–8: modelos y LoRA

> “Probamos BERT español, RoBERTa biomedical-clinical y mT5. A los tres les aplicamos LoRA para entrenar solo una parte pequeña del modelo.”

### Minuto 8–10: evaluación y resultados

> “Evaluamos con micro-F1 por documento, agrupando los chunks del mismo caso. El mejor fue RoBERTa clínico con F1 0.708, frente a 0.610 de BERT y 0.119 de mT5.”

### Minuto 10–11: cierre

> “La conclusión es que un encoder de dominio clínico con LoRA es suficiente para detectar entidades con buena calidad, y es más adecuado que un modelo generativo para esta tarea. La limitación principal es el tamaño reducido del test.”

### Frase de cierre

> “M1 no diagnostica ni recomienda: solo convierte una nota en una lista de enfermedades. Es la base sobre la que M3 busca evidencia en las guías.”

---

## 11. Preguntas frecuentes

### ¿Por qué no usar un LLM para detectar enfermedades?

Porque un encoder:

- da posiciones exactas del span;
- es más barato y rápido;
- no inventa texto;
- es más fácil de evaluar con etiquetas BIO.

Un LLM generativo podría hacerlo, pero necesitaría formato estructurado y tendría riesgo de alucinación.

### ¿Por qué LoRA y no fine-tuning completo?

Porque LoRA:

- entrena menos parámetros;
- necesita menos memoria;
- es más rápido;
- permite guardar solo el adaptador.

Para un corpus pequeño, es una opción práctica.

### ¿Por qué el modelo clínico gana?

Porque su preentrenamiento incluyó vocabulario biomédico y clínico en español. Reconoce mejor términos como “insuficiencia cardíaca”, “adenopatías” o “shock séptico”.

### ¿Por qué mT5 falla tanto?

Porque la tarea cambia de “clasificar palabras” a “generar texto”. Eso es más difícil:

- puede repetir entidades;
- puede agregar texto extra;
- puede no respetar `[SEP]`;
- un error de formato arruina la comparación.

### ¿Por qué el F1 no es accuracy?

Porque la mayoría de palabras son `O`. Un modelo que diga “todo es O” tendría accuracy altísima, pero no detectaría ninguna enfermedad. El F1 obliga a mirar aciertos reales de enfermedad.

### ¿Por qué se usa solapamiento en los chunks?

Para que una enfermedad que cae en el borde no se pierda. El solapamiento repite parte del texto entre chunks.

### ¿Por qué se agrupan los chunks por documento?

Porque un caso clínico puede estar partido en varios chunks. Si se evalúan por separado, una misma enfermedad puede contarse varias veces o perderse. Agrupar por `doc_id` evalúa el caso completo.

### ¿Qué es exactamente el “boundary”?

Es el límite del fragmento. Si el gold es “diabetes mellitus tipo 2” y el modelo predice “diabetes”, el límite quedó corto. Si predice “diabetes mellitus tipo 2 del paciente”, quedó largo. Ambos son errores de boundary.

### ¿Se normaliza a SNOMED CT en M1?

No. M1 solo detecta la mención. La normalización a SNOMED CT se hace en M3 con BioPortal.

---

## 12. Glosario mínimo

| Término | Qué significa |
|---|---|
| **NER** | Reconocimiento de entidades nombradas; detectar nombres de cosas en texto. |
| **Entidad** | Fragmento de texto que representa algo concreto; aquí, una enfermedad. |
| **Span** | El fragmento exacto de texto que corresponde a una entidad. |
| **BIO** | Esquema de etiquetas `B-` (inicio), `I-` (continuación), `O` (fuera). |
| **Token** | Unidad mínima en que el tokenizador parte el texto; no siempre es una palabra. |
| **Tokenizador** | Programa que convierte texto en tokens. |
| **Chunk** | Pedazo de un documento largo. |
| **Overlap** | Solapamiento entre chunks para no cortar información. |
| **Fine-tuning** | Ajustar un modelo preentrenado a una tarea específica. |
| **LoRA** | Técnica de fine-tuning que entrena matrices pequeñas en vez de todo el modelo. |
| **Encoder** | Modelo que lee texto y produce representaciones; aquí clasifica tokens. |
| **Encoder-decoder** | Modelo que lee texto y genera texto; aquí mT5. |
| **Seq2seq** | Tarea de secuencia a secuencia; texto de entrada → texto de salida. |
| **Precisión** | De lo que predije, cuánto era correcto. |
| **Recall** | De lo que había, cuánto encontré. |
| **F1** | Media armónica de precisión y recall. |
| **TP / FP / FN** | Verdadero positivo, falso positivo, falso negativo. |
| **Baseline** | Punto de comparación antes de entrenar. |
| **Data leakage** | Fuga de información entre train y test; infla las métricas. |
| **SNOMED CT** | Ontología médica de conceptos clínicos. |
| **DisTEMIST** | Corpus de casos clínicos en español con enfermedades anotadas. |

---

## 13. Archivos clave de M1

| Archivo | Para qué sirve |
|---|---|
| `M1/01_dataset_preparation.ipynb` | Carga DisTEMIST, mide tokens y crea splits. |
| `M1/02_dataset_chunking_and_splits_base.ipynb` | Chunking, formato BIO y formato mT5. |
| `M1/03_finetuning_base.ipynb` | Entrenamiento, baseline, LoRA y evaluación. |
| `M1/BERT/` | Ejecución y adaptador de BERT español. |
| `M1/clinical_BERT/` | Ejecución y adaptador de RoBERTa biomedical-clinical. |
| `M1/mT5/` | Ejecución y adaptador de mT5. |
| `M1/saved_models/clinical_bert-distemist-lora/` | Adaptador seleccionado para el resto del sistema. |
| `M1/README.md` | Instrucciones de ejecución en Colab. |

---

## 14. Resumen final para memorizar

```text
M1 = detector de enfermedades.

Entrada: nota clínica.
Salida: lista de enfermedades.

Datos: DisTEMIST, subtrack de entidades.
Split: 80/10/10, semilla 42, sobre documentos completos.
Problema: notas > 512 tokens.
Solución: chunks con solapamiento, ventana por tokenizador.

Encoders: etiquetas BIO por palabra.
mT5: genera lista separada por [SEP].

Modelos: BERT español, RoBERTa clínico, mT5-small.
Técnica: LoRA (r=8, alpha=16).
Entrenamiento: 10 épocas, lr 2e-4, batch 2/4.

Métrica: micro-F1 por documento.
Resultados: BERT 0.610, RoBERTa clínico 0.708, mT5 0.119.
Ganador: RoBERTa clínico + LoRA.

Limitación principal: test de solo 3 documentos.
```

---

*Documento de apoyo para la exposición del Módulo 1. Basado en el estado de `main` en el commit `2958935`.*
