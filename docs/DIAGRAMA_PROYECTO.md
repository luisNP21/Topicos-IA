# Diagramas del proyecto

> Estado de `main` en el commit `2958935`.
> El archivo incluye una **vista general** y tres **vistas de detalle** para que el diagrama no se vuelva ilegible.

## 1. Vista general

```mermaid
%%{init: {"theme": "base", "themeVariables": {
  "background": "#fbf5e9",
  "primaryColor": "#fffaf0",
  "primaryTextColor": "#3e3a34",
  "primaryBorderColor": "#7fa8c9",
  "lineColor": "#9a8b78",
  "secondaryColor": "#ffe3c2",
  "tertiaryColor": "#e8f2fb",
  "clusterBkg": "#f8f1e3",
  "clusterBorder": "#dccbb2",
  "edgeLabelBackground": "#fbf5e9",
  "titleColor": "#3e3a34",
  "textColor": "#3e3a34",
  "fontFamily": "Trebuchet MS, Segoe UI, sans-serif"
}}}%%
flowchart LR
  subgraph PREP["Preparación (offline)"]
    direction TB
    D1["Casos clínicos anotados<br/><small>DisTEMIST</small>"]
    TRAIN["Entrenamiento del detector<br/><small>3 modelos + LoRA</small>"]
    ADAPTER[("Modelo entrenado")]
    GUIDES["Guías clínicas PDF"]
    CORPUS["Construcción del índice<br/><small>Docling + chunks + embeddings</small>"]
    D1 --> TRAIN --> ADAPTER
    GUIDES --> CORPUS
  end

  subgraph RUNTIME["Sistema en uso"]
    direction TB
    NOTE["Nota clínica"]
    UI["Aplicación<br/><small>Gradio</small>"]
    VIS["Lectura de imagen<br/><small>VLM / OCR</small>"]
    NER["Detección de enfermedades<br/><small>NER + LoRA</small>"]
    RET["Búsqueda en guías<br/><small>retrieval + reranker</small>"]
    CHROMA[("Índice de guías<br/><small>Chroma</small>")]
    GATE{"¿Hay evidencia?"}
    GEN[["Generación de respuesta<br/><small>LLM · Groq</small>"]]
    VALVE["Respuesta de abstención<br/><small>fallback_used = true</small>"]
    ANS(["Respuesta<br/><small>+ fuentes</small>"])
    NOTE -->|imagen| UI --> VIS
    NOTE -->|texto| NER
    VIS -->|texto| NER
    NER --> RET
    RET <--> CHROMA
    RET --> GATE
    GATE -->|sí| GEN --> ANS
    GATE -->|no| VALVE --> ANS
  end

  subgraph EVAL["Evaluación (offline)"]
    direction TB
    EVAL_INF["Predicciones del detector"]
    GOLD_NER[("Casos anotados")]
    HARNESS["Evaluación del NER<br/><small>exact · semántica · juez</small>"]
    SCORE["Resultado del NER"]
    EVAL_SET[("Casos de evaluación<br/><small>respuesta esperada</small>")]
    EVAL_GEN["Generación de evaluación"]
    RESULT_GEN[("resultado_generacion.json")]
    RAGAS["RAGAS"]
    CROSS["Cruce de resultados"]
    ADAPTER --> EVAL_INF --> HARNESS
    GOLD_NER --> HARNESS
    HARNESS --> SCORE --> CROSS
    EVAL_SET --> EVAL_GEN --> RESULT_GEN --> RAGAS --> CROSS
  end

  CORPUS -->|indexa| CHROMA
  ADAPTER --> NER

  classDef core fill:#e6f1fb,stroke:#6d9dc5,stroke-width:1.5px,color:#243b53
  classDef data fill:#ffe8cf,stroke:#d98b4a,stroke-width:1.5px,color:#5a3b22
  classDef valve fill:#ffe0d6,stroke:#d97b66,stroke-width:1.5px,color:#5a2a22
  classDef train fill:#f7ead3,stroke:#c49a6c,stroke-width:1.5px,color:#4a3728
  classDef eval fill:#fff0d6,stroke:#d9a441,stroke-width:1.5px,color:#5a4419
  classDef ui fill:#e6f0ff,stroke:#7a9cc6,stroke-width:1.5px,color:#2f3f5a
  classDef vis fill:#ffeadb,stroke:#e09a6a,stroke-width:1.5px,color:#5a3520
  classDef io fill:#fffaf0,stroke:#b8a58f,stroke-width:1.5px,color:#3f3a34
  class NER,RET,GATE,EVAL_INF,HARNESS,EVAL_GEN core
  class ADAPTER,CHROMA,EVAL_SET,GOLD_NER,RESULT_GEN data
  class TRAIN,CORPUS train
  class SCORE,RAGAS,CROSS eval
  class UI ui
  class VIS vis
  class GEN core
  class VALVE valve
  class D1,GUIDES,NOTE,ANS io

  style PREP fill:#fdf3e3,stroke:#d9a441,color:#5a4419
  style RUNTIME fill:#eaf4fb,stroke:#7fa8c9,color:#243b53
  style EVAL fill:#f6efe2,stroke:#c9a86a,color:#4a3a24
```

## 2. Detalle del detector de enfermedades

```mermaid
flowchart LR
  subgraph TRAIN_BLOCK["Entrenamiento"]
    direction TB
    D1["DisTEMIST<br/><small>subtrack1_entities</small>"]
    PREP["Preparación<br/><small>split 80/10/10</small>"]
    CH["Chunking + overlap<br/><small>BIO / seq2seq</small>"]
    TRAIN["Fine-tuning<br/><small>BERT · RoBERTa clínico · mT5<br/>PEFT + LoRA</small>"]
    ADAPTER[("Adaptador LoRA<br/><small>RoBERTa clínico</small>")]
    D1 --> PREP --> CH --> TRAIN --> ADAPTER
  end

  subgraph EVAL_BLOCK["Evaluación"]
    direction TB
    INF["run_inference<br/><small>chunks 277/50</small>"]
    CACHE[("Cache NER<br/><small>doc_id → entidades</small>")]
    GOLD[("Gold sets<br/><small>59 gold · 10 adversariales</small>")]
    HARNESS["Evaluación del NER<br/><small>exact · semántica · LLM-judge</small>"]
    SCORE["Scorecard<br/><small>diagnóstico por documento</small>"]
    RESULT(["F1 por dimensión"])

    ADAPTER --> INF
    INF --> CACHE --> HARNESS
    GOLD --> HARNESS
    HARNESS --> SCORE --> RESULT
  end

  subgraph EVAL["Cruce"]
    direction TB
    CROSS["RAGAS × resultado del NER"]
    SCORE --> CROSS
  end

  classDef train fill:#eef2ff,stroke:#4f46e5,stroke-width:1.5px
  classDef data fill:#f0fdfa,stroke:#0d9488,stroke-width:1.5px
  classDef core fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px
  classDef eval fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px
  class PREP,CH,TRAIN train
  class ADAPTER,CACHE,GOLD data
  class INF,HARNESS core
  class SCORE,RESULT,CROSS eval
```

## 3. Detalle del corpus, búsqueda y generación

```mermaid
flowchart LR
  subgraph CORPUS["Corpus"]
    direction TB
    G1["Guías PDF<br/><small>3 guías</small>"]
    DOC["Docling<br/><small>secciones + tablas</small>"]
    CH["Chunks<br/><small>500/50 tokens</small>"]
    EMB["Embeddings<br/><small>multilingual-e5-base</small>"]
    CHROMA[("Chroma<br/><small>64 chunks</small>")]
    G1 --> DOC --> CH --> EMB --> CHROMA
  end

  subgraph RUNTIME["Runtime"]
    direction TB
    NER["NER<br/><small>RoBERTa + LoRA</small>"]
    RET["Retrieval<br/><small>densa + BM25 + RRF</small>"]
    RER["Reranker<br/><small>mmarco-mMiniLMv2</small>"]
    GATE{"Compuerta<br/><small>de evidencia</small>"}
    TOOL["Tool de normalización<br/><small>SNOMED CT</small>"]
    SEL{"elegir_principal<br/><small>solo app</small>"}
    GEN[["Generación RAG<br/><small>prompt S07 + válvula</small>"]]
    ANS(["Respuesta<br/><small>+ fuentes</small>"])
    VALVE["Válvula<br/><small>«No tengo esa información»</small>"]

    NER --> RET --> RER --> GATE
    RET <--> CHROMA
    GATE -->|score bajo| TOOL
    TOOL -->|reintenta| RET
    GATE -->|sin evidencia| VALVE
    GATE -->|evidencia| SEL
    GATE -.->|por entidad| GEN
    SEL --> GEN --> ANS
  end

  subgraph EXT["Servicios externos"]
    direction TB
    BIO[("BioPortal<br/><small>SNOMED CT</small>")]
    GROQ[["Groq API<br/><small>qwen3.8-27b · T=0</small>"]]
    TOOL <--> BIO
    GEN --> GROQ
  end

  subgraph EVAL["Evaluación"]
    direction TB
    M2SCORE["Resultado del NER"]
    RAGAS["RAGAS<br/><small>faithfulness · precision<br/>recall · relevancy</small>"]
    CROSS["Cruce<br/><small>RAGAS × resultado del NER</small>"]
    ANS --> RAGAS --> CROSS
    M2SCORE --> CROSS
  end

  classDef data fill:#f0fdfa,stroke:#0d9488,stroke-width:1.5px
  classDef core fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px
  classDef valve fill:#fef2f2,stroke:#dc2626,stroke-width:1.5px
  classDef llm fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px
  classDef ext fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px
  classDef eval fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px
  class CHROMA data
  class NER,RET,RER core
  class GATE,TOOL,SEL,VALVE valve
  class GEN,ANS llm
  class BIO,GROQ ext
  class M2SCORE,RAGAS,CROSS eval
```

## 4. Detalle de visión y app

```mermaid
flowchart LR
  subgraph INPUT["Entrada"]
    direction TB
    NOTE["Nota clínica<br/><small>texto o imagen</small>"]
    UI["App Gradio<br/><small>interfaz</small>"]
    NOTE -->|imagen| UI
  end

  subgraph VISION["Visión"]
    direction TB
    VIS["Transcripción<br/><small>VLM Qwen2-VL<br/>u OCR Docling/Tesseract</small>"]
    OK{"¿Texto<br/>legible?"}
    SET[("Set visual sintético<br/><small>15 casos · 28 niveles</small>")]
    CER["CER/WER<br/><small>+ auditoría de entidades</small>"]
    UI --> VIS
    SET --> VIS
    VIS --> OK
    VIS --> CER
  end

  subgraph RUNTIME["Runtime RAG"]
    direction TB
    NER["NER<br/><small>RoBERTa + LoRA</small>"]
    CHROMA[("Chroma<br/><small>3 guías</small>")]
    RET["Retrieval + reranker"]
    TOOL["Tool SNOMED<br/><small>BioPortal</small>"]
    GATE{"Compuerta<br/><small>de evidencia</small>"}
    SEL{"elegir_principal<br/><small>mayor score</small>"}
    GEN[["Generación<br/><small>Groq · qwen3.8-27b</small>"]]
    ANS(["Respuesta<br/><small>+ fuentes + texto leído</small>"])
    VALVE["Válvula<br/><small>sin evidencia</small>"]

    NOTE -->|texto| NER
    OK -->|sí| NER
    OK -->|no| VALVE
    NER --> RET --> GATE
    RET <--> CHROMA
    GATE -->|score bajo| TOOL
    TOOL -->|reintenta| RET
    GATE -->|sin evidencia| VALVE
    GATE -->|evidencia| SEL
    SEL --> GEN --> ANS
  end

  subgraph EVAL["Evaluación"]
    direction TB
    HARNESS["Evaluación del NER<br/><small>delta vs texto gold</small>"]
    SCORE["Scorecard visión"]
    RAGAS["RAGAS"]
    NER --> HARNESS --> SCORE
    ANS --> RAGAS
  end

  classDef io fill:#f8fafc,stroke:#475569,stroke-width:1.5px
  classDef ui fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px
  classDef vis fill:#fff7ed,stroke:#ea580c,stroke-width:1.5px
  classDef data fill:#f0fdfa,stroke:#0d9488,stroke-width:1.5px
  classDef core fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px
  classDef valve fill:#fef2f2,stroke:#dc2626,stroke-width:1.5px
  classDef llm fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px
  classDef eval fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px
  class NOTE,ANS io
  class UI ui
  class VIS,CER vis
  class CHROMA,SET data
  class NER,RET core
  class OK,TOOL,GATE,SEL,VALVE valve
  class GEN llm
  class HARNESS,SCORE,RAGAS eval
```

## Cómo leer los diagramas

- **Celeste claro:** núcleo del pipeline y app.
- **Naranja claro:** datos persistidos y visión.
- **Crema/mostaza:** entrenamiento del detector.
- **Durazno/rojo suave:** decisiones y válvulas (compuerta, `elegir_principal`, fallback).
- **Amarillo suave:** evaluación.
- **Gris:** servicios externos.

> La **vista general** usa la paleta crema con celeste y naranja claros. Los diagramas de detalle conservan la paleta anterior.

## Flujo resumido

```text
Texto:   nota → NER → retrieval → compuerta → generación → respuesta
Imagen:  nota → VLM/OCR → NER → mismo flujo
Eval:    se evalúa el NER · RAGAS evalúa una corrida offline de generación · CER/WER evalúa la visión
```

> **Nota:** el entrenamiento y la evaluación son procesos **offline**. En una solicitud real se usan el modelo entrenado, el detector de enfermedades, el índice de guías, la búsqueda y la generación. La compuerta y `elegir_principal` son reglas deterministas, no modelos.
>
> **RAGAS no evalúa la respuesta en vivo de la app.** Evalúa una corrida offline sobre el eval set del sistema (`resultado_generacion.json`). Por eso el diagrama separa la respuesta de la app de la rama de evaluación.
>
> **Si la compuerta falla:** no se llama al LLM. Se devuelve la válvula de escape (`fallback_used = true`) con el mensaje «No tengo esa información en mis fuentes»; en la app, además, no se elige enfermedad principal.
