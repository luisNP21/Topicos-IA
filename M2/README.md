# Módulo M2 — Evaluación con LLM-as-a-Judge y Mitigación de Sesgos

Este directorio contiene la implementación de la dimensión cualitativa del sistema de evaluación de NER clínico (DistemIST) para la entrega M2, liderada por **Agustín**.

---

## 1. Contenido del Directorio

```text
M2/
├── 04_llm_judge.ipynb              # Notebook principal con la evaluación completa y salidas ejecutadas
├── eval_harness/
│   └── gold_examples.jsonl        # Gold set curado (59 ejemplos clínicos con input, esperado y criterio)
└── README.md                       # Documentación del módulo M2
```

---

## 2. Arquitectura del Juez LLM

- **Modelo Juez:** `openai/gpt-oss-120b` a través de la API de Groq (`temperature=0.0`, determinístico).
- **Modelo Evaluado:** `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` fine-tuneado con LoRA (M1).
- **Entrada al Juez:** Presentación estandarizada como listas de strings para ambas partes (Gold y Predicciones), procesadas con las funciones de inferencia de M1 (`chunking` con ventana de 277 palabras y 50 palabras de solapamiento).

### Rúbrica Clínica (1 a 5)
1. **Completitud:** ¿Capturó la mayoría de las enfermedades mencionadas?
2. **Exactitud de boundary:** ¿Los nombres coinciden con el gold (o casi)?
3. **Relevancia clínica:** ¿Las predicciones son términos patológicos válidos?
4. **Ausencia de ruido:** ¿Evitó marcar términos que no corresponden a patologías?

---

## 3. Protocolo de Mitigación de Sesgos

1. **Sesgo de Posición:**
   - Cada ejemplo se evalúa en orden **normal** e **invertido**.
   - Score final mitigado = promedio de ambos órdenes.
   - **Resultado:** $\Delta$ medio = 0.360 (Veredicto: **Estable**, $\Delta < 0.5$).
2. **Sesgo de Longitud:**
   - 5 pares de control (Tipo A: correcta corta vs. incorrecta larga; Tipo B: correcta larga vs. incorrecta corta).
   - **Resultado:** 5/5 (100%) victorias de calidad sobre longitud. No presenta sesgo de longitud relevante.
3. **Sesgo de Auto-preferencia:**
   - Output totalmente anonimizado (sin mención de arquitecturas ni modelos evaluados en los prompts).
   - Uso de un juez de familia independiente (Groq / GPT-OSS vs. RoBERTa).

---

## 4. Resumen de Resultados (59 Ejemplos Evaluados)

- **F1 Exact-Match Medio:** 0.718
- **Score Juez Medio (Mitigado):** 3.983 / 5.0
- **Correlación de Pearson:** 0.660 (Fuerte coherencia entre la métrica dura y el juicio clínico cualitativo).
- **Distribución de Patrones:**
  - **Patrón D (Referencia positiva — exacto y clínico alto):** 24 documentos (40.7%).
  - **Patrón E (Caso mixto — comprensión clínica adecuada con variaciones menores de span):** 35 documentos (59.3%).
