# Revisión docente de M3 y orientación para M4 / Demo

## Alcance y referencia

Revisión técnica de `luisNP21/Topicos-IA`, preparada desde el fork docente. Referencia inspeccionada: `a478949f8b162d62dec022ee74def32fdab21f6f`. Esta referencia técnica no se presenta como certificación del commit entregado en Interactiva ni redefine el corte académico. La calificación se comunica por el canal privado del curso y no se publica aquí.

M3 es el objeto de la revisión. M1 y M2 se consideran únicamente en su relación con M3. Las recomendaciones de M4 y Demo no son requisitos retroactivos. Las correcciones demostrativas de la docente no son trabajo atribuible al equipo en la entrega original.

Este documento es retroalimentación. Las sugerencias ejecutables se presentan en un PR correctivo separado, desde `manularrea:revision/m3-evaluacion`. No fusionar este PR documental como si fuera una corrección del sistema.

## Fortalezas que se reconocen

**Ingeniería del harness.** La separación entre inferencia y evaluación permite volver a evaluar las mismas predicciones sin cargar de nuevo el modelo NER. El contrato `harness(eval_set, sistema, cfg, project_root, solo)` centraliza las dimensiones de evaluación. `cached_system.py` comprueba correspondencia de identificadores, tipos de predicciones y textos duplicados asociados a identificadores distintos. Estos controles y la interfaz de línea de comandos son aportes concretos de estandarización y reutilización.

Fuentes: [harness.py](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M2/harness/harness.py), [cached_system.py](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M2/harness/cached_system.py) y [adaptación documentada en M3](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M3/README.md#4-adaptación-del-harness-de-m2). El propio informe aclara que el harness evalúa `encoder_solo`; no se atribuye a esa infraestructura una evaluación completa de todas las variantes del RAG.

**Retrieval y experimentación.** Se reconoce la comparación entre búsqueda densa, BM25 + densa con RRF, reranking y transformación de consultas; la distinción entre recuperar una guía relacionada y un fragmento que responda; el análisis de latencia; la validación del gold set contra el índice y la calibración de umbrales. Las cifras del informe son resultados reportados por el equipo, no resultados de una nueva corrida completa realizada por la docente.

**Transparencia de limitaciones.** El README distingue evaluación local y real, publica resultados desfavorables y declara las limitaciones del corpus sintético, las anotaciones con IA sin revisión humana y la normalización simulada. Reconocer estas limitaciones es positivo; no equivale a corregir sus efectos sobre la validez experimental.

Fuente: [M3/README.md, secciones 5 y 7](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M3/README.md).

## Relación con lo enseñado

| Material del curso | Uso en esta revisión |
|---|---|
| [S06: harness](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s06/README.md) | La reutilización del harness ya era parte del curso; se reconoce la calidad de la adaptación, no se vuelve a calificar M2. |
| [S07: RAG ingenuo](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s07/README.md) | Pipeline, baseline, prompt aumentado y abstención. |
| [S08: retrieval avanzado](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s08/README.md) | Comparación de técnicas, delta y latencia. |
| [S10: tool use y RAGAS](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s10/README.md) | Evaluación, interpretación honesta y herramienta. El cálculo manual se enseñó: no es un defecto por sí mismo. |
| [S13: integración visual](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s13/README.md) | Orientación futura, no descuento anticipado sobre M3. |

No se exige un umbral de rendimiento inventado, W&B obligatorio para M3, Docker, una refactorización general ni herramientas externas adicionales. Los aportes de ingeniería se reconocen sin convertirlos en obligaciones nuevas para otros equipos.

## Observaciones y sugerencias accionables

### M3-01 — Procedencia de métricas y resultados no calculados

En [evaluacion_ragas.py](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M3/ragas/evaluacion_ragas.py#L275-L385), el fallo de la evaluación real o cuatro ceros pueden activar un proxy local; el resultado exterior puede conservar `modo: real`. Hay advertencia en consola: no se describe como ausencia absoluta de aviso. Un `NaN` también puede terminar convertido en cero.

**Sugerencia C1:** conservar ceros medidos; registrar valores no calculados como `null` / `N/D`, con denominadores; mantener el proxy como opción explícita; no inventar referencias ni reemplazar automáticamente el conjunto evaluado por ejemplos. Un fallo debe declararse como fallo. No se afirma que la corrida histórica haya usado el respaldo.

### M3-02 — Abstención no equivale a fidelidad perfecta

El evaluador contiene asignaciones fijas de `1.0` y `0.85` bajo determinadas condiciones de abstención. Su activación depende del esquema de salida; no se afirma que hayan alterado necesariamente la corrida histórica. La evaluación local puede premiar una respuesta mixta por contener una frase de abstención.

**Sugerencia C1:** conservar los valores originales del evaluador y registrar `abstencion_declarada` por separado. La bandera del generador no certifica seguridad. Probar una respuesta como «No hay evidencia suficiente. El objeto es azul.» sin contexto: una frase inicial no respalda la afirmación posterior. La convención de no disponibilidad en el proxy se documenta como decisión de ese proxy, no como definición obligatoria de RAGAS.

### M3-03 — No atribuir promedios a documentos no pareados

[cruce_harness.py](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M3/ragas/cruce_harness.py#L115-L143) utiliza las mismas métricas globales de M3 para diagnosticar los documentos históricos de M2. Esa operación no demuestra el comportamiento de cada documento.

**Sugerencia C2:** mantener intacto el cálculo de F1 de M2 y mostrar ambos agregados separados hasta disponer de correspondencia verificable. `diagnosticos={}` significa que no se hizo diagnóstico individual, no que no haya errores. No se requiere reentrenar M1 ni reevaluar obligatoriamente todos los documentos de M2 para aplicar esta corrección mínima.

### Mejora adicional — Persistencia de reportes

[run_ragas.py](https://github.com/luisNP21/Topicos-IA/blob/a478949f8b162d62dec022ee74def32fdab21f6f/M3/ragas/run_ragas.py) construye e imprime el JSON. Eso no invalida los notebooks guardados por el equipo.

**Sugerencia C3:** escribir un archivo nuevo por corrida con fecha UTC, resultados por caso, denominadores, método, versiones y hashes del evaluador. No sobrescribir resultados históricos. Mantener la consola y un artefacto de error cuando falle la evaluación. Es una mejora de ingeniería, no un requisito adicional retroactivo.

## Qué no cambia el PR correctivo

Solo se modifican `M3/ragas/evaluacion_ragas.py`, `M3/ragas/cruce_harness.py` y `M3/ragas/run_ragas.py`, más pruebas y evidencia docente. No se modifican retrieval, corpus, modelos, normalización, pesos de M1, harness de M2, tablas históricas ni notebooks entregados.

## Validación y límites

El paquete aprobado y sus tres archivos originales se verificaron mediante hashes Git blob. Se comprobó `git apply --check`, se aplicó el patch a una copia temporal y se comparó byte a byte con la propuesta. Las 26 pruebas de regresión con dobles controlados se reejecutaron antes de publicar y aprobaron.

No es una nueva ejecución del pipeline clínico completo ni de la suite completa del equipo. No se realizaron llamadas a Groq/BioPortal ni descargas de modelos. La revalidación local usa Python 3.13.5, NumPy 2.3.5 y pandas 2.2.3; no equivale al entorno con todas las dependencias fijadas por el equipo. Las pruebas no certifican seguridad clínica.

En la rama correctiva, desde la raíz del repositorio:

```bash
python -m unittest discover -s review/m3/tests -p test_suggestions.py -v
```

Antes de fusionar: ejecutar con sus dependencias fijadas un lote pequeño autorizado, revisar los consumidores que asumen métricas siempre `float` y comprobar que admitan `None`. Los tres módulos del PR están adaptados; no se han auditado todos los notebooks consumidores.

## Antes de M4 y la Demo

Para M4, mantener el patrón visual justificado, los casos con imagen y el delta con latencia conforme al material de S13. Conservar casos textuales comparables para detectar regresiones. No comparar M3 con el evaluador antiguo contra M4 con uno distinto: conservar lo entregado y producir, cuando sea viable, una nueva referencia M3 y una medición M4 bajo el mismo protocolo. Esto no modifica retroactivamente la evaluación académica.

Para la Demo, preparar una ejecución reproducible y mostrar tanto un caso exitoso como una limitación. Distinguir documentos sintéticos de fuentes originales; el prototipo no debe presentarse como validación clínica. Priorizar correcciones acotadas antes que una reescritura general.
