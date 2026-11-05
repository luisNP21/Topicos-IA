# Plan sugerido para el Demo Day (5-nov): Asistente de guías clínicas

> Son sugerencias de revisión, no requisitos nuevos. El equipo decide qué adopta y quién lo asume.
> Referencia inspeccionada: `4faa880`. Prototipo académico, no es un dispositivo clínico.

## Lo que ya tienen

Ya tienen un producto completo, no una colección de notebooks. `M4/app/app.py` es un chat en Gradio que encadena las cuatro entregas:

```
foto de la nota ─► M4/vision (Qwen2-VL-2B) ─► NER Clinical BERT + LoRA (M1, chunking de M2)
                 ─► orquestación + retrieval híbrido + compuerta de evidencia (M3)
                 ─► respuesta redactada por el LLM con fuentes (M3) ─► chat con texto leído, fuentes y tiempos
```

- Se abstiene cuando las guías no cubren el caso.
- Muestra el texto leído, las fuentes con su sección y el tiempo de cada etapa.
- Avisa cuando falta la clave del LLM o algún recurso.
- La lectura es muy fiel: CER 0,0002 en el set limpio, también en otro hardware.
- **Y la foto no le cuesta nada al sistema:** sobre las mismas 15 notas y 62 entidades, el NER saca exactamente lo mismo de la foto que del texto escrito (README de la revisión, §3).

El punto de entrada para desplegar (M5) es `python M4/app/app.py`, o `M4/app/start_app.ipynb` en Colab. Las sugerencias de despliegue de abajo son para cuando lo hagan.

## Guion sugerido (bloques, sin minutos obligatorios)

1. **La persona y su dolor.** Una residente con la nota de un paciente, impresa o fotografiada. Hoy busca a mano, en guías de cientos de páginas, la sección que aplica. La decisión que toma: qué revisar primero.
2. **El momento wow, en vivo.** Toma la foto, la arrastra al chat y en segundos ve:
   - la enfermedad sobre la que las guías pueden responder («DM2» → «Diabetes mellitus tipo 2»);
   - qué dicen las guías;
   - las fuentes, con guía y sección.
   No transcribió nada: la imagen es la entrada natural.
3. **Cómo funciona por dentro (breve).** El diagrama de arriba. Cada pieza es una entrega que pasó por su propio harness.
4. **Así se comporta con seguridad.**
   - Una nota de un tema que no está en las guías: dice que no tiene esa información, en vez de inventar.
   - Una foto recortada: avisa que la lectura es automática, muestra lo que leyó y pide verificarlo (P1).
5. **La prueba de valor.** Uno a tres números con su denominador (P2):
   - «Lee la nota casi sin errores: CER 0,0002 en 15 notas».
   - «La foto rinde igual que el texto: 29 de 62 hallazgos en ambos casos».
   - Tiempo por consulta y costo: la visión es local, con una llamada al LLM por hallazgo.
6. **Así lo operamos.** Versión etiquetada y tiempos por etapa en «Detalles técnicos». Si al desplegar añaden health check o W&B, este es el lugar para mostrarlos.
7. **Límites y siguiente paso.**
   - Tres guías cubiertas.
   - La extracción encuentra menos de la mitad de los hallazgos anotados: es lo siguiente que mejorarían.
   - Las notas manuscritas se leen peor.
   - No diagnostica.
   - Roadmap: más guías, normalización SNOMED (DisTEMIST ya la trae) y prueba con usuarios.

## Prioridades (máx. 3)

### P0. Que el recorrido foto → respuesta sea estable y rápido en vivo

- **Qué podría ocurrir durante la demo:** la primera foto tarda más de un minuto y el momento wow se diluye; o falla el arranque porque falta un recurso, o falla la red o la API del LLM.
- **Evidencia exacta:**
  - `M4/README.md` L301: 75,7 s para una página en una T4. En el harness, el VLM tarda unos 3,4 s por página, así que buena parte de ese tiempo puede ser la carga del modelo (no lo verificamos).
  - El VLM se crea en la primera solicitud con imagen (`M4/app/pipeline_app.py`, `Vision._backend`).
  - El único ejemplo precargado del chat es de texto (`M4/app/app.py` L270).
  - El arranque exige `PROJECT_ROOT` con el adaptador LoRA y los chunks del corpus desde Drive.
- **Cambio mínimo recomendado:**
  1. Precalentar el VLM al arrancar con una imagen pequeña y medir aparte la carga y la lectura por página.
  2. Añadir 2 o 3 ejemplos con imagen al chat: un caso cubierto, uno sin cobertura y la foto recortada de P1.
  3. Etiquetar la versión de la demo.
  4. Preparar un plan B rotulado («salida grabada»): una grabación del recorrido o las respuestas guardadas de los casos ensayados.
  5. Al desplegar, conviene no depender de Drive. El adaptador ya está versionado en `M1/clinical_BERT/clinical_bert-distemist-lora/` (con él reprodujimos su TP 29 / FP 3) y las guías están en `M3/corpus/`.
- **Qué NO hace falta rehacer:** la app, el pipeline ni el modelo de visión.
- **Cómo verificar el cierre:** tres ensayos seguidos del guion desde un arranque limpio de la versión etiquetada, con el tiempo por página anotado tras la carga, y el plan B abriéndose sin red.
- **Quién lo asume:** lo decide el equipo.

### P1. Convertir el caso límite visual en un guardrail visible

- **Qué podría ocurrir durante la demo:** alguien sube una foto recortada. El lector completa la frase cortada con una enfermedad que no está en la imagen, y esa lectura pasa al NER y a la recuperación como si fuera parte de la nota.
- **Evidencia exacta** (su VLM, con su prompt y su post-proceso, en Apple MPS fp16; observación M4-05):
  - `M4/eval_set/ejemplos_finales/retos/rotacion_recorte_severo/ex_30_p2.jpg`: la lectura añade «enfermedad pulmonar obstructiva crónica» y «elevada concentración de plaquetas», y el NER extrae la primera como hallazgo.
  - `ex_22_p2.jpg`: responde «Se aprecia texto legible en la imagen.», que `es_valvula` (`M4/vision/prompts.py` L41) no reconoce.
  - El texto leído se muestra plegado (`M4/app/app.py` L172–174).
- **Cambio mínimo recomendado:** ya está en la rama de sugerencias (C1, #CODE_PR):
  1. `es_valvula` reconoce el eco sin contenido.
  2. Junto a la respuesta, la app muestra: «⚠️ Los hallazgos salen de una lectura automática de la imagen; revisa el texto leído», y cuántas páginas no se pudieron leer.
  3. `M4/evaluacion_pareada/adversariales.csv` deja los retos con su criterio («ninguna entidad que no esté en la imagen») y `evaluar_adversariales.py` da pasa / no pasa (C2).
  4. En la demo, mostrarlo como comportamiento de seguridad del producto, no como un error.
- **Qué NO hace falta rehacer:** el prompt, el modelo ni el generador de retos.
- **Cómo verificar el cierre:** `ex_22_p2` termina como «sin texto legible»; con `ex_30_p2` la app muestra el aviso y el texto leído; `test_vision.py` pasa con las 2 pruebas nuevas.
- **Quién lo asume:** lo decide el equipo.

### P2. Una prueba de valor con denominador: foto frente a texto en las mismas notas

- **Qué podría ocurrir durante la demo:** preguntan «¿cuánto se pierde por leer una foto?» y el único delta disponible (−0,110) compara 20 documentos DisTEMIST con 15 casos de M3 (observación M4-01).
- **Evidencia exacta:**
  - `M4/README.md` L25–33 y L756 presentan los tres gold como el mismo contenido.
  - Con el mismo NER sobre las mismas 15 notas y 62 entidades: texto, imagen limpia, foto realista y sombra dan 29 aciertos y F1 0,617; manuscrito, 0,568.
  - La auditoría de entidades usa otro gold (7 aciertos y 25 FP frente a 29 y 3; M4-03), y 5 entidades del gold no están literalmente en su nota (M4-04).
- **Cambio mínimo recomendado:**
  1. Correr el harness con `eval_set_path: M4/eval_set/eval_set_casos.jsonl` en texto y en visión, y escribir `M4/scorecard_m4.csv` con `delta_pareado.py` (C2).
  2. Contar una frase de producto: «De la foto saca lo mismo que del texto: 29 de 62 hallazgos en 15 notas; tarda X s por página».
  3. Corregir o justificar las 5 entidades con `verificar_gold.py --literal` y alinear la auditoría con `auditar_entidades.py`.
- **Qué NO hace falta rehacer:** el harness ni el set de imágenes.
- **Cómo verificar el cierre:** `scorecard_m4.csv` versionado con filas texto / limpio / foto realista sobre los mismos 15 `doc_id`, y una diapositiva con 1 a 3 números y su denominador.
- **Quién lo asume:** lo decide el equipo.

## Backlog breve (no bloquea la demo)

- Restaurar `M4/eval_set/ejemplos_finales/` (se borró de `main` después de la entrega); contiene los retos.
- Versionar los `resultado_dimension1.json`, `resumen.md` y una nota de procedencia del gold.
- Un resumen de una página arriba del README de M4.
- Mejorar el recall del NER sobre los casos de M3: es el cuello de botella real, no la visión.
- Al desplegar, conviene:
  - un health check simple (modelo cargado, índice con N chunks, LLM alcanzable);
  - registrar la latencia por etapa (los tiempos ya se calculan en `pipeline_app.py`).
