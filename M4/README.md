# M4 - Componente visual

Permite que una nota clínica entre como **imagen** (foto o escaneo), no solo como texto escrito.
Un VLM (`Qwen/Qwen2-VL-2B-Instruct`) transcribe la imagen a texto y ese texto entra al **mismo**
pipeline de M1/M2/M3 (chunking → NER → retrieval → generación), sin reescribir nada aguas abajo.
Patrón de integración: **input**.

> Documentación completa (ejecución, diseño, casos de prueba y tablas de detalle) en
> **[`README_detallado.md`](README_detallado.md)**.

## Qué se implementó

- Transcripción de imagen a texto con un VLM, más un backend OCR clásico de respaldo.
- Válvula de escape: si la imagen no es legible, el sistema avisa en vez de inventar texto.
- Puente al NER de M2: el texto transcrito produce el cache `{doc_id: [entidades]}` que el
  harness ya consume, sin modificarlo.
- Set de evaluación con imágenes sintéticas de 15 casos clínicos y varios niveles de degradación.
- App de chat que acepta imagen o texto (`M4/app/`).

## Lo medido (15 casos, 57 entidades, backend VLM)

El resultado central: **la foto rinde igual que el texto escrito**. Con el mismo NER sobre las
mismas notas y el mismo gold (`M4/eval_set/eval_set_casos.jsonl`):

| Entrada | Aciertos | F1 exacto | Δ vs. texto | Latencia |
|---|---|---|---|---|
| Texto gold (techo) | 29 / 57 | 0.652 | — | — |
| Imagen limpia | 29 / 57 | 0.652 | 0.000 | ~4.5 s/caso |
| Foto realista | 29 / 57 | 0.652 | 0.000 | ~4.5 s/caso |

- **Transcripción:** CER 0.000 (limpio) y 0.003 (foto realista); 0 entidades inventadas.
- **Delta cero** en las tres dimensiones (exacto, semántico y juez): leer la nota desde una
  imagen da exactamente el mismo resultado que leer el texto.
- El recall de **0.509 es del NER de M1/M2**, no de la visión: ya aparece sobre el texto gold y
  afecta por igual a texto e imagen.

Fuente: `M4/scorecard_m4.csv`, generado por `M4/evaluacion_pareada/delta_pareado.py`, que exige
que texto e imagen usen el mismo gold o se niega a escribir el resultado.

## Adversariales (robustez)

Sobre imágenes con texto tapado (sombra, recorte, perspectiva severos): **7 de 8 retos pasan**
(la lectura se abstiene o transcribe solo lo visible). El único fallo, `caso_13` con rotación y
recorte severos, agrega texto que no está en la nota. Detalle en
`M4/evaluacion_pareada/resultados/adversariales_resultado.csv` y en el README detallado (sección 4.2).

## Conclusión

El componente habilita la entrada por imagen **sin pérdida de calidad** frente al texto en el uso
realista. Su techo es el texto, y lo alcanza. El límite de F1 es del NER de M1/M2, no de la visión.

## Cómo ejecutar

Los notebooks que producen estas cifras están en `M4/ejecucion/` (se corren en Google Colab, en
orden). Ver [`README_detallado.md`](README_detallado.md) para el flujo completo y las dependencias.
