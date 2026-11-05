# Metricas de transcripcion (M4, tarea 5a)

Generado: 2026-11-01T03:26:21 · referencia principal: texto **visible** · manifiesto: `/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M4/data/imagenes/manifest.json`

CER y WER micro (errores totales / tamano total de la referencia). `--` = no aplica.

| Sistema | Nivel | Docs | CER visible | WER visible | CER completo | CER sin tildes | CER limpio (mismos docs) | Entidades leidas | Inventadas | Mal leidas |
|---|---|---|---|---|---|---|---|---|---|---|
| vlm | limpio | 15/15 | 0.000 | 0.001 | 0.000 | 0.000 | -- | 100.0 % | 0.0 % | 0.0 % |
| vlm | foto_realista | 15/15 | 0.003 | 0.004 | 0.000 | 0.003 | 0.000 | 100.0 % | 0.0 % | 0.0 % |

**Entidades leidas:** entidades del gold visibles en la imagen que aparecen tal cual en la transcripcion (no depende del NER). **Inventadas / mal leidas:** fraccion de las entidades predichas por el NER que no estan en el texto original (ver auditoria.py).

## Origen de las entidades predichas

| Sistema | Nivel | Predichas | Acierto | FP del NER | Texto extra | Mal leida | Inventada | Fuera de transcripcion |
|---|---|---|---|---|---|---|---|---|
| vlm | limpio | 32 | 29 | 3 | 0 | 0 | 0 | 0 |
| vlm | foto_realista | 32 | 29 | 3 | 0 | 0 | 0 | 0 |
