# M4 · Set sintético de imágenes (tarea 3a)

Genera imágenes a partir de los documentos del gold set de M2, con degradaciones
reproducibles. Sirve para medir cuánto cae el desempeño del sistema cuando la nota llega como
imagen (escaneo o foto) en vez de texto limpio.

**El gold sigue siendo el texto original.** Cada imagen conserva el `doc_id` que asigna
`M2/harness/gold_loader.cargar_gold_set` (`ex_0`, `ex_1`, …), el mismo que usa el harness. Por eso
cualquier sistema que lea estas imágenes puede escribir su `predictions/<sistema>.json`
(`doc_id -> list[str]`) y evaluarse sin tocar el harness.

> Las degradaciones son sintéticas. No equivalen a escaneos ni fotos reales; los casos reales
> se documentan aparte (tarea 3b).

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.yaml` | Página, fuente, niveles de degradación y sus parámetros, semilla global |
| `render.py` | Texto → páginas de imagen (corte de líneas por ancho en píxeles y paginación) |
| `degradaciones.py` | Una función por nivel; todo el azar sale de un `rng` sembrado |
| `generar_set.py` | Entry point: `run(cfg, project_root, limite, niveles) -> dict` y CLI |
| `requirements.txt` | Pillow, numpy, PyYAML, python-dotenv |

## Cómo correrlo

```bash
pip install -r M4/eval_visual/requirements.txt
python M4/eval_visual/generar_set.py                       # set completo
python M4/eval_visual/generar_set.py --limite 3            # muestra para revisar
python M4/eval_visual/generar_set.py --niveles limpio rotacion
```

`PROJECT_ROOT` (en `.env`) define dónde se escriben las salidas, igual que en M2 y M3. Si no está
definido, se usa la raíz del repo. El gold set se lee siempre del repo (`M2/examples/gold_examples.jsonl`).

## Salidas (fuera de git)

```
<PROJECT_ROOT>/M4/data/imagenes/
  manifest.json
  limpio/        ex_0_p1.png  ex_0_p2.png ...
  jpeg_bajo/     ex_0_p1.jpg ...
  rotacion/      ex_0_p1.png ...
  desenfoque/    ex_0_p1.png ...
  sombra_celular/ex_0_p1.jpg ...
```

Nombre de archivo: `<doc_id>_p<página>.<ext>`. Un documento largo ocupa varias páginas; el VLM
transcribe página por página y las transcripciones se concatenan en el orden del manifiesto.

Se paginó en vez de renderizar una sola imagen por documento porque una imagen muy alta se
reduce al tamaño de entrada del VLM y la letra pequeña se pierde (idea de los parches de ViT).

### `manifest.json`

Por documento: `doc_id`, `n_caracteres`, `sha256_texto`, `n_paginas`, `texto_por_pagina` (el
texto tal como quedó en cada página, con sus cortes de línea) y, por nivel, la lista de páginas con
`archivo`, `semilla`, `params_usados` y `tamano_px`. A nivel global guarda la semilla, la
fuente, las versiones de Pillow y numpy, la configuración de render y de cada nivel.

## Niveles

| Nivel | Qué simula | Parámetros (en `config.yaml`) |
|---|---|---|
| `limpio` | Render sin degradar. Techo visual | — |
| `jpeg_bajo` | Escaneo o captura de baja resolución | escala 0.5, JPEG calidad 20 |
| `rotacion` | Hoja torcida | ángulo 2 a 3°, signo aleatorio, relleno blanco |
| `desenfoque` | Foto fuera de foco | Gaussian blur radio 1.5 |
| `sombra_celular` | Foto con luz irregular | sombra diagonal, viñeta, ruido, JPEG calidad 75 |

Nivel pendiente: `manuscrito_sintetico` (requiere una fuente manuscrita y se decide con el equipo).

## Reproducibilidad

- Semilla global: `proyecto.seed_global` (42).
- Cada página usa una semilla derivada de `(seed_global, nivel, doc_id, página)` con SHA-256. No depende
  del orden de ejecución ni de qué niveles se pidan.
- Comprobado: dos corridas independientes con `--limite 3` produjeron imágenes y manifiesto idénticos.
- Los píxeles también dependen de la fuente y de las versiones de Pillow/numpy. Quedan registradas en el manifiesto.

## Tamaño medido (corrida completa, 59 documentos)

- 88 páginas por nivel (1 a 3 páginas por documento), 440 imágenes en total.
- 138 MB en disco y unos 65 s en un equipo local con Windows y Arial.

## Fuente

Se usa la primera fuente disponible de `render.fuentes_candidatas` (Arial en Windows, DejaVu Sans en
Linux/Colab). Debe cubrir tildes y ñ. Si ninguna existe, usa la DejaVu Sans que trae matplotlib.
La fuente usada queda en el manifiesto. Arial y DejaVu se ven distintas, así que un set generado en Colab
no será idéntico píxel a píxel al generado en Windows. Para comparar resultados entre integrantes, conviene
generar el set una sola vez y compartirlo.

## Limitaciones

- Texto limpio y regular, sin sellos, firmas, tablas ni manuscrito real.
- No se parten palabras con guion: una entidad puede quedar dividida entre dos líneas o dos páginas,
  y eso se conserva porque es un caso de interés.
- El corte de línea cambia los saltos de línea originales. Para comparar la transcripción con el gold
  conviene normalizar los espacios (tarea 5a).
