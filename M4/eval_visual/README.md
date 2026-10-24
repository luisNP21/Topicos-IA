# M4 · Set sintético de imágenes

Genera imágenes a partir de los documentos del gold set de M2, con degradaciones
reproducibles. Sirve para medir cuánto cae el desempeño del sistema cuando la nota llega como
imagen (escaneo o foto) en vez de texto limpio.

**El gold sigue siendo el texto original.** Cada imagen conserva el `doc_id` que asigna
`M2/harness/gold_loader.cargar_gold_set` (`ex_0`, `ex_1`, …), el mismo que usa el harness. Por eso
cualquier sistema que lea estas imágenes puede escribir su `predictions/<sistema>.json`
(`doc_id -> list[str]`) y evaluarse sin tocar el harness.

> Las degradaciones son sintéticas. No equivalen a escaneos ni fotos reales; los casos reales


## Archivos

| Archivo | Qué hace |
|---|---|
| `config.yaml` | Página, fuente, niveles de degradación y sus parámetros, semilla global |
| `render.py` | Texto → páginas de imagen (corte de líneas por ancho en píxeles y paginación) |
| `degradaciones.py` | Una función por nivel; todo el azar sale de un `rng` sembrado |
| `generar_set.py` | Entry point: `run(cfg, project_root, limite, niveles) -> dict` y CLI |
| `fonts/Caveat.ttf`, `fonts/OFL.txt` | Fuente manuscrita (SIL Open Font License 1.1) y su licencia |
| `start_eval_visual.ipynb` | Lanzador para Colab |
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
En Colab se usa `start_eval_visual.ipynb`.

## Salidas (fuera de git)

```
<PROJECT_ROOT>/M4/data/imagenes/
  manifest.json
  limpio/ jpeg_bajo/ rotacion/ desenfoque/ manuscrito_sintetico/ sombra_celular/
    <doc_id>_p<k>.png|jpg      (p.ej. ex_0_p1.png)
```

Un documento largo ocupa varias páginas; el VLM transcribe página por página y las
transcripciones se concatenan en el orden `p1, p2, …`.

Se paginó en vez de renderizar una sola imagen por documento porque una imagen muy alta se
reduce al tamaño de entrada del VLM y la letra pequeña se pierde (idea de los parches de ViT).

### `manifest.json`

- **Global:** `descripcion`, `gold_set`, `seed_global`, `fuentes` (la común y la del nivel manuscrito),
  `versiones` (Pillow, numpy), `render`, `niveles_config`, `n_documentos`, `n_imagenes`.
- **Por documento:** `doc_id`, `n_caracteres`, `sha256_texto` y `niveles`.
- **Por nivel** (`documentos[i].niveles[<nivel>]`): `n_paginas`, `texto_por_pagina` y `paginas`.
  `texto_por_pagina` va por nivel porque el manuscrito usa otra fuente y otro tamaño, así que su
  paginación es distinta a la de los demás niveles.
- **Por página** (`paginas[k]`): `pagina`, `archivo` (relativo, p.ej. `limpio/ex_0_p1.png`),
  `semilla`, `params_usados`, `tamano_px`.

## Niveles

| Nivel | Qué simula | Parámetros (en `config.yaml`) |
|---|---|---|
| `limpio` | Render sin degradar. Techo visual | — |
| `jpeg_bajo` | Escaneo o captura de baja resolución | escala 0.5, JPEG calidad 20 |
| `rotacion` | Hoja torcida | ángulo 2 a 3°, signo aleatorio, relleno blanco |
| `desenfoque` | Foto fuera de foco | Gaussian blur radio 1.5 |
| `manuscrito_sintetico` | Nota escrita a mano (aproximación) | fuente Caveat 44 px, tinta azul, irregularidad de ±3 px por palabra |
| `sombra_celular` | Foto con luz irregular | sombra diagonal, viñeta, ruido, JPEG calidad 75 |

### Nivel `manuscrito_sintetico`
Usa su propio perfil de render (sección `render` del nivel en `config.yaml`): fuente Caveat,
tinta azul oscuro y cada palabra se dibuja con un pequeño desplazamiento vertical y horizontal
aleatorio y reproducible. Con letra más grande el mismo documento ocupa más páginas (125 frente a 88).

**Limitaciones de este nivel:** es una fuente, no letra de una persona. Cada letra sale casi igual,
sin variación de trazo ni de inclinación, ni tachaduras ni correcciones. Mide la dificultad de una
tipografía cursiva, no la de una letra real, que es más difícil. La letra real se cubre en la tarea 3b.
Un ejemplo visible en la primera página de `ex_0`: el "7" de "37 años" se parece a un signo de
interrogación, lo que muestra cómo las cifras son vulnerables.

## Reproducibilidad

- Semilla global: `proyecto.seed_global` (42).
- Cada página usa una semilla derivada de `(seed_global, nivel, doc_id, página)` con SHA-256. No depende
  del orden de ejecución ni de qué niveles se pidan. El nivel manuscrito usa una semilla adicional para la
  irregularidad de la escritura.
- Comprobado: dos corridas independientes con `--limite 3` produjeron imágenes y manifiesto idénticos,
  y dos corridas del nivel manuscrito produjeron las mismas 10 páginas.
- Los píxeles también dependen de la fuente común y de las versiones de Pillow/numpy. Quedan registradas
  en el manifiesto.

## Tamaño medido (corrida completa, 59 documentos, 6 niveles)

| Nivel | Páginas | Por documento |
|---|---|---|
| `limpio`, `jpeg_bajo`, `rotacion`, `desenfoque`, `sombra_celular` | 88 cada uno | 1 a 3 |
| `manuscrito_sintetico` | 125 | 1 a 4 |

- Total: **565 imágenes, 172 MB**, unos 90 s en un equipo local con Windows y Arial.

## Fuentes

- **Común:** primera disponible de `render.fuentes_candidatas` (Arial en Windows, DejaVu Sans en
  Linux/Colab). Debe cubrir tildes y ñ. Si ninguna existe, usa la DejaVu Sans de matplotlib. Arial y
  DejaVu se ven distintas, así que un set generado en Colab no será idéntico píxel a píxel al generado
  en Windows. Para comparar resultados entre integrantes, conviene generar el set una sola vez y
  compartirlo.
- **Manuscrito:** `fonts/Caveat.ttf`, incluida en el repo para que sea igual en todos los equipos.
  Caveat es de Impallari Type, licencia SIL OFL 1.1 (texto en `fonts/OFL.txt`). Se descargó de
  https://github.com/google/fonts/tree/main/ofl/caveat.

## Limitaciones

- Texto limpio y regular, sin sellos, firmas ni tablas.
- No se parten palabras con guion: una entidad puede quedar dividida entre dos líneas o dos páginas,
  y eso se conserva porque es un caso de interés.
- El corte de línea cambia los saltos de línea originales. Para comparar la transcripción con el gold
  conviene normalizar los espacios (tarea 5a).
