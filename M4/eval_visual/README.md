# M4 · Set sintético de imágenes (tarea 3a)

Genera imágenes a partir de los documentos del gold set de M2 para medir cuánto cae el sistema
cuando la nota llega como imagen (escaneo, foto o nota escrita a mano) en vez de texto limpio.

**El gold sigue siendo el texto original.** Cada imagen conserva el `doc_id` que asigna
`M2/harness/gold_loader.cargar_gold_set` (`ex_0`, `ex_1`, …), el mismo que usa el harness.

> Todo es sintético. Las degradaciones aproximan escaneos y fotos, y la letra a mano sale de
> fuentes tipográficas: no reemplazan notas reales. Los casos reales van en la tarea 3b.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.yaml` | Niveles, sus efectos y parámetros, caligrafías, semilla, subconjuntos |
| `generar_set.py` | Entry point: `run(cfg, project_root, limite, niveles, procesos) -> dict` y CLI |
| `render.py` | Render impreso. Guarda cada palabra con su posición en el texto original y su caja en la página |
| `manuscrito.py` | Render a mano: caligrafía por documento, variación por letra, errores de escritura, papel |
| `degradaciones.py` | Efectos de geometría, luz, óptica y sensor, y el registro de todos los efectos |
| `marcas.py` | Sello, firma, mancha de café, resaltador, perforaciones y notas al margen |
| `visibilidad.py` | Qué palabras y entidades del gold siguen visibles en la imagen final |
| `utilidades.py` | Estado de página, remapeo, homografías, semillas |
| `fonts/` | 16 caligrafías libres con su licencia (ver "Fuentes") |
| `start_eval_visual.ipynb` | Lanzador para Colab |

## Cómo correrlo

```bash
pip install -r M4/eval_visual/requirements.txt
python M4/eval_visual/generar_set.py                         # los 28 niveles
python M4/eval_visual/generar_set.py --limite 3              # muestra para revisar
python M4/eval_visual/generar_set.py --niveles limpio foto_realista manuscrito_foto
python M4/eval_visual/generar_set.py --procesos 1            # en serie (por defecto: núcleos - 1)
```

`PROJECT_ROOT` (en `.env`) define dónde se escriben las salidas, igual que en M2 y M3. Si no está
definido, se usa la raíz del repo. En Colab se usa `start_eval_visual.ipynb`.

## Niveles

Hay dos grupos. Los **base** usan los 59 documentos. Los **barridos** aíslan un factor en tres
severidades sobre un subconjunto fijo de 20 documentos (`subconjuntos.barrido`), el mismo para
todos los niveles, para poder atribuir una caída del F1 a una causa sin multiplicar las llamadas al VLM.

| Nivel | Docs | Qué simula |
|---|---|---|
| `limpio` | 59 | Render impreso sin degradar. Techo visual |
| `foto_realista` | 59 | Nota impresa fotografiada: mezcla aleatoria por página de sello, firma, resaltador, notas, mancha, perforaciones, hoja doblada, perspectiva sobre una mesa, sombra de mano, flash, tinte, poca luz, desenfoque y ruido |
| `manuscrito_foto` | 59 | Nota escrita a mano (caligrafía variable, tachaduras, palabras pegadas) y fotografiada |
| `manuscrito_{leve,moderado,severo}` | 20 | Letra prolija en papel blanco → irregular en papel rayado con tachaduras → apurada, renglones que se montan, tinta que falla, cuadriculado |
| `sombra_{…}` | 20 | Sombra de mano o celular: suave en un borde → hasta el centro → dos sombras oscuras de borde duro |
| `flash_{…}` | 20 | Reflejo del flash: débil → texto pálido en el centro → texto borrado en una zona |
| `perspectiva_{…}` | 20 | Foto casi de frente → en ángulo con hoja abombada y un pliegue → muy inclinada, doblada en cruz y parte fuera de cuadro |
| `desenfoque_{…}` | 20 | Leve en toda la hoja → media hoja fuera de foco → mano que tiembla más media hoja fuera de foco |
| `rotacion_recorte_{…}` | 20 | 3-5° → 6-9° con un borde cortado dentro del texto → 10-15° con dos bordes cortados |
| `baja_resolucion_{…}` | 20 | 70 % / JPEG 45 → 50 % / JPEG 22 → 35 % / JPEG 12 |
| `marcas_{…}` | 20 | Perforaciones y resaltador → sello, firma, resaltador y nota → sello grande en el centro, firma sobre el texto, mancha de café y dos notas |
| `orientacion` | 20 | Foto girada 90, 180 o 270° |

Los parámetros exactos están en `config.yaml`. Un parámetro puede ser un valor, un rango `[min, max]`
(se sortea por página) o una lista de opciones; `prob` hace que un efecto se aplique solo a una parte de
las páginas. Lo sorteado en cada página queda en el manifiesto.

### Manuscrito
- **Caligrafía:** 16 fuentes libres, una por documento y la misma en las tres severidades. Cada una lleva una
  etiqueta de dificultad (`facil`, `media`, `dificil`) puesta a ojo, para poder separar resultados por tipo de letra.
  El tamaño se iguala por el alto de la "x".
- **Variación:** tamaño, posición y presión de cada letra; línea base ondulada y con pendiente; inclinación por
  documento; giro de cada palabra; distorsión elástica de los trazos.
- **Errores de escritura:** palabra empezada y tachada (o tapada con un garabato) antes de la correcta, palabras
  pegadas, tinta que falla y letras repasadas.
- **Papel:** blanco, rayado (con margen rojo) o cuadriculado.

Ningún error cambia las palabras del texto. La tachadura **agrega** una palabra extra, que se registra como texto
extra; el texto original queda completo.

## Texto oculto: qué se pudo leer de verdad

Algunos efectos sacan texto de la imagen: el recorte, la rotación fuerte, la hoja fuera de cuadro, el flash que
satura, la sombra casi negra o la firma encima. Si no se registra, el harness castigaría al VLM por palabras que
no estaban en la imagen.

Por eso cada palabra lleva 15 puntos de muestra que siguen todos los efectos geométricos (se comprobó con una
prueba: los puntos caen sobre la marca correspondiente después de rotar, recortar, orientar, reescalar, poner en
perspectiva y deformar el papel). Una palabra cuenta como **visible** si sobrevive al menos el 50 % de sus puntos
(`visibilidad.umbral_token`).

Con eso, por documento y nivel, el manifiesto trae:
- `texto_visible_por_pagina`: el texto que realmente quedó en la imagen.
- `palabras.ocultas` y, para cada entidad del gold, si quedó `visible`, `parcial`, `oculta` o `no_localizada`
  (la cadena no aparece tal cual en el texto).

La legibilidad (desenfoque, baja resolución, letra difícil, sombra moderada) **no** cuenta como texto oculto: eso
es lo que se mide. Los umbrales de "ilegible" del flash (≥ 0,9 de saturación) y de la sombra (< 20 % de luz) son
una decisión de diseño documentada en `config.yaml`.

## Salidas (fuera de git)

```
<PROJECT_ROOT>/M4/data/imagenes/
  manifest.json
  casos/<nivel>.csv                       # un CSV por nivel, formato de M4/vision
  <nivel>/<doc_id>_p<k>.png|jpg
```

### `manifest.json`
- **Global:** semilla, fuentes, versiones, `visibilidad`, `subconjuntos` (qué documentos tiene cada uno),
  `niveles` (resumen por nivel: documentos, páginas, % de palabras ocultas, conteo de entidades por estado),
  `niveles_config` y `render`.
- **Por documento y nivel** (`documentos[i].niveles[<nivel>]`): `n_paginas`, `texto_por_pagina`,
  `texto_visible_por_pagina`, `palabras`, `entidades` (con `no_visibles`), `estilo_manuscrito` (si aplica) y `paginas`.
- **Por página:** `archivo`, `tamano_px`, `formato`, `calidad_jpeg`, `efectos` (tipo, si se aplicó, semilla y
  parámetros usados) y `texto_extra` (sello, notas al margen y tachaduras: texto que está en la imagen pero no en el gold).

### `casos/<nivel>.csv` (para `M4/vision`)
Columnas `id, doc_id, pagina, n_paginas, imagen, nivel, familia, severidad, tipo`. `id` es `<doc_id>__p<k>` porque
el manifiesto de visión exige un id único por imagen. Se usa con `rutas.imagenes_dir: M4/data/imagenes`:

```bash
python M4/vision/run_vision.py --config M4/vision/config.yaml --manifest M4/data/imagenes/casos/limpio.csv
```

## Cómo conectarlo con el harness (para tareas 1, 4 y 5a)
1. **Varias páginas por documento.** Hay que unir las transcripciones de un `doc_id` en orden de página antes del
   NER (una entidad puede quedar partida entre dos páginas). El resultado es un cache `doc_id -> list[str]`, como el de M2.
2. **Subconjuntos.** Un nivel de barrido solo tiene 20 documentos. `cached_system.sistema_desde_cache` exige que el
   cache y el eval set tengan los mismos `doc_id`, así que hay que filtrar el eval set con
   `manifest["subconjuntos"]["barrido"]`. Para comparar contra el techo, usar `limpio` restringido a esos mismos 20.
3. **Texto oculto.** Para un F1 "sobre lo visible", excluir del gold las entidades `oculta` (y decidir qué hacer con
   las `parcial`). Conviene reportar ambos: F1 contra todo el gold y F1 contra lo visible.
4. **Texto extra.** Para CER/WER, `texto_extra` dice qué texto de la imagen no está en el gold (sellos, notas,
   tachaduras): el VLM puede transcribirlo sin que sea una alucinación.

## Reproducibilidad
- Semilla global: `proyecto.seed_global` (42). Cada sorteo usa una semilla derivada con SHA-256 de
  (semilla, documento, página, efecto). No depende del orden de ejecución, de qué niveles se pidan ni de cuántos
  procesos se usen.
- Los sorteos de un efecto no dependen del nombre del nivel: el mismo documento recibe la misma sombra (posición,
  forma) en `sombra_leve`, `_moderado` y `_severo`, y solo cambia la intensidad. Lo mismo con la caligrafía.
- Comprobado: dos corridas independientes (`--limite 2`, 4 niveles) dieron 27 de 27 archivos idénticos byte por
  byte y el mismo manifiesto. Una corrida en serie y otra con 3 procesos (`--limite 4`, 5 niveles) dieron 61 de
  61 archivos idénticos y el mismo manifiesto.
- Los píxeles también dependen de la fuente impresa (Arial en Windows, DejaVu Sans en Colab) y de las versiones de
  Pillow y numpy, que quedan en el manifiesto. Conviene generar el set una vez y compartirlo.

## Cifras medidas (corrida completa)

Corrida local: Windows, Arial, Python 3.13, 11 procesos en un equipo de 12 hilos.

- **1.142 imágenes, 363 MB, 214 s.** La suma del tiempo de proceso de todos los niveles fue 2.234 s; en serie
  (por ejemplo, Colab con 2 núcleos usa 1 proceso) debería tardar del orden de media hora o más. Esa cifra es una
  estimación, no se midió. Si el equipo tiene varios núcleos, conviene generar el set en local y subirlo a Drive.
- **Páginas:** impreso 88 páginas en los 59 documentos (31 en los 20 del barrido); `manuscrito_foto` 141; los
  barridos manuscritos 46, 47 y 50 (la letra ocupa más).
- **Entidades:** 758 entidades distintas del gold en los 59 documentos (284 en el barrido). Todas se localizaron en
  el texto (`no_localizada` = 0).

Niveles donde quedó texto oculto (en el resto, 0 palabras ocultas y todas las entidades visibles):

| Nivel | Palabras ocultas | Entidades visibles / parciales / ocultas |
|---|---|---|
| `sombra_severo` | 29,6 % | 164 / 68 / 52 de 284 |
| `rotacion_recorte_severo` | 13,2 % | 213 / 50 / 21 de 284 |
| `rotacion_recorte_moderado` | 2,1 % | 272 / 10 / 2 de 284 |
| `flash_severo` | 1,2 % | 280 / 4 / 0 de 284 |
| `marcas_moderado` | 0,3 % | 282 / 2 / 0 de 284 |
| `marcas_severo` | 0,3 % | 279 / 5 / 0 de 284 |
| `foto_realista` | 0,1 % | 757 / 1 / 0 de 758 |
| `perspectiva_severo` | 0,1 % | 284 / 0 / 0 de 284 |

En `sombra_severo` casi un tercio del texto queda bajo sombras con menos del 20 % de luz. Si el VLM falla ahí,
la causa principal es que el texto no se ve, y así hay que reportarlo.

## Fuentes
- **Impresa:** la primera de `render.fuentes_candidatas` que exista (Arial en Windows, DejaVu Sans en Linux/Colab).
- **Manuscritas** (`fonts/<familia>/`, unos 2 MB): Caveat, Kalam, Patrick Hand, Indie Flower, Architects Daughter,
  Handlee, Shadows Into Light, Reenie Beanie, Kristi, Homemade Apple, Nothing You Could Do, Dancing Script,
  Dawning of a New Day, Cedarville Cursive, La Belle Aurore y Zeyada. Todas de
  [google/fonts](https://github.com/google/fonts): licencia SIL OFL 1.1, salvo Homemade Apple (Apache 2.0). Cada
  carpeta trae su archivo de licencia. Se verificó que todas tienen tildes y ñ; los pocos símbolos que les faltan
  (α, β, μ, χ) se dibujan con la fuente impresa.
- El sello usa nombres ficticios ("HOSPITAL CLINICO SIMULADO", "CLINICA DE PRUEBA M4", "IPS SINTETICA").

## Limitaciones
- Letra a mano a partir de fuentes: más regular que la de una persona, sin abreviaturas propias ni formas de letra
  ambiguas reales. Mide la dificultad de la cursiva, no la de una letra médica real.
- Sombras, flash y perspectiva son modelos simples de iluminación y cámara, sin reflejos especulares del papel ni
  aberraciones del lente.
- No hay tablas de laboratorio, formularios ni texto en columnas: el texto de DisTEMIST es corrido.
- No se parten palabras con guion; una entidad puede quedar dividida entre líneas o páginas, a propósito.
- El corte de línea cambia los saltos de línea originales. Para CER/WER hay que normalizar los espacios.
- La etiqueta de dificultad de cada caligrafía es una apreciación a ojo, no una medida.
