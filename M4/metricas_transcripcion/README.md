# M4 · Métricas de transcripción 

Mide qué tan bien la visión (VLM u OCR) leyó cada imagen del set de `M4/eval_visual`, y de dónde
salen las entidades que el NER extrajo de esas transcripciones. El F1 del harness dice **cuánto**
cae el sistema; este módulo dice **por qué**: lectura mala, texto inventado, texto que no estaba en
la imagen o error del NER.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.yaml` | Rutas, entradas a evaluar, normalización, umbrales de la auditoría y simulación |
| `run_metricas.py` | Entry point: `run(cfg, project_root, entradas_extra, simular) -> dict` y CLI |
| `paginas.py` | Lee los caches de `M4/vision` y **une las páginas de cada documento** |
| `cer_wer.py` | Normalización, CER y WER (rapidfuzz) |
| `auditoria.py` | Lectura de entidades del gold y origen de cada entidad predicha |
| `simular.py`, `validar_simulacion.py` | Datos simulados con errores conocidos para probar el módulo |
| `start_metricas.ipynb` | Lanzador para Colab |

## Cómo correrlo

```bash
pip install -r M4/metricas_transcripcion/requirements.txt

# Un cache de transcripciones por nivel; el cache NER es opcional y va en el mismo orden.
python M4/metricas_transcripcion/run_metricas.py \
  --transcripciones M4/vision/predictions/transcripciones_vlm_limpio.json \
  --ner M2/predictions/vision_vlm_limpio.json

# Prueba sin transcripciones reales (resultados marcados como SIMULADO)
python M4/metricas_transcripcion/run_metricas.py --simular
```

Las rutas son relativas a `PROJECT_ROOT`. Las entradas también se pueden fijar en `entradas` de
`config.yaml`. El nivel se deduce de la ruta de cada imagen (`<nivel>/<doc_id>_p<k>.ext`); si no se
puede deducir, se pasa con `--nivel`. Necesita el `manifest.json` del set de imágenes.

## Entradas

- **Transcripciones:** el cache que escribe `M4/vision/run_vision.py` (`doc_id -> Transcripcion`).
  Puede venir **por página** (id `<doc_id>__p<k>`, que es lo que sale si se usan los CSV de
  `eval_visual/casos/`) o por documento. Un cache por nivel.
- **NER (opcional):** `doc_id -> list[str]`, el formato del harness de M2. Si viene por página, se une
  por documento y se avisa en el reporte.
- **NER sobre texto (opcional):** `rutas.ner_texto` (por defecto `M2/predictions/encoder_solo.json`).
  Sirve para marcar qué falsos positivos del NER también aparecen leyendo texto limpio.

## Qué mide

### 1. CER y WER
- **CER** = distancia de edición en caracteres / caracteres de la referencia. **WER**, lo mismo en
  palabras. Pueden pasar de 1 si la transcripción trae mucho texto extra.
- Se reportan **micro** (errores totales / tamaño total, la principal) y **macro** (promedio por documento).
- **Dos referencias:**
  - `visible`: solo el texto que quedó en la imagen (`texto_visible_por_pagina` del manifiesto). Es la principal.
  - `completo`: todo el texto del documento. La diferencia entre las dos es el texto que la imagen ocultó.
- **`cer_sin_diacriticos`:** el mismo CER, quitando tildes y la virgulilla de la ñ en ambos lados. La
  diferencia con el CER es cuánto del error son tildes.
- **Operaciones** (sustituciones, borrados e inserciones): indicativas, porque hay varias alineaciones
  óptimas posibles. Muchas inserciones sugieren texto agregado.
- **`limpio_mismos_docs`:** el CER del nivel `limpio` del mismo sistema, restringido a los mismos
  documentos. Es la comparación justa para los barridos, que solo tienen 20 documentos.
- **Por dificultad de la caligrafía** en los niveles manuscritos.

**Normalización** (en `config.yaml`, igual para referencia y transcripción): Unicode NFC, minúsculas,
espacios y saltos de línea colapsados a un espacio (el render corta líneas distinto que el original).
**Las tildes no se quitan**, porque perderlas es uno de los fallos que se quieren ver. El CER conserva la
puntuación; el WER quita la puntuación de los bordes de cada palabra (`diabetes,` = `diabetes`).

**Valvula y errores:** si `M4/vision` devolvió la válvula ("No se aprecia texto legible...") o un
error en una página, esa página cuenta como transcripción vacía (todo borrado). El reporte cuenta
cuántas páginas fueron válvula, error o faltantes.

**Texto extra:** los sellos, notas al margen y tachaduras están en la imagen pero no en el gold. Si el
VLM los copia, suben las inserciones. `texto_extra_pct_de_referencia` es la cota de cuánto pueden
inflar el CER.

### 2. Lectura de entidades (no depende del NER)
De las entidades del gold que se ven en la imagen, cuántas aparecen escritas tal cual en la
transcripción (`tasa_exacta`), y cuántas con un parecido de 85 o más (`tasa_exacta_o_aproximada`). Se
excluyen las entidades `oculta` y, con `excluir_parciales: true`, también las `parcial`. Separa "la
visión perdió la entidad" de "el NER no la reconoció".

### 3. Origen de cada entidad predicha (si hay cache NER)

| Categoría | Regla | Qué significa |
|---|---|---|
| `acierto` | Está en el gold | — |
| `fp_ner` | No está en el gold, sí en el texto original | Error del NER; puede pasar también con texto limpio (`en_sistema_texto`) |
| `texto_extra` | No está en el original, sí en un sello, nota o tachadura | La visión leyó bien; no es alucinación |
| `mal_leida` | No está en el original, pero se parece (partial ratio ≥ 85) a un fragmento del original | Error de lectura (p. ej. `diabetis`) |
| `inventada` | No está en el original ni se parece a nada de él | Candidata a alucinación del VLM |
| `fuera_de_transcripcion` | Tampoco está en la transcripción | Inconsistencia entre el cache NER y el de transcripciones |

Las comparaciones son con la entidad normalizada (minúsculas, sin puntuación en los bordes) y como
cadena completa, sin cortar palabras. `mal_leida` e `inventada` se exportan a `candidatos_juez.jsonl`
con una afirmación lista para el juez visual de `M4/vision/juez.py`, que mira la imagen y decide.
Este módulo **no** llama al juez porque necesita el VLM (GPU).

## Salidas

En `<PROJECT_ROOT>/M4/outputs/metricas_transcripcion/` (fuera de git), o en `.../simulado/` con `--simular`:

| Archivo | Contenido |
|---|---|
| `resumen.json` / `resumen.md` | Métricas por (sistema, nivel), tablas de origen de entidades y por caligrafía, avisos |
| `por_documento.csv` | Una fila por documento y nivel: CER/WER, páginas con válvula o error, lectura y categorías |
| `entidades.csv` | Una fila por entidad predicha con su categoría, parecido y fragmento más cercano |
| `candidatos_juez.jsonl` | Entidades `mal_leida` e `inventada` con las imágenes del documento, para el juez visual |
| `transcripciones_por_documento/<sistema>__<nivel>.json` | **Transcripción unida por documento** |

### Unir páginas antes del NER (importante para la tarea 1 y la tarea 4)
Con los CSV de `eval_visual`, `M4/vision` escribe una transcripción **por página**. Si el NER corre así,
el cache NER sale con claves `<doc_id>__p<k>`, que no coinciden con los `doc_id` del harness, y las
entidades partidas entre dos páginas se pierden. `transcripciones_por_documento/*.json` tiene la
transcripción unida por documento con la misma forma que una `Transcripcion` de `M4/vision` (`texto`,
`valvula`, `error`, ...), así que se puede pasar tal cual a `M4/vision/ner_bridge.generar_cache_ner`
para obtener un cache NER con los `doc_id` del harness.

## Validación con datos simulados

`--simular` parte del texto visible de cada página e inyecta errores conocidos: ruido de caracteres
fuera de las entidades, pérdida de tildes, una entidad mal escrita, enfermedades inventadas, copia del
texto extra y páginas en válvula. También genera una salida de NER con aciertos, falsos positivos y
esas entidades. `validar_simulacion.py` compara lo medido con lo inyectado.

Resultado (12 documentos, 6 niveles: `limpio`, `foto_realista`, `manuscrito_foto`, `sombra_severo`,
`rotacion_recorte_severo`, `manuscrito_severo`; corrida local del 24-10-2026):
- La distancia medida fue entre el 99,6 % y el 99,9 % de las ediciones inyectadas. Es menor o igual,
  como se espera: Levenshtein encuentra la alineación mínima.
- Las 6 categorías de entidades coincidieron exactamente con lo inyectado en los 6 niveles.
- La pérdida de tildes inyectada coincidió (±3 caracteres por nivel) con la diferencia entre CER y CER
  sin diacríticos.

Estas cifras validan el módulo. **No son resultados del sistema.**

## Limitaciones
- `mal_leida` frente a `inventada` depende de un umbral de parecido (85). En entidades cortas (6
  caracteres o menos) un solo error ya baja el parecido de 85 y la entidad cae como `inventada`. Por
  eso conviene revisar esos casos con el juez visual.
- `fp_ner` se decide por presencia de la cadena en el original. Si el NER une dos palabras que en el
  original separaba texto oculto, la entidad cae como `inventada`.
- Una entidad que el VLM "corrige" (escribe bien algo que estaba mal escrito en el original) no se
  puede detectar con texto; solo con el juez visual.
- Las operaciones (sustitución, borrado e inserción) dependen de la alineación elegida; solo la
  distancia total es exacta.
