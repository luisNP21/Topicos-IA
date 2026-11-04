# M4 · App de punto de entrada
 
Chat en Gradio: el usuario escribe un caso clínico o adjunta la foto de una nota clínica, y recibe
las **entidades** detectadas y los **fragmentos de guía** que las respaldan, con la respuesta
redactada por el LLM de M3.
 
```
imagen ──► M4/vision (VLM u OCR) ──┐
                                   ├─► texto ─► NER (M1/M2) ─► por entidad: orquestación + retrieval (M3)
texto ─────────────────────────────┘                              └─► fragmentos ─► generación Groq (M3)
```
 
No reimplementa ninguna pieza: llama al código del equipo (`M4/vision`, `M2/run_inference`,
`M3/retrieval`, `M3/generacion`).
 
## Archivos
 
| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz Gradio (chat multimodal) y arranque |
| `pipeline_app.py` | Carga las piezas y expone `transcribir`, `extraer_entidades`, `buscar`, `elegir_principal`, `redactar` y `analizar` |
| `evaluar_principal.py` | Compara la enfermedad principal que elige la app con la `entidad` del eval set de M3 |
| `config_app.yaml` | Rutas, umbrales, backend de visión por defecto, límite de entidades |
| `operacion.py` | Versión de Git, logs por consulta, health check y generador del `requirements.txt` fijado |
| `requirements.txt` | Dependencias con versiones fijadas (autocontenido; `torch` no se fija) |
 
## Qué necesita
 
En el `.env` de la raíz del repo:
 
```
PROJECT_ROOT=<ruta a la carpeta Proyecto-Salud>
GROQ_API_KEY=<tu clave>
```
 
Dentro de `PROJECT_ROOT`, con la misma estructura que en Drive:
 
```
M1/saved_models/clinical_bert-distemist-lora/     # adaptador LoRA del NER
M3/data/guias_clinicas/chunks/*.json              # chunks del corpus (salida de la ingesta)
M3/data/guias_clinicas/corpus_manifest.json       # manifiesto del corpus
```
 
Sin `GROQ_API_KEY` la app funciona, pero solo muestra entidades y fragmentos, sin respuesta redactada.
 
## Cómo ejecutarla
 
Desde la raíz del repo, con Python 3.10 o superior:
 
```bash
pip install -r M4/app/requirements.txt
python M4/app/app.py            # abre http://127.0.0.1:7860
```
 
La primera vez descarga los modelos (embeddings e5-base, reranker, Clinical BERT) y reconstruye el
índice Chroma en `M4/app/.cache/` (no se versiona). Las siguientes veces reutiliza ese índice.
 
**Con GPU (Colab T4)**, para usar el VLM y obtener un enlace público temporal:
 
```python
!git clone --branch dev https://github.com/luisNP21/Topicos-IA.git /content/corpus
%cd /content/corpus
!pip install -r M4/app/requirements.txt
# escribir /content/corpus/.env con PROJECT_ROOT=/content/drive/MyDrive/TopicosIA/Proyecto-Salud y GROQ_API_KEY
!python M4/app/app.py --share
```

## Operación: versión, health check, logs y dependencias

**Versión visible.** El pie de la descripción de la app y la primera línea de la consola muestran
`git describe` (el tag si lo hay, si no el commit), la rama y la fecha del commit. `-dirty` significa
que lo que corre tiene cambios sin commit. Para la demo, etiquetar el commit que se presenta:

```bash
git tag -a v1.0-demo -m "Demo M4"
git push origin v1.0-demo
```

Sin carpeta `.git` (p. ej. un Space que copia solo los archivos) se usa la variable `APP_VERSION`.

**Estado del sistema (en la app).** Un acordeón al pie, pensado para el usuario y sin detalle
técnico: muestra solo el **estado** (funcionando / funcionando con limitaciones / con fallas) y la
**latencia** (última consulta y promedio de la sesión). Se llena al abrir la página y con el botón *Actualizar*.

**Health check técnico.** El JSON completo (`estado`: `ok` / `degradado` / `error`, con cada chequeo)
no se muestra en la interfaz; queda para monitoreo:

- Por API, también sobre el enlace `gradio.live`:
  `from gradio_client import Client; Client("<url>").predict(api_name="/health")`.
- Antes de lanzar: `python M4/app/app.py --health` carga todo, imprime el JSON y sale con código 1 si
  hay error.

Revisa el índice de guías y el NER (críticos), el motor de visión, el LLM y la GPU (si faltan, la app
funciona con limitaciones: `degradado`), y hace una búsqueda de prueba real
(`diabetes mellitus tipo 2`, ~0,4 s, sin LLM). Tiene su propia cola, así que responde aunque haya una
consulta en curso. También informa cuántas consultas van, cuántas con error y la última.

**Logs por consulta.** Cada consulta agrega una línea a `M4/app/.cache/logs/consultas.jsonl` (no se
versiona) y un resumen en la consola:

| Campo | Contenido |
|---|---|
| `entrada` | tipo (`texto` / `imagen` / `imagen+texto`), tamaño y hash del texto y de cada imagen, motor pedido |
| `ruta` | motor de visión y estado por página, entidades detectadas/consultadas/con evidencia, enfermedad principal, si se normalizó, `chunk_id` usados |
| `confianza` | score del reranker de la principal vs. `umbral_evidencia`, nº de fragmentos, fallback, error del LLM |
| `latencia_s` | `vision`, `ner`, `busqueda`, `redaccion` y `total` |
| `resultado` | `respuesta`, `solo_fragmentos`, `sin_cobertura`, `sin_entidades`, `imagen_ilegible` o `error` |

El texto de la nota **no** se guarda, solo su tamaño y hash (`logs.guardar_texto` en `config_app.yaml`).
Si una consulta falla, igual queda registrada con el error, y la app muestra su `id` para buscarla.

**Dependencias fijadas.** `requirements.txt` tiene `==` en todo lo que el repo ya tenía fijado. Las que
no (`chromadb`, `docling`, `groq`, `qwen-vl-utils`, ...) se fijan generando el archivo en el entorno
donde la app funcionó de punta a punta:

```bash
python M4/app/operacion.py --requirements > M4/app/requirements.txt
```
 
## Capturas
 
**Caso escrito con cobertura en las guías.** La app identifica diabetes mellitus tipo 2 e
insuficiencia cardíaca y responde con base en las guías, mostrando sus fuentes.
 
![caso cubierto](https://github.com/user-attachments/assets/940eff92-6cd0-4077-b407-bef03de2e262)
 
**Caso desde una imagen, sin cobertura.** La nota fotografiada es de TFNA. La app
la lee con el VLM y detecta 4 hallazgos. Como ninguno pertenece a las guías del corpus, no responde
en vez de inventar.
 
![caso no cubierto](https://github.com/user-attachments/assets/6172cdd4-750e-4785-9c8a-04ffd97ab2e8)
 
## Decisiones
 
- **Backend de visión.** `auto` usa el VLM (Qwen2-VL-2B) si hay GPU; si no, OCR con Docling, porque
  el VLM en CPU tarda minutos por página. La interfaz no ofrece elegir motor ni el máximo de
  hallazgos: siempre usa `auto` y `rag.max_entidades` de `config_app.yaml`.
- **Precarga del motor de visión.** El backend por defecto se carga al iniciar la app, no con la
  primera imagen; así el tiempo de la página es solo lectura (~30 s con el VLM en una T4). Si la
  precarga falla, la app arranca igual y lo carga con la primera imagen.
- **Aviso «lectura automática, verifique el original».** Aparece en toda respuesta que viene de una
  imagen, porque los hallazgos dependen de lo que el modelo leyó.
- **`vlm_max_new_tokens: 1024`.** El config de `M4/vision` usa 512; una página A4 completa puede
  superarlo y cortar el final de la nota.
- **`vlm_max_pixels: 1003520` (1280×28×28).** Qwen2-VL usa ~1 token por parche de 28×28 px; una
  foto de celular sin límite son miles de tokens de imagen (más lento y más VRAM). Con este límite la
  imagen se reescala, manteniendo la proporción, a ~1000×1000 px como máximo. Si una nota con letra
  pequeña se lee mal, subirlo (p. ej. `1605632` = 2048×28×28) o poner `null` (sin límite, como en
  las métricas de `M4/vision`).
- **Válvula ampliada.** Además de respuestas vacías o que empiezan con «No se aprecia…», ahora se
  detecta una respuesta que entera es solo un comentario («Se aprecia texto legible»): antes esa
  frase pasaba como si fuera la nota. Detalle en `M4/vision/README.md`.
- **Umbrales.** Usa `M3/retrieval/outputs/umbrales.json` si existe; si no, 0.174 / 0.174 (calibración
  final de `M3/README.md`, sección 5). Los provisionales de `config_retrieval.yaml` (0.754 / 0.554)
  vienen del corpus de prueba y aquí no se usan.
- **Una enfermedad principal por caso.** El NER detecta todas las menciones de enfermedad (así se
  entrenó y se evalúa en M2), pero la respuesta se redacta para una sola, como en el eval set de M3.
  Se buscan hasta 15 entidades en las guías y se elige la que tiene el
  mayor score del reranker entre las que pasan la compuerta de evidencia. Es un criterio
  determinista, como la orquestación de M3: elige la enfermedad sobre la que las guías pueden
  responder mejor, no un diagnóstico clínico. Las demás entidades se muestran plegadas. Si ninguna
  pasa la compuerta, no hay principal y la app no responde. `evaluar_principal.py` mide cuántas
  veces coincide con la `entidad` de los 15 casos de `M3/ejecucion/eval_set_casos.json`.
- **Pregunta del usuario.** Si el mensaje trae una pregunta entre `¿` y `?`, la respuesta se genera
  sobre esa pregunta (como la columna `pregunta` del eval set de M3). Si no, se usa la pregunta de
  intención del retrieval ("¿Cuál es el tratamiento de …?").
- **Presentación para el médico.** Primero un resumen con la enfermedad principal, su respuesta y sus
  fuentes con relevancia alta/media/baja (cortes de presentación en 0.5 y 0.3 sobre el score del
  reranker, no calibrados). Los otros hallazgos quedan plegados. Scores, normalización, `chunk_id` y
  tiempos quedan en "Detalles técnicos".
- **Varias imágenes** se tratan como páginas de la misma nota, en el orden en que se adjuntan.
- **No usa `M4/vision/integracion.sistema_multimodal`**, porque devuelve un solo texto y la app necesita
  entidades y fragmentos por separado.
## Manejo de errores
 
- Falta un recurso en `PROJECT_ROOT`: la app no arranca y lista qué falta y dónde.
- La imagen no se puede leer (válvula del VLM/OCR): se informa y no se inventan entidades.
- Ninguna entidad: se informa. Una consulta corta sin entidades (por ejemplo, "DM2") se busca tal cual.
- Una entidad sin fragmentos sobre el umbral de evidencia: se muestra la válvula de escape de M3, sin
  llamar al LLM.
- Error del LLM en una entidad: se muestra el error en esa entidad; las demás siguen.
- Error inesperado en una consulta: se muestra el error con el `id` de la consulta y queda en el log.
 
