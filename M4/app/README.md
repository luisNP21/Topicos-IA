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
| `pipeline_app.py` | Carga las piezas y expone `transcribir`, `extraer_entidades`, `consultar` y `analizar` |
| `config_app.yaml` | Rutas, umbrales, backend de visión por defecto, límite de entidades |
| `requirements.txt` | Dependencias (reutiliza las de M2 y M4/vision) |
 
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
!git clone --branch appM4 https://github.com/luisNP21/Topicos-IA.git /content/corpus
%cd /content/corpus
!pip install -r M4/app/requirements.txt
# escribir /content/corpus/.env con PROJECT_ROOT=/content/drive/MyDrive/TopicosIA/Proyecto-Salud y GROQ_API_KEY
!python M4/app/app.py --share
```
 
## Decisiones
 
- **Backend de visión.** `auto` usa el VLM (Qwen2-VL-2B) si hay GPU; si no, OCR con Docling, porque
  el VLM en CPU tarda minutos por página. El usuario puede elegirlo en "Opciones".
- **`vlm_max_new_tokens: 1024`.** El config de `M4/vision` usa 512; una página A4 completa puede
  superarlo y cortar el final de la nota.
- **Umbrales.** Usa `M3/retrieval/outputs/umbrales.json` si existe; si no, 0.174 / 0.174 (calibración
  final de `M3/README.md`, sección 5). Los provisionales de `config_retrieval.yaml` (0.754 / 0.554)
  vienen del corpus de prueba y aquí no se usan.
- **Entidades consultadas.** Se consultan hasta 15 en orden de aparición (configurable en la app).
  Consultar una cuesta ~0,35 s y el LLM solo se llama si hay fragmentos sobre el umbral de evidencia.
- **Presentación para el médico.** Primero un resumen y los hallazgos con respaldo en las guías (respuesta
  y fuentes con relevancia alta/media/baja: cortes de presentación en 0.5 y 0.3 sobre el score del
  reranker, no calibrados). Los hallazgos sin cobertura van en una sola línea. Scores, normalización,
  `chunk_id` y tiempos quedan en "Detalles técnicos".
- **Varias imágenes** se tratan como páginas de la misma nota, en el orden en que se adjuntan.
- **No usa `M4/vision/integracion.sistema_multimodal`**, porque devuelve un solo texto y la app necesita
  entidades y fragmentos por separado.
- **Choque de nombres.** `contratos.py` existe en `M3/generacion` y en `M4/vision`, y `orquestacion.py`
  en `M3/generacion` y en `M3/tools`. `pipeline_app.py` importa la visión aislada para que M3 use los suyos.
## Manejo de errores
 
- Falta un recurso en `PROJECT_ROOT`: la app no arranca y lista qué falta y dónde.
- La imagen no se puede leer (válvula del VLM/OCR): se informa y no se inventan entidades.
- Ninguna entidad: se informa. Una consulta corta sin entidades (por ejemplo, "DM2") se busca tal cual.
- Una entidad sin fragmentos sobre el umbral de evidencia: se muestra la válvula de escape de M3, sin
  llamar al LLM.
- Error del LLM en una entidad: se muestra el error en esa entidad; las demás siguen.