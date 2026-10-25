"""
App de punto de entrada de M4: chat donde el usuario escribe un caso clínico o
adjunta la imagen de una nota, y recibe las entidades detectadas y los fragmentos de las
guías clínicas que las respaldan, con la respuesta redactada por el LLM de M3.

Ejecutar desde la raíz del repo:
    python M4/app/app.py                # http://127.0.0.1:7860
    python M4/app/app.py --share        # enlace público temporal (útil en Colab con GPU)
"""

from __future__ import annotations
 
import argparse
import html
from pathlib import Path
 
import gradio as gr
 
from pipeline_app import PipelineClinico
 
EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
BACKENDS = [
    ("Automático (VLM con GPU, OCR sin GPU)", "auto"),
    ("VLM Qwen2-VL (requiere GPU)", "vlm"),
    ("OCR Docling (CPU)", "ocr_docling"),
    ("OCR Tesseract (CPU, requiere binario)", "ocr_tesseract"),
]
 
AVISO = (
    "**Demostración académica (Tópicos IA, EAFIT).** No reemplaza el criterio de un profesional "
    "de la salud. No ingreses datos reales de pacientes: el texto se envía a un servicio externo "
    "(Groq) para redactar la respuesta. Las guías disponibles son las del corpus del proyecto."
)
 
EJEMPLO_TEXTO = (
    "Mujer de 64 años con diabetes mellitus tipo 2 de larga evolución e insuficiencia cardíaca "
    "con fracción de eyección reducida, que consulta por disnea progresiva y edemas en miembros "
    "inferiores. Refiere además ánimo deprimido en los últimos meses."
)
 
 
# ---------------------------------------------------------------------------
# Presentación (pensada para un médico: primero lo útil, lo técnico plegado al final)
# ---------------------------------------------------------------------------
def _recortar(texto: str, n: int = 700) -> str:
    texto = " ".join((texto or "").split())
    return texto if len(texto) <= n else texto[:n].rsplit(" ", 1)[0] + "…"
 
 
def _capitalizar(texto: str) -> str:
    return texto[:1].upper() + texto[1:] if texto else texto
 
 
def _relevancia(score) -> str:
    """Etiqueta legible del score del reranker (probabilidad en [0, 1]).
 
    Los cortes 0.5 y 0.3 son de presentación, no están calibrados. Solo llegan aquí
    fragmentos que ya superaron el umbral de evidencia de M3."""
    if not isinstance(score, (int, float)):
        return ""
    if score >= 0.5:
        return "relevancia alta"
    if score >= 0.3:
        return "relevancia media"
    return "relevancia baja"
 
 
def _fuente(f: dict) -> str:
    seccion = f" — {f['seccion']}" if f.get("seccion") else ""
    return html.escape(f"{f['fuente']}{seccion}")
 
 
def _bloque_cubierto(r: dict) -> str:
    partes = [f"### {html.escape(_capitalizar(r['entidad']))}"]
    if r["error"]:
        partes.append(f"No se pudo redactar la respuesta ({html.escape(r['error'])}). "
                      "Abajo están los fragmentos de las guías.")
    elif r["respuesta"] is not None and not r["fallback"]:
        partes.append(f"**Qué dicen las guías:** {r['respuesta']}")
    elif r["respuesta"] is not None and r["fallback"]:
        partes.append("Las guías tienen fragmentos relacionados, pero no alcanzan para responder "
                      "sobre el tratamiento. Revisa las fuentes.")
    partes.append(f"**Fuentes ({len(r['fragmentos'])}):**")
    for f in r["fragmentos"]:
        etiqueta = _relevancia(f.get("score"))
        partes.append(
            f"<details><summary>{_fuente(f)}{' · ' + etiqueta if etiqueta else ''}</summary>\n\n"
            f"{html.escape(_recortar(f['texto']))}\n\n</details>"
        )
    return "\n\n".join(partes)
 
 
def _bloque_tecnico(transcripcion: dict | None, resultados: list[dict]) -> str:
    lineas = []
    if transcripcion:
        lineas.append(f"**Lectura de la imagen** — motor `{transcripcion['backend']}`")
        for p in transcripcion["paginas"]:
            estado = "error" if p["error"] else ("sin texto legible" if p["valvula"] else "ok")
            lineas.append(f"- Página {p['doc_id'][1:]}: {estado}, {p['segundos']} s"
                          + (f" — {html.escape(p['error'])}" if p["error"] else ""))
    if resultados:
        lineas.append("\n**Consulta por hallazgo**")
        for r in resultados:
            scores = ", ".join(f"{f['score']:.3f}" for f in r["fragmentos"]
                               if isinstance(f.get("score"), (int, float))) or "ninguno sobre el umbral"
            buscado = f" (buscado como «{html.escape(r['query_final'])}»)" if r["query_final"] != r["entidad"] else ""
            linea = f"- `{html.escape(r['entidad'])}`{buscado}: scores {scores}; {r['segundos']} s"
            if r["tool_invoked"]:
                linea += f"<br><sub>{html.escape(r['tool_reason'])}</sub>"
            ids = ", ".join(str(f["chunk_id"]) for f in r["fragmentos"])
            if ids:
                linea += f"<br><sub>chunk_id: {html.escape(ids)}</sub>"
            lineas.append(linea)
    if not lineas:
        return ""
    lineas.append("\n<sub>El score es la probabilidad que asigna el reranker a que el fragmento "
                  "responda «¿Cuál es el tratamiento de …?». Solo se muestran fragmentos por encima "
                  "del umbral de evidencia calibrado en M3.</sub>")
    return "<details><summary>Detalles técnicos</summary>\n\n" + "\n".join(lineas) + "\n\n</details>"
 
 
def armar_respuesta(transcripcion: dict | None, entidades: list[str], origen: str,
                    resultados: list[dict], no_consultadas: list[str], guias: list[str]) -> str:
    cubiertos = [r for r in resultados if r["fragmentos"]]
    sin_cobertura = [r for r in resultados if not r["fragmentos"]]
    partes = []
 
    n = len(entidades)
    resumen = f"**Resumen:** se identificaron {n} hallazgo{'s' if n != 1 else ''} clínico{'s' if n != 1 else ''}. "
    if not cubiertos:
        resumen += "Ninguno tiene recomendaciones en las guías disponibles"
    else:
        resumen += f"{len(cubiertos)} con recomendaciones en las guías disponibles"
        if sin_cobertura:
            resumen += f", {len(sin_cobertura)} sin cobertura"
    if no_consultadas:
        resumen += f" ({len(no_consultadas)} sin consultar por el límite de hallazgos)"
    resumen += "."
    if origen == "consulta_directa":
        resumen += "\n\n<sub>No se detectaron hallazgos en el texto; se buscó lo escrito tal cual.</sub>"
    partes.append(resumen)
 
    if transcripcion and transcripcion["texto"]:
        partes.append("<details><summary>Ver el texto leído de la imagen</summary>\n\n"
                      f"{html.escape(transcripcion['texto'])}\n\n</details>")
 
    for r in cubiertos:
        partes.append("---\n\n" + _bloque_cubierto(r))
 
    if sin_cobertura:
        lista = ", ".join(html.escape(_capitalizar(r["entidad"])) for r in sin_cobertura)
        partes.append("---\n\n**Sin cobertura en las guías disponibles:** " + lista
                      + "\n\n<sub>Las guías disponibles son: " + html.escape("; ".join(guias))
                      + ". Si un hallazgo no pertenece a esos temas, el sistema no responde "
                        "para no inventar información.</sub>")
    if no_consultadas:
        lista = ", ".join(html.escape(_capitalizar(e)) for e in no_consultadas)
        partes.append(f"<sub>No consultados (límite de hallazgos por consulta): {lista}</sub>")
 
    tecnico = _bloque_tecnico(transcripcion, resultados)
    if tecnico:
        partes.append("---\n\n" + tecnico)
    return "\n\n".join(partes)
 
 
# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
def construir_app(pipeline: PipelineClinico) -> gr.ChatInterface:
    info = pipeline.info
 
    def responder(mensaje: dict, historial, backend: str, max_entidades: int):
        texto = (mensaje.get("text") or "").strip()
        archivos = mensaje.get("files") or []
        imagenes = [f for f in archivos if Path(f).suffix.lower() in EXT_IMAGEN]
        ignorados = [Path(f).name for f in archivos if f not in imagenes]
 
        if not texto and not imagenes:
            yield "Escribe un caso clínico o adjunta la imagen de una nota clínica (PNG o JPG)."
            return
 
        aviso_archivos = (f"<sub>Archivos ignorados (solo se aceptan imágenes): {', '.join(ignorados)}</sub>\n\n"
                          if ignorados else "")
 
        def estado(msg: str) -> str:
            return f"{aviso_archivos}_{msg}_"
 
        transcripcion = None
        if imagenes:
            nombre = pipeline.vision.resolver_backend(backend)
            espera = " Puede tardar alrededor de un minuto por página." if nombre == "vlm" else ""
            yield estado(f"Leyendo {'la imagen' if len(imagenes) == 1 else f'{len(imagenes)} imágenes'}...{espera}")
            transcripcion = pipeline.transcribir(imagenes, backend)
            if not transcripcion["texto"] and not texto:
                yield (aviso_archivos + "No se pudo leer texto en la imagen. Prueba con una foto más nítida, "
                       "de frente y con buena luz, o escribe el caso.\n\n" + _bloque_tecnico(transcripcion, []))
                return
 
        texto_total = "\n".join(p for p in [transcripcion["texto"] if transcripcion else "", texto] if p)
        yield estado("Identificando hallazgos clínicos...")
        entidades, origen = pipeline.extraer_entidades(texto_total)
        if not entidades:
            yield (aviso_archivos + "No se identificaron hallazgos clínicos (enfermedades o diagnósticos) "
                   "en el texto, así que no hay nada que buscar en las guías.")
            return
 
        n = int(max_entidades)
        resultados = []
        for i, entidad in enumerate(entidades[:n], 1):
            yield estado(f"Buscando en las guías: {_capitalizar(entidad)} ({i} de {min(n, len(entidades))})...")
            resultados.append(pipeline.consultar(entidad))
 
        yield aviso_archivos + armar_respuesta(transcripcion, entidades, origen, resultados,
                                               entidades[n:], info["guias"])
 
    descripcion = f"{AVISO}\n\n**Guías disponibles:** {html.escape('; '.join(info['guias']))}"
    for aviso in pipeline.avisos_carga:
        descripcion += f"\n\n**Aviso:** {aviso}"
    print(f"[app] Umbrales {info['umbral']}/{info['umbral_evidencia']} ({info['origen_umbrales']}) | "
          f"LLM: {info['modelo_llm'] or 'no disponible'} | GPU: {'sí' if info['gpu'] else 'no'}", flush=True)
 
    return gr.ChatInterface(
        fn=responder,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_types=["image"], file_count="multiple",
            placeholder="Escribe el caso clínico o adjunta la foto de la nota clínica (si tiene varias páginas, adjúntalas en orden)",
        ),
        additional_inputs=[
            gr.Dropdown(BACKENDS, value=pipeline.cfg["vision"]["backend_por_defecto"],
                        label="Motor para leer imágenes"),
            gr.Slider(1, 30, value=int(pipeline.cfg["rag"]["max_entidades"]), step=1,
                      label="Máximo de hallazgos a buscar en las guías"),
        ],
        additional_inputs_accordion=gr.Accordion("Opciones", open=False),
        examples=[[{"text": EJEMPLO_TEXTO, "files": []}]],
        run_examples_on_click=False,
        title=pipeline.cfg["interfaz"]["titulo"],
        description=descripcion,
        fill_height=True,
    )
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="App de punto de entrada de M4")
    parser.add_argument("--share", action="store_true", help="Enlace público temporal de Gradio")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 para exponerla en la red")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
 
    pipeline = PipelineClinico().cargar()
    app = construir_app(pipeline)
    app.queue(default_concurrency_limit=1)     # un solo juego de modelos en memoria
    app.launch(share=args.share, server_name=args.host,
               server_port=args.port or int(pipeline.cfg["interfaz"]["puerto"]))
 
 
if __name__ == "__main__":
    main()
 