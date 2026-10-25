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
# Presentación
# ---------------------------------------------------------------------------
def _recortar(texto: str, n: int = 600) -> str:
    texto = " ".join((texto or "").split())
    return texto if len(texto) <= n else texto[:n].rsplit(" ", 1)[0] + "…"


def _bloque_transcripcion(tr: dict | None) -> str:
    if not tr:
        return ""
    lineas = [f"**Transcripción de la imagen** (backend `{tr['backend']}`)"]
    for p in tr["paginas"]:
        estado = "error" if p["error"] else ("sin texto legible" if p["valvula"] else "ok")
        lineas.append(f"- Página {p['doc_id'][1:]}: {estado}, {p['segundos']} s"
                      + (f" — {p['error']}" if p["error"] else ""))
    if tr["texto"]:
        lineas.append(f"\n<details><summary>Ver texto transcrito</summary>\n\n"
                      f"{html.escape(tr['texto'])}\n\n</details>")
    return "\n".join(lineas)


def _bloque_resultado(r: dict) -> str:
    titulo = f"### {r['entidad']}"
    if r["query_final"] != r["entidad"]:
        titulo += f"  →  consultada como «{r['query_final']}»"
    partes = [titulo]
    if r["tool_invoked"]:
        partes.append(f"*Normalización terminológica: {r['tool_reason']}*")

    if r["error"]:
        partes.append(f"**Error al generar la respuesta:** {r['error']}")
    elif r["respuesta"] is not None:
        etiqueta = "Sin evidencia suficiente en las guías" if r["fallback"] else "Respuesta basada en las guías"
        partes.append(f"**{etiqueta}:** {r['respuesta']}")

    if r["fragmentos"]:
        partes.append(f"**Fragmentos de guía ({len(r['fragmentos'])}):**")
        for i, f in enumerate(r["fragmentos"], 1):
            seccion = f" · {f['seccion']}" if f.get("seccion") else ""
            score = f" · score {f['score']:.3f}" if isinstance(f.get("score"), (int, float)) else ""
            partes.append(
                f"<details><summary>{i}. {html.escape(str(f['fuente']))}{html.escape(seccion)}{score}"
                f"</summary>\n\n{html.escape(_recortar(f['texto']))}\n\n"
                f"<sub>chunk_id: {html.escape(str(f['chunk_id']))}</sub>\n\n</details>"
            )
    else:
        partes.append("_Ningún fragmento superó la compuerta de evidencia: la guía no cubre esta entidad._")
    partes.append(f"<sub>{r['segundos']} s</sub>")
    return "\n\n".join(partes)


def _bloque_entidades(entidades: list[str], origen: str, n: int) -> str:
    if not entidades:
        return "**Entidades detectadas:** ninguna."
    consultadas = ", ".join(f"`{e}`" for e in entidades[:n])
    texto = f"**Entidades detectadas ({len(entidades)}):** {consultadas}"
    if len(entidades) > n:
        resto = ", ".join(f"`{e}`" for e in entidades[n:])
        texto += f"\n\n<sub>No consultadas (límite de {n}): {resto}</sub>"
    if origen == "consulta_directa":
        texto += "\n\n<sub>El NER no detectó entidades; se consultó el texto escrito tal cual.</sub>"
    return texto


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

        bloques: list[str] = []
        if ignorados:
            bloques.append(f"<sub>Archivos ignorados (solo se aceptan imágenes): {', '.join(ignorados)}</sub>")

        def mostrar(estado: str = "") -> str:
            return "\n\n".join(bloques + ([f"_{estado}_"] if estado else []))

        transcripcion = None
        if imagenes:
            nombre = pipeline.vision.resolver_backend(backend)
            aviso_cpu = " En CPU el VLM puede tardar varios minutos por página." if (
                nombre == "vlm" and not info["gpu"]) else ""
            yield mostrar(f"Transcribiendo {len(imagenes)} imagen(es) con {nombre}...{aviso_cpu}")
            transcripcion = pipeline.transcribir(imagenes, backend)
            bloques.append(_bloque_transcripcion(transcripcion))
            if not transcripcion["texto"] and not texto:
                bloques.append("No se pudo leer texto en la imagen, así que no hay entidades que consultar. "
                               "Prueba con una foto más nítida o escribe el caso.")
                yield mostrar()
                return

        texto_total = "\n".join(p for p in [transcripcion["texto"] if transcripcion else "", texto] if p)
        yield mostrar("Detectando entidades (NER)...")
        entidades, origen = pipeline.extraer_entidades(texto_total)
        n = int(max_entidades)
        bloques.append(_bloque_entidades(entidades, origen, n))
        if not entidades:
            bloques.append("Sin entidades de enfermedad no hay nada que buscar en las guías.")
            yield mostrar()
            return

        for i, entidad in enumerate(entidades[:n], 1):
            yield mostrar(f"Consultando guías para «{entidad}» ({i}/{min(n, len(entidades))})...")
            bloques.append(_bloque_resultado(pipeline.consultar(entidad)))
        yield mostrar()

    descripcion = (
        f"{AVISO}\n\n"
        f"<sub>Guías indexadas: {', '.join(info['guias'])} ({info['n_chunks']} fragmentos) · "
        f"umbrales {info['umbral']}/{info['umbral_evidencia']} ({info['origen_umbrales']}) · "
        f"LLM: {info['modelo_llm'] or 'no disponible'} · GPU: {'sí' if info['gpu'] else 'no'}</sub>"
    )
    for aviso in pipeline.avisos_carga:
        descripcion += f"\n\n**Aviso:** {aviso}"

    return gr.ChatInterface(
        fn=responder,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_types=["image"], file_count="multiple",
            placeholder="Escribe el caso clínico o adjunta la foto de la nota (varias imágenes = varias páginas)",
        ),
        additional_inputs=[
            gr.Dropdown(BACKENDS, value=pipeline.cfg["vision"]["backend_por_defecto"],
                        label="Motor para leer imágenes"),
            gr.Slider(1, 10, value=int(pipeline.cfg["rag"]["max_entidades"]), step=1,
                      label="Máximo de entidades a consultar en las guías"),
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
