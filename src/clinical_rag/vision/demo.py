"""Demo sintética para probar el componente visual sin datos personales.

Genera una nota clínica FICTICIA en una imagen. No es el eval set de M4 (ese lo
construyen Luis y Pau): es un fixture para humo/pruebas y para correr el
pipeline en Colab sin subir fotos reales.

La imagen lleva un rótulo visible de que es sintética, siguiendo la regla ética
de S13: el contenido generado se marca.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXTO_NOTA_DEMO = """NOTA CLÍNICA SINTÉTICA - SIN DATOS PERSONALES

Motivo de consulta: control de enfermedad crónica.

Paciente ficticio de 52 años con diagnóstico de DM2 e HTA.
Antecedente de EPOC. Se solicita TFNA por dolor torácico atípico.

Signos vitales: TA 148/92 mmHg, FC 84 lpm, FR 18 rpm, T 36.8 C.
Laboratorio: glucemia 142 mg/dL, HbA1c 7.2 %, creatinina 1.1 mg/dL.

Impresión: DM2 con control subóptimo; HTA no controlada.
Plan: ajustar manejo farmacológico y educación sobre estilo de vida.
"""


def _fuente(tamano: int):
    candidatas = [
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "arial.ttf",
        "Arial.ttf",
    ]
    for nombre in candidatas:
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default()


def _envolver(draw: ImageDraw.ImageDraw, texto: str, fuente, ancho: int) -> list[str]:
    lineas: list[str] = []
    for parrafo in texto.splitlines():
        if not parrafo.strip():
            lineas.append("")
            continue
        actual = ""
        for palabra in parrafo.split():
            prueba = f"{actual} {palabra}".strip()
            try:
                ancho_prueba = draw.textlength(prueba, font=fuente)
            except AttributeError:  # Pillow viejo
                ancho_prueba = fuente.getsize(prueba)[0]
            if ancho_prueba <= ancho or not actual:
                actual = prueba
            else:
                lineas.append(actual)
                actual = palabra
        if actual:
            lineas.append(actual)
    return lineas


def crear_nota_sintetica(
    ruta: str | Path,
    texto: str | None = None,
    *,
    ancho: int = 900,
    margen: int = 40,
) -> Path:
    """Escribe una imagen PNG con la nota ficticia y devuelve su ruta."""
    texto = texto or TEXTO_NOTA_DEMO
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    fuente = _fuente(20)
    fuente_header = _fuente(16)
    alto_header = 46

    # Primera pasada para calcular el alto total.
    temporal = Image.new("RGB", (ancho, 100), "white")
    draw_tmp = ImageDraw.Draw(temporal)
    lineas = _envolver(draw_tmp, texto, fuente, ancho - 2 * margen)
    alto_linea = 26
    alto = alto_header + 2 * margen + alto_linea * max(1, len(lineas)) + 20

    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)

    # Rótulo visible de contenido sintético.
    draw.rectangle([0, 0, ancho, alto_header], fill=(20, 20, 20))
    draw.text(
        (margen, 14),
        "NOTA CLÍNICA SINTÉTICA · SIN DATOS PERSONALES · M4",
        font=fuente_header,
        fill=(255, 255, 255),
    )

    y = alto_header + margen
    for linea in lineas:
        draw.text((margen, y), linea, font=fuente, fill=(15, 15, 15))
        y += alto_linea

    imagen.save(ruta, format="PNG")
    return ruta


if __name__ == "__main__":  # pragma: no cover
    salida = crear_nota_sintetica(Path(__file__).parent / "data" / "demo" / "nota_sintetica.png")
    print(f"Imagen sintética creada en: {salida}")
