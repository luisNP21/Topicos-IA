"""Render de texto a paginas de imagen (tarea 3a).

El texto se parte en lineas por ancho en pixeles (nunca se parten palabras con
guion, asi una entidad puede quedar dividida entre dos lineas o dos paginas,
que es uno de los casos limite de interes) y se reparte en paginas de tamano fijo.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def resolver_fuente(render_cfg: dict, base_dir: Path | None = None) -> str:
    """Devuelve la ruta de la primera fuente disponible."""
    for candidata in render_cfg["fuentes_candidatas"]:
        ruta = Path(candidata)
        if not ruta.is_absolute() and base_dir is not None:
            ruta = base_dir / ruta
        if ruta.exists():
            return str(ruta)

    if render_cfg.get("fuente_matplotlib_fallback", False):
        try:
            import matplotlib
            ruta = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            if ruta.exists():
                return str(ruta)
        except ImportError:
            pass

    raise FileNotFoundError(
        "No se encontro ninguna fuente con soporte de tildes y enie. "
        "Agrega una ruta valida en render.fuentes_candidatas de config.yaml."
    )


def cargar_fuente(ruta: str, tamano: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(ruta, tamano)


def _partir_palabra_larga(palabra: str, fuente, ancho_max: float) -> list[str]:
    """Parte por caracteres una palabra que no cabe sola en una linea."""
    partes, actual = [], ""
    for ch in palabra:
        if actual and fuente.getlength(actual + ch) > ancho_max:
            partes.append(actual)
            actual = ch
        else:
            actual += ch
    if actual:
        partes.append(actual)
    return partes


def partir_en_lineas(texto: str, fuente, ancho_max: float) -> list[str]:
    """Envuelve el texto por ancho en pixeles respetando los saltos de linea originales."""
    lineas = []
    for parrafo in texto.split("\n"):
        palabras = parrafo.split()
        if not palabras:
            lineas.append("")
            continue
        actual = ""
        for palabra in palabras:
            candidata = palabra if not actual else f"{actual} {palabra}"
            if fuente.getlength(candidata) <= ancho_max:
                actual = candidata
                continue
            if actual:
                lineas.append(actual)
            if fuente.getlength(palabra) <= ancho_max:
                actual = palabra
            else:
                *completas, resto = _partir_palabra_larga(palabra, fuente, ancho_max)
                lineas.extend(completas)
                actual = resto
        lineas.append(actual)
    return lineas


def paginar(lineas: list[str], lineas_por_pagina: int) -> list[list[str]]:
    paginas = [lineas[i:i + lineas_por_pagina] for i in range(0, len(lineas), lineas_por_pagina)]
    # Descarta lineas vacias al final de una pagina para no dejar paginas en blanco.
    paginas = [p for p in paginas if any(linea.strip() for linea in p)]
    return paginas or [[""]]


def _dibujar_linea_irregular(draw, linea: str, x0: int, y0: int, fuente, color, rng,
                             jitter_v: float, jitter_h: float) -> None:
    """Dibuja palabra por palabra con pequenas variaciones de posicion (aspecto manuscrito)."""
    x = float(x0)
    espacio = fuente.getlength(" ")
    for palabra in linea.split(" "):
        dy = rng.uniform(-jitter_v, jitter_v)
        draw.text((x, y0 + dy), palabra, font=fuente, fill=color)
        x += fuente.getlength(palabra) + espacio + rng.uniform(-jitter_h, jitter_h)


def dibujar_pagina(lineas: list[str], fuente, render_cfg: dict, rng=None) -> Image.Image:
    """Dibuja una pagina. Si el perfil define jitter, las lineas se dibujan con irregularidad."""
    ancho, alto = render_cfg["ancho_px"], render_cfg["alto_px"]
    margen = render_cfg["margen_px"]
    paso = int(render_cfg["tamano_fuente"] * render_cfg["interlineado"])
    color = tuple(render_cfg.get("color_tinta", (20, 20, 20)))
    jitter_v = float(render_cfg.get("jitter_vertical_px", 0))
    jitter_h = float(render_cfg.get("jitter_horizontal_px", 0))
    irregular = rng is not None and (jitter_v > 0 or jitter_h > 0)

    img = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(img)
    y = margen
    for linea in lineas:
        if irregular:
            _dibujar_linea_irregular(draw, linea, margen, y, fuente, color, rng, jitter_v, jitter_h)
        else:
            draw.text((margen, y), linea, font=fuente, fill=color)
        y += paso
    return img


def renderizar_documento(texto: str, fuente, render_cfg: dict, semilla_pagina=None) -> list[dict]:
    """Devuelve una lista de paginas: {"imagen": PIL.Image, "texto": str renderizado}.

    semilla_pagina: funcion opcional k -> int (k desde 1). Se usa solo si el perfil
    tiene jitter, para que la irregularidad sea reproducible.
    """
    import numpy as np

    ancho_util = render_cfg["ancho_px"] - 2 * render_cfg["margen_px"]
    alto_util = render_cfg["alto_px"] - 2 * render_cfg["margen_px"]
    paso = int(render_cfg["tamano_fuente"] * render_cfg["interlineado"])
    lineas_por_pagina = max(1, alto_util // paso)

    lineas = partir_en_lineas(texto, fuente, ancho_util)
    paginas = paginar(lineas, lineas_por_pagina)
    resultado = []
    for k, p in enumerate(paginas, start=1):
        rng = np.random.default_rng(semilla_pagina(k)) if semilla_pagina else None
        resultado.append({"imagen": dibujar_pagina(p, fuente, render_cfg, rng), "texto": "\n".join(p)})
    return resultado
