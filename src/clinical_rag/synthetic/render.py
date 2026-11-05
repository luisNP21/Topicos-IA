"""Render impreso: texto -> paginas de imagen (tarea 3a).

Cada palabra se guarda como token con su posicion (ini, fin) en el texto original
y con su caja en la pagina. Asi se puede saber, despues de degradar la imagen,
que palabras y que entidades del gold siguen visibles.

No se parten palabras con guion: una entidad puede quedar dividida entre dos
lineas o dos paginas, que es uno de los casos limite de interes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# Fuentes
# --------------------------------------------------------------------------- #

def resolver_fuente(render_cfg: dict, base_dir: Path | None = None) -> str:
    """Devuelve la ruta de la primera fuente impresa disponible."""
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


# --------------------------------------------------------------------------- #
# Tokens y corte de lineas
# --------------------------------------------------------------------------- #

def tokenizar(texto: str) -> tuple[list[dict], list[list[int]]]:
    """Tokens (palabras separadas por espacios) con su posicion en el texto.

    Devuelve (tokens, parrafos): `parrafos` es una lista de listas de indices de token,
    una por cada linea del texto original (los parrafos vacios quedan como []).
    """
    tokens, parrafos = [], []
    pos = 0
    for linea in texto.split("\n"):
        indices = []
        for m in re.finditer(r"\S+", linea):
            indices.append(len(tokens))
            tokens.append({"texto": m.group(), "ini": pos + m.start(), "fin": pos + m.end()})
        parrafos.append(indices)
        pos += len(linea) + 1
    return tokens, parrafos


def _partir_token_largo(tok: dict, ancho_de, ancho_max: float) -> list[dict]:
    """Parte por caracteres un token que no cabe solo en una linea."""
    partes, ini = [], 0
    texto = tok["texto"]
    for j in range(1, len(texto) + 1):
        if j - ini > 1 and ancho_de(texto[ini:j]) > ancho_max:
            partes.append((ini, j - 1))
            ini = j - 1
    partes.append((ini, len(texto)))
    return [{"texto": texto[a:b], "ini": tok["ini"] + a, "fin": tok["ini"] + b} for a, b in partes]


def envolver(tokens: list[dict], parrafos: list[list[int]], ancho_de, ancho_espacio: float,
             ancho_max: float) -> tuple[list[dict], list[list[int]]]:
    """Corta en lineas por ancho en pixeles. Puede partir tokens muy largos (crea tokens nuevos)."""
    nuevos, lineas = [], []
    for parrafo in parrafos:
        if not parrafo:
            lineas.append([])
            continue
        actual, ancho_actual = [], 0.0
        for idx in parrafo:
            tok = tokens[idx]
            piezas = [tok] if ancho_de(tok["texto"]) <= ancho_max else _partir_token_largo(tok, ancho_de, ancho_max)
            for pieza in piezas:
                w = ancho_de(pieza["texto"])
                extra = w if not actual else ancho_espacio + w
                if actual and ancho_actual + extra > ancho_max:
                    lineas.append(actual)
                    actual, ancho_actual = [], 0.0
                    extra = w
                nuevos.append(pieza)
                actual.append(len(nuevos) - 1)
                ancho_actual += extra
        lineas.append(actual)
    return nuevos, lineas


def paginar(lineas: list[list[int]], lineas_por_pagina: int) -> list[list[list[int]]]:
    paginas = [lineas[i:i + lineas_por_pagina] for i in range(0, len(lineas), lineas_por_pagina)]
    paginas = [p for p in paginas if any(p)]
    return paginas or [[[]]]


def texto_de_lineas(tokens: list[dict], lineas: list[list[int]], visibles: set | None = None) -> str:
    """Texto de una pagina; si se pasa `visibles`, solo con los tokens visibles."""
    salida = []
    for linea in lineas:
        palabras = [tokens[i]["texto"] for i in linea if visibles is None or i in visibles]
        if visibles is not None and linea and not palabras:
            continue
        salida.append(" ".join(palabras))
    return "\n".join(salida)


# --------------------------------------------------------------------------- #
# Render impreso
# --------------------------------------------------------------------------- #

def renderizar_impreso(texto: str, fuente, render_cfg: dict) -> dict:
    """Devuelve {"tokens", "paginas"}; cada pagina trae imagen, lineas y cajas de palabra."""
    ancho, alto = render_cfg["ancho_px"], render_cfg["alto_px"]
    margen = render_cfg["margen_px"]
    paso = int(render_cfg["tamano_fuente"] * render_cfg["interlineado"])
    color = tuple(render_cfg.get("color_tinta", (20, 20, 20)))
    ascent, descent = fuente.getmetrics()

    tokens, parrafos = tokenizar(texto)
    espacio = fuente.getlength(" ")
    tokens, lineas = envolver(tokens, parrafos, fuente.getlength, espacio, ancho - 2 * margen)
    lineas_por_pagina = max(1, (alto - 2 * margen) // paso)

    paginas = []
    for lineas_pag in paginar(lineas, lineas_por_pagina):
        img = Image.new("RGB", (ancho, alto), "white")
        draw = ImageDraw.Draw(img)
        cajas = []
        y = margen
        for linea in lineas_pag:
            cadena = " ".join(tokens[i]["texto"] for i in linea)
            draw.text((margen, y), cadena, font=fuente, fill=color)
            x = float(margen)
            for j, i in enumerate(linea):
                if j:
                    x += espacio
                w = fuente.getlength(tokens[i]["texto"])
                cajas.append((i, x, y + 0.2 * ascent, x + w, y + ascent + 0.5 * descent))
                x += w
            y += paso
        paginas.append({
            "imagen": np.asarray(img, dtype=np.float32).copy(),
            "lineas": lineas_pag,
            "cajas": cajas,
            "alto_linea": float(paso),
            "color_papel": (255, 255, 255),
            "extra": [],
        })
    return {"tokens": tokens, "paginas": paginas}
