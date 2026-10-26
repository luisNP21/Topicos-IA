"""Marcas sobre el papel (tarea 3a): sello, firma, mancha de cafe, resaltador,
perforaciones y notas a mano en el margen.

Todas se aplican en coordenadas de pagina, antes de deformar el papel o tomar la foto.
El texto que agregan (sello, notas) no esta en el gold y se registra en `pagina.extra`.
Los datos del sello son ficticios.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from manuscrito import _fuente, tamano_para_x_alto
from utilidades import Pagina, campo_suave, muestrear

MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]


def _multiplicar(p: Pagina, mascara: np.ndarray, color, opacidad: float, x0: int = 0, y0: int = 0) -> None:
    """Mezcla 'multiplicar' (tinta translucida sobre papel): el texto negro sigue negro."""
    h, w = mascara.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(p.ancho, x0 + w), min(p.alto, y0 + h)
    if xa >= xb or ya >= yb:
        return
    m = mascara[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * opacidad
    c = np.array(color, np.float32) / 255.0
    p.img[ya:yb, xa:xb] *= 1 - m * (1 - c)


def _tinta_irregular(mascara: np.ndarray, rng, fuerza: float) -> np.ndarray:
    """Tinta de sello: carga desigual y puntos sin tinta."""
    h, w = mascara.shape
    carga = np.clip(1 - fuerza * np.abs(campo_suave(rng, h, w, 25, 0.9)), 0, 1)
    huecos = rng.random((h, w)) > 0.12 * fuerza
    return mascara * carga * huecos


def _elegir_centro(p: Pagina, rng, posicion: str, radio: float) -> tuple[float, float]:
    caja = p.caja_texto() or (0, 0, p.ancho, p.alto)
    x0, y0, x1, y1 = caja
    if posicion == "centro":
        cx = (x0 + x1) / 2 + rng.uniform(-0.15, 0.15) * (x1 - x0)
        cy = (y0 + y1) / 2 + rng.uniform(-0.15, 0.15) * (y1 - y0)
    elif posicion == "abajo":
        cx = rng.uniform(0.55, 0.8) * p.ancho
        cy = min(p.alto - radio - 10, y1 + rng.uniform(-0.6, 0.3) * radio)
    else:
        cx = rng.uniform(x0 + radio * 0.5, max(x0 + radio * 0.5 + 1, x1 - radio * 0.5))
        cy = rng.uniform(y0 + radio * 0.5, max(y0 + radio * 0.5 + 1, y1 - radio * 0.5))
    return cx, cy


# --------------------------------------------------------------------------- #

def sello(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    radio = float(muestrear(params["radio_px"], rng))
    opacidad = float(muestrear(params["opacidad"], rng))
    color = muestrear(params["colores"], rng)
    institucion = muestrear(params["instituciones"], rng)
    centro_txt = muestrear(params["textos_centro"], rng)
    fecha = f"{int(rng.integers(1, 29)):02d} {MESES[int(rng.integers(12))]} {int(rng.integers(2022, 2026))}"

    lado = int(2 * radio + 20)
    img = Image.new("L", (lado, lado), 0)
    d = ImageDraw.Draw(img)
    c = lado / 2
    ancho_borde = max(3, int(radio * 0.045))
    d.ellipse([c - radio, c - radio, c + radio, c + radio], outline=255, width=ancho_borde)
    r2 = radio * 0.72
    d.ellipse([c - r2, c - r2, c + r2, c + r2], outline=255, width=max(2, ancho_borde // 2))

    # Texto en arco entre los dos circulos: tamano que llena la circunferencia; si el texto
    # es corto, se repite (como en los sellos reales).
    r_txt = (radio + r2) / 2
    circ = 2 * math.pi * r_txt * 0.97
    texto_arco = f"{institucion} * "
    f40 = _fuente(ctx["fuente_impresa"], 40)
    t_cap = max(10, int(radio * 0.19))
    t_arco = int(min(t_cap, 40 * circ / max(1.0, f40.getlength(texto_arco))))
    while 40 * circ / f40.getlength(texto_arco * 2) >= 0.65 * t_cap and len(texto_arco) < 200:
        texto_arco *= 2
        t_arco = int(min(t_cap, 40 * circ / f40.getlength(texto_arco)))
    t_arco = max(9, t_arco)
    f_arco = _fuente(ctx["fuente_impresa"], t_arco)
    total = sum(f_arco.getlength(ch) for ch in texto_arco)
    escala = min(circ / max(total, 1), 1.0)         # espaciado natural; si sobra arco, queda un hueco abajo
    ang = -math.pi / 2 - (total * escala / r_txt) / 2   # texto centrado arriba
    for ch in texto_arco:
        w = f_arco.getlength(ch) * escala
        medio = ang + (w / 2) / r_txt
        ci = Image.new("L", (t_arco * 2, t_arco * 2), 0)
        ImageDraw.Draw(ci).text((t_arco, t_arco), ch, font=f_arco, fill=255, anchor="mm", stroke_width=1, stroke_fill=255)
        ci = ci.rotate(-math.degrees(medio) - 90, resample=Image.BICUBIC)
        px, py = c + r_txt * math.cos(medio), c + r_txt * math.sin(medio)
        img.paste(255, (int(px - t_arco), int(py - t_arco)), ci)
        ang += w / r_txt

    t_c = max(10, int(radio * 0.24))
    largo_c = _fuente(ctx["fuente_impresa"], t_c).getlength(centro_txt)
    t_c = max(9, int(t_c * min(1.0, 1.35 * r2 / max(largo_c, 1))))   # que quepa en el circulo interior
    f_c = _fuente(ctx["fuente_impresa"], t_c)
    f_f = _fuente(ctx["fuente_impresa"], max(9, int(radio * 0.15)))
    d.text((c, c - radio * 0.08), centro_txt, font=f_c, fill=255, anchor="mm")
    d.text((c, c + radio * 0.25), fecha, font=f_f, fill=255, anchor="mm")

    angulo = float(rng.uniform(-35, 35))
    img = img.rotate(angulo, resample=Image.BICUBIC)
    mascara = _tinta_irregular(np.asarray(img, np.float32) / 255.0, rng, float(params.get("irregularidad", 0.5)))
    cx, cy = _elegir_centro(p, rng, params.get("posicion", "aleatoria"), radio)
    _multiplicar(p, mascara, color, opacidad, int(cx - lado / 2), int(cy - lado / 2))
    p.extra.append({"tipo": "sello", "texto": f"{institucion} {centro_txt} {fecha}"})
    return {"radio_px": radio, "opacidad": opacidad, "color": color, "centro": [cx, cy], "angulo": angulo}


def firma(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    largo = float(muestrear(params["largo_px"], rng))
    grosor = int(muestrear(params["grosor_px"], rng))
    color = muestrear(params["colores"], rng)
    alto = largo * 0.35
    cx, cy = _elegir_centro(p, rng, params.get("posicion", "abajo"), alto)

    t = np.linspace(0, 1, 260)
    f1, f2, f3 = rng.uniform(3, 6), rng.uniform(5, 9), rng.uniform(1, 2)
    ph = rng.uniform(0, 2 * math.pi, 3)
    x = cx - largo / 2 + largo * t + 0.12 * largo * np.sin(2 * math.pi * f1 * t + ph[0])
    y = cy + 0.45 * alto * np.sin(2 * math.pi * f2 * t + ph[1]) * np.exp(-1.5 * t) \
        + 0.2 * alto * np.sin(2 * math.pi * f3 * t + ph[2])
    lienzo = Image.new("L", (p.ancho, p.alto), 0)
    d = ImageDraw.Draw(lienzo)
    d.line(list(zip(x.tolist(), y.tolist())), fill=255, width=grosor, joint="curve")
    yr = cy + alto * rng.uniform(0.45, 0.7)
    d.line([(cx - largo * 0.45, yr), (cx + largo * 0.5, yr + rng.uniform(-8, 8))], fill=255, width=grosor)
    m = np.asarray(lienzo, np.float32) / 255.0
    _multiplicar(p, m, color, 0.95)
    tapados = p.ocultar_donde(np.asarray(lienzo.filter(ImageFilter.MaxFilter(5))) > 128)
    return {"largo_px": largo, "grosor_px": grosor, "color": color, "centro": [cx, cy], "puntos_tapados": tapados}


def mancha_cafe(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    radio = float(muestrear(params["radio_px"], rng))
    opacidad = float(muestrear(params["opacidad"], rng))
    cx, cy = _elegir_centro(p, rng, params.get("posicion", "aleatoria"), radio)
    lado = int(2.4 * radio)
    ys, xs = np.mgrid[0:lado, 0:lado].astype(np.float32) - lado / 2
    r = np.hypot(xs, ys)
    th = np.arctan2(ys, xs)
    ondula = 1 + 0.06 * np.sin(3 * th + rng.uniform(0, 6)) + 0.04 * np.sin(7 * th + rng.uniform(0, 6))
    borde = r / (radio * ondula)
    anillo = np.exp(-((borde - 1) / 0.05) ** 2)
    interior = (borde < 1) * 0.3
    m = np.clip(anillo + interior, 0, 1).astype(np.float32)
    m *= np.clip(1 + campo_suave(rng, lado, lado, 30, 0.3), 0.5, 1.3)
    _multiplicar(p, m, (150, 95, 45), opacidad, int(cx - lado / 2), int(cy - lado / 2))
    return {"radio_px": radio, "opacidad": opacidad, "centro": [cx, cy]}


def resaltador(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    n = int(muestrear(params["n_lineas"], rng))
    color = muestrear(params["colores"], rng)
    usados = []
    vivos = np.where(p.vivo)[0]
    for _ in range(n):
        if not len(vivos):
            break
        x, y = p.pts[vivos[int(rng.integers(len(vivos)))]]
        alto = p.alto_linea * 0.75
        largo = rng.uniform(0.25, 0.6) * p.ancho
        x0 = max(0, x - rng.uniform(0, 0.3) * largo)
        lado_x, lado_y = int(largo), int(alto * 1.4)
        m = np.zeros((lado_y, lado_x), np.float32)
        perfil = (alto / 2) * (1 + 0.08 * np.sin(np.linspace(0, rng.uniform(4, 9), lado_x)))
        filas = np.arange(lado_y)[:, None] - lado_y / 2
        m[np.abs(filas) < perfil[None, :]] = 1.0
        _multiplicar(p, m, color, 0.55, int(x0), int(y - lado_y / 2 - alto * 0.15))
        usados.append([float(x0), float(y)])
    return {"color": color, "lineas": usados}


def perforaciones(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    n = int(muestrear(params["n"], rng))
    radio = float(params.get("radio_px", 13))
    img = p.a_pil()
    d = ImageDraw.Draw(img)
    x = 42 + rng.uniform(-4, 4)
    ys = [p.alto * f for f in ([0.3, 0.7] if n == 2 else [0.2, 0.5, 0.8])]
    for y in ys:
        d.ellipse([x - radio - 2, y - radio - 2, x + radio + 2, y + radio + 2], fill=(200, 200, 200))
        d.ellipse([x - radio, y - radio, x + radio, y + radio], fill=(70, 70, 75))
    p.desde_pil(img)
    return {"n": n}


def notas_margen(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    n = int(muestrear(params["n"], rng))
    color = muestrear(params["colores"], rng)
    fuentes = ctx["fuentes_manuscritas"]
    caja = p.caja_texto() or (100, 100, p.ancho - 100, p.alto - 100)
    huecos = [("arriba", 15, caja[1] - 10), ("abajo", caja[3] + 25, p.alto - 15)]
    huecos = [h for h in huecos if h[2] - h[1] > 45]
    usadas = []
    for _ in range(n):
        texto = muestrear(params["textos"], rng)
        fuente = fuentes[int(rng.integers(len(fuentes)))]
        f = _fuente(fuente["ruta"], tamano_para_x_alto(fuente["ruta"], float(params.get("x_alto_px", 17))))
        w = int(f.getlength(texto)) + 20
        h = int(f.size * 1.8)
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((10, h * 0.7), texto, font=f, fill=255, anchor="ls")
        if huecos and rng.random() < 0.7:
            lugar, ya, yb = huecos[int(rng.integers(len(huecos)))]
            im = im.rotate(float(rng.uniform(-6, 6)), resample=Image.BICUBIC, expand=True)
            x0 = int(rng.uniform(caja[0], max(caja[0] + 1, caja[2] - im.width)))
            y0 = int(rng.uniform(ya, max(ya + 1, yb - im.height)))
        else:
            lugar = "margen_derecho"
            im = im.rotate(90 if rng.random() < 0.5 else -90, expand=True)
            x0 = int(min(p.ancho - im.width - 5, caja[2] + rng.uniform(5, 25)))
            y0 = int(rng.uniform(caja[1], max(caja[1] + 1, caja[3] - im.height)))
        _multiplicar(p, np.asarray(im, np.float32) / 255.0, color, 0.95, x0, y0)
        p.extra.append({"tipo": "nota_margen", "texto": texto})
        usadas.append({"texto": texto, "lugar": lugar, "fuente": fuente["nombre"]})
    return {"notas": usadas}


MARCAS = {
    "sello": sello,
    "firma": firma,
    "mancha_cafe": mancha_cafe,
    "resaltador": resaltador,
    "perforaciones": perforaciones,
    "notas_margen": notas_margen,
}
