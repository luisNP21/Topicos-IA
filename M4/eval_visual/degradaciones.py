"""Efectos de degradacion del set sintetico (tarea 3a).

Cada efecto tiene la firma `efecto(pagina, rng, params, ctx) -> dict` y devuelve los
parametros que uso. Los parametros de config.yaml pueden ser un valor, un rango
[min, max] (se sortea uniforme) o una lista de opciones (se sortea una).

Un nivel de config.yaml es una lista de efectos en orden fisico:
marcas en el papel -> deformacion del papel -> foto (perspectiva) -> luz -> optica -> sensor.

Efectos que borran o tapan texto (recorte, rotacion fuerte, perspectiva fuera de cuadro,
flash muy fuerte, sombra muy oscura, firma encima) marcan los puntos de muestra afectados,
para que el manifiesto diga que palabras del gold ya no se pueden leer.

Todo es sintetico: aproxima escaneos y fotos, pero no los reemplaza.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from marcas import MARCAS
from utilidades import (
    Pagina,
    aplicar_desplazamiento,
    aplicar_homografia,
    campo_suave,
    homografia,
    muestrear,
)


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #

def rotacion(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Gira la hoja sin agrandar el lienzo: con angulos grandes se cortan las esquinas."""
    magnitud = float(muestrear(params["angulo"], rng))
    angulo = magnitud if rng.random() < 0.5 else -magnitud
    img = p.a_pil().rotate(angulo, resample=Image.BICUBIC, expand=False, fillcolor=p.color_papel)
    p.desde_pil(img)
    cx, cy = p.ancho / 2, p.alto / 2
    t = math.radians(angulo)
    x, y = p.pts[:, 0] - cx, p.pts[:, 1] - cy
    p.pts = np.stack([cx + x * math.cos(t) + y * math.sin(t), cy - x * math.sin(t) + y * math.cos(t)], 1).astype(np.float32)
    antes = int(p.vivo.sum())
    p.descartar_fuera_de_lienzo()
    return {"angulo_grados": angulo, "puntos_cortados": antes - int(p.vivo.sum())}


def recorte(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Corta uno o mas bordes metiendose una fraccion dentro del bloque de texto."""
    caja = p.caja_texto()
    if caja is None:
        return {"bordes": []}
    n = int(muestrear(params.get("bordes", 1), rng))
    bordes = [str(b) for b in rng.choice(["arriba", "abajo", "izquierda", "derecha"], size=n, replace=False)]
    x0, y0, x1, y1 = 0.0, 0.0, float(p.ancho), float(p.alto)
    tx0, ty0, tx1, ty1 = caja
    usados = {}
    for b in bordes:
        f = float(muestrear(params["fraccion"], rng))
        usados[b] = f
        if b == "arriba":
            y0 = ty0 + f * (ty1 - ty0)
        elif b == "abajo":
            y1 = ty1 - f * (ty1 - ty0)
        elif b == "izquierda":
            x0 = tx0 + f * (tx1 - tx0)
        else:
            x1 = tx1 - f * (tx1 - tx0)
    xi0, yi0, xi1, yi1 = int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))
    p.img = p.img[yi0:yi1, xi0:xi1].copy()
    p.pts = (p.pts - np.array([xi0, yi0], np.float32)).astype(np.float32)
    antes = int(p.vivo.sum())
    p.descartar_fuera_de_lienzo()
    return {"bordes": usados, "puntos_cortados": antes - int(p.vivo.sum())}


def orientacion(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Foto mal orientada: 90, 180 o 270 grados."""
    angulo = int(muestrear(params["angulos"], rng))
    w, h = p.ancho, p.alto
    x, y = p.pts[:, 0].copy(), p.pts[:, 1].copy()
    img = p.a_pil()
    if angulo == 90:
        img = img.transpose(Image.ROTATE_90)
        p.pts = np.stack([y, w - 1 - x], 1)
    elif angulo == 180:
        img = img.transpose(Image.ROTATE_180)
        p.pts = np.stack([w - 1 - x, h - 1 - y], 1)
    elif angulo == 270:
        img = img.transpose(Image.ROTATE_270)
        p.pts = np.stack([h - 1 - y, x], 1)
    p.pts = p.pts.astype(np.float32)
    p.desde_pil(img)
    return {"angulo_grados": angulo}


def resolucion(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    escala = float(muestrear(params["escala"], rng))
    w0, h0 = p.ancho, p.alto
    w, h = max(1, int(w0 * escala)), max(1, int(h0 * escala))
    p.desde_pil(p.a_pil().resize((w, h), Image.BILINEAR))
    p.pts = (p.pts * np.array([w / w0, h / h0], np.float32)).astype(np.float32)
    p.descartar_fuera_de_lienzo()
    return {"escala": escala, "tamano_resultante": [w, h]}


def deformacion_papel(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Ondulacion, curvatura (hoja abombada) y pliegues, con su sombreado."""
    alto, ancho = p.alto, p.ancho
    ys, xs = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    dx = np.zeros((alto, ancho), np.float32)
    dy = np.zeros((alto, ancho), np.float32)
    sombreado = np.ones((alto, ancho), np.float32)
    usados = {}

    amp = float(muestrear(params.get("ondulacion_amplitud_px", 0), rng))
    if amp > 0:
        lam = float(muestrear(params.get("ondulacion_longitud_px", 600), rng))
        fase = rng.uniform(0, 2 * math.pi)
        dy += amp * np.sin(2 * math.pi * xs / lam + fase)
        dx += 0.5 * amp * np.sin(2 * math.pi * ys / (lam * 1.3) + fase / 2)
        sombreado *= 1 + 0.025 * (amp / 5) * np.cos(2 * math.pi * xs / lam + fase)
        usados.update(ondulacion_amplitud_px=amp, ondulacion_longitud_px=lam)

    curv = float(muestrear(params.get("curvatura_px", 0), rng))
    if curv > 0:
        u = (xs - ancho / 2) / (ancho / 2)
        dy += curv * u ** 2 * ((ys - alto / 2) / (alto / 2))
        usados["curvatura_px"] = curv

    n_pliegues = int(muestrear(params.get("pliegues", 0), rng))
    desp = float(muestrear(params.get("pliegue_desplazamiento_px", 2), rng))
    marca = float(muestrear(params.get("pliegue_sombra", 0.25), rng))
    pliegues = []
    for k in range(n_pliegues):
        horizontal = (k % 2 == 0) if n_pliegues > 1 else rng.random() < 0.6
        theta = math.radians(rng.uniform(-6, 6)) + (0 if horizontal else math.pi / 2)
        px0 = ancho * rng.uniform(0.35, 0.65)
        py0 = alto * rng.uniform(0.35, 0.65)
        nx, ny = -math.sin(theta), math.cos(theta)
        dist = (xs - px0) * nx + (ys - py0) * ny
        paso = desp * np.tanh(dist / 12.0)
        dx += paso * nx
        dy += paso * ny
        sombreado *= 1 - marca * np.exp(-(dist / 3.5) ** 2)
        sombreado *= 1 - 0.07 * marca * (dist > 0) * np.exp(-dist.clip(0) / 160)
        sombreado *= 1 + 0.12 * marca * np.exp(-((dist + 6) / 4) ** 2)
        pliegues.append({"horizontal": horizontal, "punto": [px0, py0]})
    if pliegues:
        usados.update(pliegues=pliegues, pliegue_desplazamiento_px=desp, pliegue_sombra=marca)

    aplicar_desplazamiento(p, dx, dy)
    p.img *= sombreado[..., None]
    return usados


def _fondo(rng, ancho: int, alto: int, colores: list) -> np.ndarray:
    base = np.array(muestrear(colores, rng), np.float32)
    arr = np.broadcast_to(base, (alto, ancho, 3)).astype(np.float32).copy()
    arr *= (1 + campo_suave(rng, alto, ancho, 120, 0.08))[..., None]
    vetas = campo_suave(rng, max(2, alto // 40), ancho, 8, 0.05)
    arr *= (1 + np.asarray(Image.fromarray(vetas, "F").resize((ancho, alto), Image.BILINEAR)))[..., None]
    arr += rng.normal(0, 3, size=arr.shape).astype(np.float32)
    return arr


def perspectiva(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Foto tomada en angulo: la hoja queda como un trapecio sobre una mesa."""
    w, h = p.ancho, p.alto
    margen = float(muestrear(params["margen"], rng))
    incl = float(muestrear(params["inclinacion"], rng))
    destino = []
    for sx, sy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        bx = w * (margen if sx == 0 else 1 - margen)
        by = h * (margen if sy == 0 else 1 - margen)
        destino.append((bx + rng.uniform(-incl, incl) * w, by + rng.uniform(-incl, incl) * h))
    origen = [(0, 0), (w, 0), (w, h), (0, h)]
    adelante = homografia(origen, destino)
    atras = homografia(destino, origen)

    hoja = p.a_pil().transform((w, h), Image.PERSPECTIVE, tuple(atras), Image.BICUBIC, fillcolor=(0, 0, 0))
    mascara = Image.new("L", (w, h), 255).transform((w, h), Image.PERSPECTIVE, tuple(atras), Image.BILINEAR, fillcolor=0)
    fondo = _fondo(rng, w, h, params["colores_fondo"])
    sombra = np.asarray(mascara.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255.0
    sombra = np.roll(np.roll(sombra, 10, axis=0), 7, axis=1)
    fondo *= (1 - 0.45 * sombra)[..., None]
    m = (np.asarray(mascara, np.float32) / 255.0)[..., None]
    p.img = fondo * (1 - m) + np.asarray(hoja, np.float32) * m
    p.pts = aplicar_homografia(adelante, p.pts)
    antes = int(p.vivo.sum())
    p.descartar_fuera_de_lienzo()
    return {"margen": margen, "inclinacion": incl, "esquinas": [list(map(float, c)) for c in destino],
            "puntos_fuera_de_cuadro": antes - int(p.vivo.sum())}


# --------------------------------------------------------------------------- #
# Luz
# --------------------------------------------------------------------------- #

def _silueta(rng, ancho: int, alto: int, ancho_frac: float, alcance: float) -> Image.Image:
    """Mano o celular entrando desde un borde hasta `alcance` (0.5 = centro de la pagina)."""
    borde = int(rng.integers(4))
    largo_borde = ancho if borde in (0, 2) else alto
    fondo_dim = alto if borde in (0, 2) else ancho
    centro = rng.uniform(0.25, 0.75) * largo_borde
    medio = ancho_frac * largo_borde / 2
    prof = alcance * fondo_dim
    desvio = rng.uniform(-0.15, 0.15) * largo_borde
    pts = [(centro - medio * 1.2, -60), (centro - medio + desvio * 0.5, prof * 0.55)]
    for a in np.linspace(math.pi, 0, 9):
        pts.append((centro + desvio + medio * 0.9 * math.cos(a), prof + medio * 0.6 * math.sin(a) * 0.8))
    pts += [(centro + medio + desvio * 0.5, prof * 0.55), (centro + medio * 1.2, -60)]
    if borde == 1:
        pts = [(ancho - y, x) for x, y in pts]
    elif borde == 2:
        pts = [(x, alto - y) for x, y in pts]
    elif borde == 3:
        pts = [(y, x) for x, y in pts]
    img = Image.new("L", (ancho, alto), 0)
    ImageDraw.Draw(img).polygon(pts, fill=255)
    return img


def sombra_dura(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Sombra de una mano o del celular que cruza la hoja (puede llegar al centro)."""
    n = int(muestrear(params.get("n", 1), rng))
    factor = np.ones((p.alto, p.ancho), np.float32)
    usadas = []
    for _ in range(n):
        osc = float(muestrear(params["oscuridad"], rng))
        pen = float(muestrear(params["penumbra_px"], rng))
        anc = float(muestrear(params["ancho"], rng))
        alc = float(muestrear(params["alcance"], rng))
        m = _silueta(rng, p.ancho, p.alto, anc, alc).filter(ImageFilter.GaussianBlur(pen))
        factor *= 1 - osc * np.asarray(m, np.float32) / 255.0
        usadas.append({"oscuridad": osc, "penumbra_px": pen, "ancho": anc, "alcance": alc})
    tinte = np.array([0.94, 0.985, 1.0], np.float32)   # la sombra queda levemente azulada
    p.img *= factor[..., None] * (1 - (1 - factor[..., None]) * (1 - tinte))
    umbral = params.get("umbral_ilegible")
    tapados = p.ocultar_donde(factor < umbral) if umbral is not None else 0
    return {"sombras": usadas, "puntos_tapados": tapados}


def sombra_suave(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Iluminacion desigual: degradado diagonal y vineta (version anterior de sombra_celular)."""
    ys, xs = np.mgrid[0:p.alto, 0:p.ancho].astype(np.float32)
    theta = float(rng.uniform(0, 2 * np.pi))
    proy = np.cos(theta) * xs / p.ancho + np.sin(theta) * ys / p.alto
    proy = (proy - proy.min()) / (proy.max() - proy.min() + 1e-6)
    osc = float(muestrear(params["oscuridad"], rng))
    cx, cy = p.ancho / 2, p.alto / 2
    dist = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) / np.sqrt(2)
    vin = float(params.get("vineta", 0.2))
    p.img *= ((1 - osc * proy) * (1 - vin * dist ** 2))[..., None]
    return {"angulo_rad": theta, "oscuridad": osc, "vineta": vin}


def flash(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Reflejo del flash: una zona brillante que deja el texto casi blanco."""
    inten = float(muestrear(params["intensidad"], rng))
    radio = float(muestrear(params["radio"], rng)) * min(p.ancho, p.alto)
    cx, cy = rng.uniform(0.3, 0.7) * p.ancho, rng.uniform(0.3, 0.7) * p.alto
    ys, xs = np.mgrid[0:p.alto, 0:p.ancho].astype(np.float32)
    elong = rng.uniform(0.7, 1.3)
    d2 = ((xs - cx) / (radio * elong)) ** 2 + ((ys - cy) / (radio / elong)) ** 2
    s = np.clip(inten * np.exp(-d2), 0, 1)
    # Con flash, la luz cae lejos del punto caliente: el resto de la hoja queda mas oscuro.
    caida = float(muestrear(params.get("caida", [0.12, 0.25]), rng))
    p.img *= (1 - caida * (1 - np.exp(-d2 / 9.0)))[..., None]
    p.img = p.img + (255 - p.img) * s[..., None]
    umbral = params.get("umbral_ilegible")
    tapados = p.ocultar_donde(s >= umbral) if umbral is not None else 0
    return {"intensidad": inten, "radio_px": radio, "caida": caida, "centro": [cx, cy], "puntos_tapados": tapados}


def tinte(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Balance de blancos: calido (luz de bombillo), frio o verdoso (fluorescente)."""
    modo = muestrear(params["modos"], rng)
    f = float(muestrear(params["fuerza"], rng))
    mult = {"calido": (1.0, 1 - 0.05 * f, 1 - 0.16 * f),
            "frio": (1 - 0.10 * f, 1 - 0.02 * f, 1.0),
            "verdoso": (1 - 0.07 * f, 1.0, 1 - 0.09 * f)}[modo]
    p.img *= np.array(mult, np.float32)
    return {"modo": modo, "fuerza": f}


def poca_luz(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    expo = float(muestrear(params["exposicion"], rng))
    gamma = float(muestrear(params["gamma"], rng))
    sigma = float(muestrear(params.get("ruido_sigma", 8), rng))
    base = np.clip(p.img / 255.0, 0, 1) ** gamma * expo * 255.0
    p.img = base + rng.normal(0, sigma, size=p.img.shape).astype(np.float32)
    return {"exposicion": expo, "gamma": gamma, "ruido_sigma": sigma}


# --------------------------------------------------------------------------- #
# Optica y sensor
# --------------------------------------------------------------------------- #

def desenfoque(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    radio = float(muestrear(params["radio"], rng))
    p.desde_pil(p.a_pil().filter(ImageFilter.GaussianBlur(radius=radio)))
    return {"radio": radio}


def desenfoque_parcial(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Profundidad de campo: una parte de la hoja sale nitida y la otra fuera de foco."""
    radio = float(muestrear(params["radio"], rng))
    borroso = np.asarray(p.a_pil().filter(ImageFilter.GaussianBlur(radius=radio)), np.float32)
    ys, xs = np.mgrid[0:p.alto, 0:p.ancho].astype(np.float32)
    theta = rng.uniform(0, 2 * math.pi)
    proy = math.cos(theta) * xs / p.ancho + math.sin(theta) * ys / p.alto
    proy = (proy - proy.min()) / (proy.max() - proy.min() + 1e-6)
    inicio = float(muestrear(params.get("inicio", [0.3, 0.5]), rng))
    m = np.clip((proy - inicio) / 0.3, 0, 1)[..., None]
    p.img = p.img * (1 - m) + borroso * m
    return {"radio": radio, "angulo_rad": theta, "inicio": inicio}


def movimiento(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Desenfoque de movimiento (mano que tiembla al tomar la foto)."""
    largo = int(round(float(muestrear(params["largo_px"], rng))))
    theta = rng.uniform(0, math.pi)
    if largo < 2:
        return {"largo_px": largo}
    pad = largo + 1
    arr = np.pad(p.img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(p.img)
    for t in np.linspace(-largo / 2, largo / 2, largo):
        ox, oy = int(round(t * math.cos(theta))), int(round(t * math.sin(theta)))
        acc += arr[pad + oy:pad + oy + p.alto, pad + ox:pad + ox + p.ancho]
    p.img = acc / largo
    return {"largo_px": largo, "angulo_rad": theta}


def ruido(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    sigma = float(muestrear(params["sigma"], rng))
    p.img = p.img + rng.normal(0, sigma, size=p.img.shape).astype(np.float32)
    return {"sigma": sigma}


def jpeg(p: Pagina, rng, params: dict, ctx: dict) -> dict:
    """Compresion intermedia (p. ej., una foto reenviada por chat antes de reescalar)."""
    calidad = int(muestrear(params["calidad"], rng))
    buf = io.BytesIO()
    p.a_pil().save(buf, format="JPEG", quality=calidad)
    buf.seek(0)
    p.desde_pil(Image.open(buf))
    return {"calidad": calidad}


EFECTOS = {
    "rotacion": rotacion,
    "recorte": recorte,
    "orientacion": orientacion,
    "resolucion": resolucion,
    "deformacion_papel": deformacion_papel,
    "perspectiva": perspectiva,
    "sombra_dura": sombra_dura,
    "sombra_suave": sombra_suave,
    "flash": flash,
    "tinte": tinte,
    "poca_luz": poca_luz,
    "desenfoque": desenfoque,
    "desenfoque_parcial": desenfoque_parcial,
    "movimiento": movimiento,
    "ruido": ruido,
    "jpeg": jpeg,
    **MARCAS,
}
