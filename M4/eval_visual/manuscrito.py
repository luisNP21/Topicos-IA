"""Render manuscrito sintetico (tarea 3a).

Simula una nota escrita a mano con:
- varias caligrafias (fuentes libres; una por documento),
- variacion por letra (tamano, posicion, presion de la tinta), linea base ondulada
  y con pendiente, inclinacion de la letra y giro de cada palabra,
- distorsion elastica de los trazos,
- errores de escritura: palabras empezadas y tachadas, palabras pegadas, tinta que
  falla y letras repasadas,
- papel blanco, rayado o cuadriculado.

Los errores nunca cambian las palabras del texto: una tachadura agrega una palabra
extra (que se registra como texto extra) y el texto original queda completo.
Sigue siendo una aproximacion: no es letra de una persona real.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import _partir_token_largo, paginar, tokenizar
from utilidades import campo_suave, mover_puntos_inverso, muestrear, puntos_de_cajas, remap_bilineal


# --------------------------------------------------------------------------- #
# Fuentes
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _fuente(ruta: str, tamano: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(ruta, tamano)


@lru_cache(maxsize=None)
def _notdef(ruta: str) -> tuple:
    m = _fuente(ruta, 40).getmask("\U0010FFFD")
    return m.size, bytes(m)


@lru_cache(maxsize=None)
def tiene_glifo(ruta: str, caracter: str) -> bool:
    if caracter.isspace():
        return True
    m = _fuente(ruta, 40).getmask(caracter)
    return m.getbbox() is not None and (m.size, bytes(m)) != _notdef(ruta)


@lru_cache(maxsize=None)
def tamano_para_x_alto(ruta: str, x_alto: float) -> int:
    """Tamano de fuente con el que la 'x' mide `x_alto` px (iguala fuentes distintas)."""
    b = _fuente(ruta, 40).getbbox("x")
    return max(10, int(round(40 * x_alto / max(1, b[3] - b[1]))))


# --------------------------------------------------------------------------- #
# Papel
# --------------------------------------------------------------------------- #

def _papel(tipo: str, ancho: int, alto: int, cfg: dict, paso: float, primera_base: float,
           rng: np.random.Generator) -> tuple[np.ndarray, tuple]:
    base = np.array([248, 246, 238], np.float32) + rng.uniform(-4, 4, size=3).astype(np.float32)
    img = Image.new("RGB", (ancho, alto), tuple(int(v) for v in base))
    draw = ImageDraw.Draw(img)
    if tipo == "rayado":
        y = primera_base + 3
        while y < alto - 20:
            draw.line([(0, y), (ancho, y)], fill=(160, 190, 225), width=2)
            y += paso
        xm = cfg["margen_izq_px"] - 18
        draw.line([(xm, 0), (xm, alto)], fill=(225, 140, 140), width=2)
    elif tipo == "cuadriculado":
        lado = paso / 2
        y = (primera_base + 3) % lado
        while y < alto:
            draw.line([(0, y), (ancho, y)], fill=(190, 208, 230), width=1)
            y += lado
        x = cfg["margen_izq_px"] % lado
        while x < ancho:
            draw.line([(x, 0), (x, alto)], fill=(190, 208, 230), width=1)
            x += lado
    arr = np.asarray(img, dtype=np.float32).copy()
    textura = campo_suave(rng, alto, ancho, 60, 2.5) + rng.normal(0, 1.2, size=(alto, ancho)).astype(np.float32)
    arr += textura[..., None]
    return np.clip(arr, 0, 255), tuple(int(v) for v in base)


# --------------------------------------------------------------------------- #
# Palabras
# --------------------------------------------------------------------------- #

def _dibujar_palabra(texto: str, ruta: str, ruta_respaldo: str, tamanos: list[int], rng, est: dict):
    """Dibuja la palabra letra por letra. Devuelve (mascara float 0..1, fila_base, avance)."""
    t_max = max(tamanos)
    ancho_img = int(sum(_fuente(ruta, t_max).getlength(c) for c in texto) * 1.25 + 4 * t_max)
    alto_img = int(t_max * 2.4)
    fila_base = int(t_max * 1.55)
    img = Image.new("L", (ancho_img, alto_img), 0)
    draw = ImageDraw.Draw(img)

    falla = None
    if len(texto) > 3 and rng.random() < est["p_tinta_falla"]:
        a = int(rng.integers(0, len(texto) - 2))
        falla = (a, a + int(rng.integers(2, 4)))
    repaso = int(rng.integers(len(texto))) if texto and rng.random() < est["p_repaso"] else -1

    x = float(t_max)
    centro = (len(tamanos) - 1) / 2
    dy = 0.0
    for j, c in enumerate(texto):
        k = int(np.clip(round(centro + rng.normal(0, 1) * est["dispersion_tamano"]), 0, len(tamanos) - 1))
        r = ruta if tiene_glifo(ruta, c) else ruta_respaldo
        f = _fuente(r, tamanos[k])
        presion = rng.uniform(*est["presion"])
        if falla and falla[0] <= j < falla[1]:
            presion *= rng.uniform(0.2, 0.4)
        valor = int(255 * presion)
        # Deriva vertical correlacionada entre letras vecinas (una mano no salta al azar en
        # cada letra). Desviacion estacionaria ~ jitter_vertical_px.
        dy = 0.6 * dy + rng.normal(0, 0.8 * est["jitter_vertical_px"])
        pos = (x, fila_base + dy)
        draw.text(pos, c, font=f, fill=valor, anchor="ls", stroke_width=est["grosor"], stroke_fill=valor)
        if j == repaso:
            draw.text((x + rng.uniform(-1.5, 1.5), fila_base + dy + rng.uniform(-1.5, 1.5)), c, font=f,
                      fill=valor, anchor="ls", stroke_width=est["grosor"], stroke_fill=valor)
        x += f.getlength(c) + rng.normal(0, est["jitter_horizontal_px"])
    avance = x - t_max
    return np.asarray(img, dtype=np.float32) / 255.0, fila_base, avance, t_max


def _transformar(mascara: np.ndarray, ancla: tuple[float, float], inclinacion: float, angulo: float):
    """Inclina (cursiva) y gira la mascara alrededor de `ancla`.

    Devuelve (mascara_t, ox, oy): el ancla queda en (ox, oy) de la mascara transformada.
    """
    k = math.tan(inclinacion)
    c, s = math.cos(angulo), math.sin(angulo)
    # M = R * S: S inclina la parte alta hacia la derecha; R gira (sentido antihorario visual).
    S = np.array([[1.0, -k], [0.0, 1.0]])
    R = np.array([[c, s], [-s, c]])
    M = R @ S
    alto, ancho = mascara.shape
    ax, ay = ancla
    esquinas = np.array([[0, 0], [ancho, 0], [0, alto], [ancho, alto]], float) - [ax, ay]
    trans = esquinas @ M.T
    mn, mx = trans.min(axis=0), trans.max(axis=0)
    w, h = int(math.ceil(mx[0] - mn[0])) + 1, int(math.ceil(mx[1] - mn[1])) + 1
    Mi = np.linalg.inv(M)
    data = (Mi[0, 0], Mi[0, 1], Mi[0, 0] * mn[0] + Mi[0, 1] * mn[1] + ax,
            Mi[1, 0], Mi[1, 1], Mi[1, 0] * mn[0] + Mi[1, 1] * mn[1] + ay)
    img = Image.fromarray((mascara * 255).astype(np.uint8), "L")
    out = img.transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(out, dtype=np.float32) / 255.0, -mn[0], -mn[1]


def _pegar(lienzo: np.ndarray, mascara: np.ndarray, x0: int, y0: int) -> tuple | None:
    """np.maximum de la mascara sobre el lienzo; devuelve la caja de tinta en coordenadas de pagina."""
    alto, ancho = lienzo.shape
    h, w = mascara.shape
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(ancho, x0 + w), min(alto, y0 + h)
    if xa >= xb or ya >= yb:
        return None
    sub = mascara[ya - y0:yb - y0, xa - x0:xb - x0]
    lienzo[ya:yb, xa:xb] = np.maximum(lienzo[ya:yb, xa:xb], sub)
    filas = np.where(sub.max(axis=1) > 0.25)[0]
    cols = np.where(sub.max(axis=0) > 0.25)[0]
    if not len(filas):
        return None
    return xa + cols[0], ya + filas[0], xa + cols[-1] + 1, ya + filas[-1] + 1


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def elegir_estilo(cfg: dict, fuentes: list[dict], rng_estilo: np.random.Generator) -> dict:
    """Caligrafia y color por documento. No depende de la severidad, asi el mismo documento
    usa la misma letra en manuscrito_leve, _moderado y _severo."""
    fuente = fuentes[int(rng_estilo.integers(len(fuentes)))]
    colores = cfg["colores_tinta"]
    color = colores[int(rng_estilo.integers(len(colores)))]
    return {"fuente": fuente, "color": color, "u_inclinacion": float(rng_estilo.random())}


def renderizar_manuscrito(texto: str, cfg: dict, estilo_doc: dict, ruta_respaldo: str,
                          rng: np.random.Generator) -> dict:
    ancho, alto = cfg["ancho_px"], cfg["alto_px"]
    ruta = estilo_doc["fuente"]["ruta"]

    paso = float(muestrear(cfg["paso_linea_px"], rng))
    x_alto = float(muestrear(cfg["x_alto_px"], rng))
    lo, hi = cfg["inclinacion_grados"]
    inclinacion = math.radians(lo + (hi - lo) * estilo_doc["u_inclinacion"])
    papel = muestrear(cfg["papel"], rng)
    est = {
        "dispersion_tamano": float(cfg["dispersion_tamano"]),
        "presion": tuple(cfg["presion"]),
        "jitter_vertical_px": float(cfg["jitter_vertical_px"]),
        "jitter_horizontal_px": float(cfg["jitter_horizontal_px"]),
        "p_tinta_falla": float(cfg["p_tinta_falla"]),
        "p_repaso": float(cfg["p_repaso"]),
        "grosor": int(muestrear(cfg["grosor"], rng)),
    }
    t0 = tamano_para_x_alto(ruta, x_alto)
    v = float(cfg["variacion_tamano"])
    tamanos = sorted({max(8, int(round(t0 * (1 + v * f)))) for f in (-1, -0.5, 0, 0.5, 1)})
    f_base = _fuente(ruta, t0)
    # Algunas caligrafias traen un espacio muy angosto; con la inclinacion las palabras se pegarian.
    espacio = max(f_base.getlength(" "), 0.8 * x_alto) * 1.15

    def ancho_est(t: str) -> float:
        return f_base.getlength(t) * (1 + 0.6 * v) + 2 * len(t) * est["jitter_horizontal_px"]

    # 1) Tokens reales + tachaduras insertadas (solo visuales, no estan en el gold).
    tokens_orig, parrafos = tokenizar(texto)
    margen_izq, margen_der = cfg["margen_izq_px"], cfg["margen_der_px"]
    ancho_max = ancho - margen_izq - margen_der
    tokens, parrafos_items = [], []
    for parrafo in parrafos:
        items = []
        for idx in parrafo:
            tok = tokens_orig[idx]
            if (len(tok["texto"]) >= 4 and tok["texto"][:2].isalpha()
                    and rng.random() < cfg["p_tachadura"]):
                if rng.random() < 0.7:
                    falso = tok["texto"][:int(rng.integers(2, len(tok["texto"])))]
                else:
                    falso = tokens_orig[int(rng.integers(len(tokens_orig)))]["texto"]
                items.append({"tipo": "tachadura", "texto": falso, "w": ancho_est(falso)})
            piezas = [tok] if ancho_est(tok["texto"]) <= ancho_max else _partir_token_largo(tok, ancho_est, ancho_max)
            for pieza in piezas:
                tokens.append(pieza)
                items.append({"tipo": "token", "idx": len(tokens) - 1, "texto": pieza["texto"],
                              "w": ancho_est(pieza["texto"])})
        parrafos_items.append(items)

    # 2) Corte de lineas por ancho estimado.
    lineas = []
    for items in parrafos_items:
        actual, w_act = [], 0.0
        for it in items:
            extra = it["w"] if not actual else espacio + it["w"]
            if actual and w_act + extra > ancho_max:
                lineas.append(actual)
                actual, w_act, extra = [], 0.0, it["w"]
            actual.append(it)
            w_act += extra
        lineas.append(actual)
    lineas_por_pagina = max(1, int((alto - cfg["margen_sup_px"] - cfg["margen_inf_px"]) // paso))

    paginas = []
    for lineas_pag in paginar(lineas, lineas_por_pagina):
        primera_base = cfg["margen_sup_px"] + 0.78 * paso
        fondo, color_papel = _papel(papel, ancho, alto, cfg, paso, primera_base, rng)
        tinta = np.zeros((alto, ancho), np.float32)
        trazos = Image.new("L", (ancho, alto), 0)
        draw_trazos = ImageDraw.Draw(trazos)
        cajas, extra = [], []

        pend_max = math.radians(cfg["pendiente_linea_grados"])
        for i, linea in enumerate(lineas_pag):
            base0 = primera_base + i * paso
            pendiente = math.tan(rng.uniform(-pend_max, pend_max))
            amp = float(muestrear(cfg["onda_amplitud_px"], rng))
            lam = float(muestrear(cfg["onda_longitud_px"], rng))
            fase = rng.uniform(0, 2 * math.pi)
            x = margen_izq + rng.uniform(0, 8)
            for it in linea:
                y_base = base0 + pendiente * (x - margen_izq) + amp * math.sin(2 * math.pi * x / lam + fase)
                mascara, fila_base, avance, t_max = _dibujar_palabra(it["texto"], ruta, ruta_respaldo, tamanos, rng, est)
                giro = math.radians(rng.normal(0, cfg["giro_palabra_grados"]))
                mt, ox, oy = _transformar(mascara, (t_max, fila_base), inclinacion, giro)
                caja = _pegar(tinta, mt, int(round(x - ox)), int(round(y_base - oy)))
                if it["tipo"] == "token":
                    if caja is None:
                        caja = (x, y_base - x_alto, x + max(avance, 4), y_base)
                    cajas.append((it["idx"], *caja))
                else:
                    _tachar(draw_trazos, caja or (x, y_base - x_alto, x + avance, y_base), est, rng, cfg)
                    extra.append({"tipo": "tachadura", "texto": it["texto"]})
                sep = espacio * rng.uniform(0.8, 1.25)
                if rng.random() < cfg["p_pegadas"]:
                    sep = espacio * rng.uniform(0.05, 0.2)
                x += avance + sep

        tinta = np.maximum(tinta, np.asarray(trazos, dtype=np.float32) / 255.0)
        pts, pt_token = puntos_de_cajas(cajas)

        amp_el = float(muestrear(cfg["elastico_amplitud_px"], rng))
        if amp_el > 0:
            celda = float(cfg["elastico_celda_px"])
            dx = campo_suave(rng, alto, ancho, celda, amp_el)
            dy = campo_suave(rng, alto, ancho, celda, amp_el)
            ys, xs = np.mgrid[0:alto, 0:ancho].astype(np.float32)
            tinta = remap_bilineal(tinta, xs + dx, ys + dy, relleno=0.0)
            pts = mover_puntos_inverso(pts, dx, dy)

        color = np.array(estilo_doc["color"], np.float32)
        a = np.clip(tinta, 0, 1)[..., None]
        img = fondo * (1 - a) + color * a
        paginas.append({
            "imagen": img.astype(np.float32),
            "lineas": [[it["idx"] for it in linea if it["tipo"] == "token"] for linea in lineas_pag],
            "cajas": cajas,
            "pts": (pts, pt_token),
            "alto_linea": paso,
            "color_papel": color_papel,
            "extra": extra,
        })

    estilo = {
        "fuente": estilo_doc["fuente"]["nombre"],
        "dificultad_fuente": estilo_doc["fuente"].get("dificultad"),
        "color_tinta": list(estilo_doc["color"]),
        "inclinacion_grados": round(math.degrees(inclinacion), 2),
        "papel": papel,
        "paso_linea_px": round(paso, 1),
        "x_alto_px": round(x_alto, 1),
        "tamano_fuente": t0,
        "grosor": est["grosor"],
    }
    return {"tokens": tokens, "paginas": paginas, "estilo": estilo}


def _tachar(draw: ImageDraw.ImageDraw, caja: tuple, est: dict, rng, cfg: dict) -> None:
    """Una o dos lineas sobre la palabra; con `garabato_tachadura`, un zigzag que la tapa."""
    x0, y0, x1, y1 = caja
    ancho_linea = 2 + est["grosor"]
    if rng.random() < cfg.get("p_garabato", 0.0):
        pts, x = [], x0 - 2
        alto = y1 - y0
        while x < x1 + 2:
            pts.append((x, y0 + rng.uniform(0, 0.3) * alto))
            pts.append((x + rng.uniform(3, 7), y1 - rng.uniform(0, 0.3) * alto))
            x += rng.uniform(6, 11)
        draw.line(pts, fill=230, width=ancho_linea, joint="curve")
        return
    for _ in range(1 if rng.random() < 0.6 else 2):
        ym = y0 + (y1 - y0) * rng.uniform(0.4, 0.65)
        draw.line([(x0 - 3, ym + rng.uniform(-3, 3)), ((x0 + x1) / 2, ym + rng.uniform(-2, 2)),
                   (x1 + 3, ym + rng.uniform(-3, 3))], fill=235, width=ancho_linea, joint="curve")
