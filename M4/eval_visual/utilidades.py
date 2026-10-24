"""Utilidades compartidas del set sintetico (tarea 3a).

- `Pagina`: estado de una pagina mientras pasa por los efectos. Ademas de la imagen
  guarda puntos de muestra dentro de cada palabra; los efectos geometricos mueven
  esos puntos y los que tapan texto los marcan como perdidos. Asi se sabe que
  palabras del gold siguen visibles en la imagen final.
- Remapeo bilineal, campos de desplazamiento suaves, homografias y muestreo de
  parametros, implementados con numpy y Pillow (sin OpenCV).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def semilla(*partes) -> int:
    """Semilla estable a partir de cualquier combinacion de valores."""
    clave = "|".join(str(p) for p in partes).encode("utf-8")
    return int.from_bytes(hashlib.sha256(clave).digest()[:4], "big")


def muestrear(valor, rng: np.random.Generator):
    """[a, b] numerico -> uniforme en [a, b]; lista de otro tipo -> eleccion; escalar -> igual."""
    if isinstance(valor, list) and len(valor) == 2 and all(isinstance(v, (int, float)) for v in valor):
        a, b = float(valor[0]), float(valor[1])
        return a if a == b else float(rng.uniform(a, b))
    if isinstance(valor, list) and valor:
        return valor[int(rng.integers(len(valor)))]
    return valor


def redondear(obj):
    """Redondea floats para que el manifiesto sea legible."""
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: redondear(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redondear(v) for v in obj]
    if isinstance(obj, np.generic):
        return redondear(obj.item())
    return obj


# --------------------------------------------------------------------------- #
# Estado de una pagina
# --------------------------------------------------------------------------- #

@dataclass
class Pagina:
    img: np.ndarray                 # float32 (H, W, 3) en 0..255
    pts: np.ndarray                 # float32 (N, 2): puntos de muestra (x, y)
    pt_token: np.ndarray            # int32 (N,): token global al que pertenece cada punto
    vivo: np.ndarray                # bool (N,): el punto sigue visible
    alto_linea: float
    color_papel: tuple = (255, 255, 255)
    extra: list = field(default_factory=list)   # texto que aparece en la imagen y no esta en el gold

    @property
    def alto(self) -> int:
        return self.img.shape[0]

    @property
    def ancho(self) -> int:
        return self.img.shape[1]

    def a_pil(self) -> Image.Image:
        return Image.fromarray(np.clip(self.img, 0, 255).astype(np.uint8), "RGB")

    def desde_pil(self, img: Image.Image) -> None:
        self.img = np.asarray(img.convert("RGB"), dtype=np.float32).copy()

    def descartar_fuera_de_lienzo(self) -> None:
        x, y = self.pts[:, 0], self.pts[:, 1]
        dentro = (x >= 0) & (x <= self.ancho - 1) & (y >= 0) & (y <= self.alto - 1)
        self.vivo &= dentro

    def ocultar_donde(self, mascara: np.ndarray) -> int:
        """Marca como perdidos los puntos vivos donde `mascara` (bool HxW) es True."""
        if not self.vivo.any():
            return 0
        xi = np.clip(np.round(self.pts[:, 0]).astype(int), 0, self.ancho - 1)
        yi = np.clip(np.round(self.pts[:, 1]).astype(int), 0, self.alto - 1)
        tapados = self.vivo & mascara[yi, xi]
        self.vivo &= ~tapados
        return int(tapados.sum())

    def caja_texto(self) -> tuple[float, float, float, float] | None:
        """Caja que encierra los puntos vivos (x0, y0, x1, y1)."""
        if not self.vivo.any():
            return None
        p = self.pts[self.vivo]
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


# --------------------------------------------------------------------------- #
# Remapeo y campos de desplazamiento
# --------------------------------------------------------------------------- #

def remap_bilineal(arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, relleno) -> np.ndarray:
    """out[y, x] = arr[map_y[y, x], map_x[y, x]] con interpolacion bilineal."""
    alto, ancho = arr.shape[:2]
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    wx = (map_x - x0).astype(np.float32)
    wy = (map_y - y0).astype(np.float32)
    validos = (x0 >= 0) & (x0 < ancho - 1) & (y0 >= 0) & (y0 < alto - 1)
    x0 = np.clip(x0, 0, ancho - 2)
    y0 = np.clip(y0, 0, alto - 2)
    if arr.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    a = arr[y0, x0]
    b = arr[y0, x0 + 1]
    c = arr[y0 + 1, x0]
    d = arr[y0 + 1, x0 + 1]
    out = (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy).astype(np.float32)
    out[~validos] = relleno
    return out


def campo_suave(rng: np.random.Generator, alto: int, ancho: int, celda: float, amplitud: float) -> np.ndarray:
    """Campo aleatorio suave (H, W): ruido en una grilla gruesa interpolado bicubico."""
    gh = max(2, int(np.ceil(alto / celda)) + 2)
    gw = max(2, int(np.ceil(ancho / celda)) + 2)
    grilla = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    campo = np.asarray(Image.fromarray(grilla, "F").resize((ancho, alto), Image.BICUBIC), dtype=np.float32)
    return campo * amplitud


def aplicar_desplazamiento(p: Pagina, dx: np.ndarray, dy: np.ndarray) -> None:
    """Deforma la imagen con out(q) = in(q + d(q)) y mueve los puntos de muestra.

    Para un punto de entrada p se busca q con q + d(q) = p por iteracion de punto fijo,
    que converge porque los desplazamientos son suaves y pequenos.
    """
    alto, ancho = p.alto, p.ancho
    ys, xs = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    p.img = remap_bilineal(p.img, xs + dx, ys + dy, relleno=np.array(p.color_papel, np.float32))
    p.pts = mover_puntos_inverso(p.pts, dx, dy)
    p.descartar_fuera_de_lienzo()


def mover_puntos_inverso(pts: np.ndarray, dx: np.ndarray, dy: np.ndarray, iteraciones: int = 4) -> np.ndarray:
    alto, ancho = dx.shape
    q = pts.astype(np.float32).copy()
    for _ in range(iteraciones):
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, ancho - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, alto - 1)
        q = np.stack([pts[:, 0] - dx[yi, xi], pts[:, 1] - dy[yi, xi]], axis=1).astype(np.float32)
    return q


# --------------------------------------------------------------------------- #
# Homografias
# --------------------------------------------------------------------------- #

def homografia(origen: list, destino: list) -> np.ndarray:
    """Coeficientes (a..h) que llevan cada punto de `origen` a `destino`."""
    filas, b = [], []
    for (x, y), (u, v) in zip(origen, destino):
        filas.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        filas.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    return np.linalg.solve(np.array(filas, dtype=np.float64), np.array(b, dtype=np.float64))


def aplicar_homografia(coef: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f, g, h = coef
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    den = g * x + h * y + 1.0
    return np.stack([(a * x + b * y + c) / den, (d * x + e * y + f) / den], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Puntos de muestra
# --------------------------------------------------------------------------- #

def puntos_de_cajas(cajas: list[tuple], nx: int = 5, ny: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Grilla nx x ny dentro de cada caja (token, x0, y0, x1, y1), con un margen del 15 %."""
    if not cajas:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    fx = (np.arange(nx) + 0.5) / nx
    fy = (np.arange(ny) + 0.5) / ny
    pts, toks = [], []
    for tok, x0, y0, x1, y1 in cajas:
        mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        xs = x0 + mx + fx * (x1 - x0 - 2 * mx)
        ys = y0 + my + fy * (y1 - y0 - 2 * my)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        toks.append(np.full(gx.size, tok, np.int32))
    return np.concatenate(pts).astype(np.float32), np.concatenate(toks)
