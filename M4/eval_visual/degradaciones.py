"""Degradaciones del set sintetico (tarea 3a).

Cada funcion recibe (imagen PIL, rng numpy, params) y devuelve (imagen, params_usados).
Todo el azar sale del rng recibido, que se siembra por (nivel, doc_id, pagina), asi
que regenerar el set produce exactamente las mismas imagenes.

Son aproximaciones sinteticas: no equivalen a escaneos o fotos reales.
"""

import io

import numpy as np
from PIL import Image, ImageFilter


def limpio(img: Image.Image, rng: np.random.Generator, params: dict):
    return img, {}


def jpeg_bajo(img: Image.Image, rng: np.random.Generator, params: dict):
    """Baja la resolucion y comprime con JPEG de calidad baja."""
    escala = params["escala"]
    nuevo = (max(1, int(img.width * escala)), max(1, int(img.height * escala)))
    chica = img.resize(nuevo, Image.BILINEAR)
    buf = io.BytesIO()
    chica.save(buf, format="JPEG", quality=params["calidad_jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {
        "escala": escala,
        "calidad_jpeg": params["calidad_jpeg"],
        "tamano_resultante": list(nuevo),
    }


def rotacion(img: Image.Image, rng: np.random.Generator, params: dict):
    """Rota unos grados, con signo aleatorio, rellenando de blanco."""
    magnitud = float(rng.uniform(params["angulo_min"], params["angulo_max"]))
    signo = 1.0 if rng.random() < 0.5 else -1.0
    angulo = signo * magnitud
    rotada = img.rotate(angulo, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return rotada, {"angulo_grados": round(angulo, 3)}


def desenfoque(img: Image.Image, rng: np.random.Generator, params: dict):
    return img.filter(ImageFilter.GaussianBlur(radius=params["radio"])), {"radio": params["radio"]}


def sombra_celular(img: Image.Image, rng: np.random.Generator, params: dict):
    """Ilumina de forma desigual (sombra diagonal + vineta) y agrega ruido de sensor."""
    arr = np.asarray(img, dtype=np.float32)
    alto, ancho = arr.shape[:2]
    ys, xs = np.mgrid[0:alto, 0:ancho].astype(np.float32)

    # Sombra: un degradado lineal en una direccion aleatoria.
    theta = float(rng.uniform(0, 2 * np.pi))
    proyeccion = np.cos(theta) * xs / ancho + np.sin(theta) * ys / alto
    proyeccion = (proyeccion - proyeccion.min()) / (proyeccion.max() - proyeccion.min() + 1e-6)
    oscuridad = float(rng.uniform(0.5, 1.0)) * params["oscuridad_max"]
    sombra = 1.0 - oscuridad * proyeccion

    # Vineta: oscurece hacia los bordes.
    cx, cy = ancho / 2, alto / 2
    dist = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) / np.sqrt(2)
    vineta = 1.0 - params["vineta"] * dist ** 2

    luz = (sombra * vineta)[..., None]
    ruido = rng.normal(0.0, params["ruido_sigma"], size=arr.shape).astype(np.float32)
    resultado = np.clip(arr * luz + ruido, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(resultado).save(buf, format="JPEG", quality=params["calidad_jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {
        "angulo_sombra_rad": round(theta, 4),
        "oscuridad": round(oscuridad, 4),
        "vineta": params["vineta"],
        "ruido_sigma": params["ruido_sigma"],
        "calidad_jpeg": params["calidad_jpeg"],
    }


def manuscrito_sintetico(img: Image.Image, rng: np.random.Generator, params: dict):
    """La imagen ya sale con fuente y jitter manuscritos desde render.py; aqui no se degrada mas."""
    return img, {}


DEGRADACIONES = {
    "limpio": limpio,
    "manuscrito_sintetico": manuscrito_sintetico,
    "jpeg_bajo": jpeg_bajo,
    "rotacion": rotacion,
    "desenfoque": desenfoque,
    "sombra_celular": sombra_celular,
}
