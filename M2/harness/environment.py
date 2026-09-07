"""
Seccion 0 del harness: instalacion/verificacion de dependencias,
fijado de seeds, y registro de versiones para reproducibilidad.
"""

import subprocess
import sys

import numpy as np


def asegurar_paquetes(pkgs: list[str]):
    """
    Instala paquetes faltantes. En produccion (fuera de notebook), preferir
    'pip install -r requirements.txt' antes de correr el harness -- esta
    funcion es un respaldo, no el mecanismo principal de instalacion.
    """
    for pkg in pkgs:
        modulo = pkg.replace("-", "_")
        try:
            __import__(modulo)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg])


def fijar_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def imprimir_versiones() -> dict:
    """Registra versiones de librerias clave -- requisito de reproducibilidad de la rubrica."""
    versiones = {}
    try:
        import torch
        versiones["torch"] = torch.__version__
    except ImportError:
        pass
    for nombre_modulo, alias in [("transformers", "transformers"),
                                   ("datasets", "datasets"),
                                   ("peft", "peft"),
                                   ("numpy", "numpy")]:
        try:
            mod = __import__(nombre_modulo)
            versiones[alias] = mod.__version__
        except ImportError:
            pass

    print("Versiones de librerias:")
    for k, v in versiones.items():
        print(f"  {k:<14}: {v}")
    return versiones


def preparar_entorno(cfg: dict) -> dict:
    """Punto de entrada unico para la Seccion 0: instala, fija seeds, registra versiones."""
    asegurar_paquetes(cfg["paquetes_requeridos"])
    fijar_seeds(cfg["proyecto"]["seed_global"])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seeds fijadas ({cfg['proyecto']['seed_global']}). Device: {device}")

    versiones = imprimir_versiones()
    return {"device": device, "versiones": versiones}