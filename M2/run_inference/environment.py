import subprocess
import sys

import numpy as np


def asegurar_paquetes(pkgs: list[str]):
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
    versiones = {}
    try:
        import torch
        versiones["torch"] = torch.__version__
    except ImportError:
        pass
    for nombre_modulo, alias in [("transformers", "transformers"), ("datasets", "datasets"), ("peft", "peft"), ("numpy", "numpy")]:
        try:
            mod = __import__(nombre_modulo)
            versiones[alias] = mod.__version__
        except ImportError:
            pass
    return versiones


def preparar_entorno(cfg: dict) -> dict:
    asegurar_paquetes(cfg.get("paquetes_requeridos", []))
    fijar_seeds(cfg.get("seed_global", 42))
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"device": device, "versiones": imprimir_versiones()}
