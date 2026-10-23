"""Pieza de visión del M4 (Agustín).

Punto de entrada: `run_vision.run(cfg, project_root)`.
Interfaces del laboratorio S13: `ver()` y `sistema_multimodal()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from integracion import crear_responder_m3, sistema_multimodal  # noqa: E402
from transcripcion import transcribir, ver  # noqa: E402

__all__ = [
    "ver",
    "transcribir",
    "sistema_multimodal",
    "crear_responder_m3",
]
