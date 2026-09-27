"""CLI wrapper para el harness de M2.

La logica de orquestacion real vive en `harness.py`, para mantener la
responsabilidad de evaluacion centralizada en ese modulo.
"""

import argparse
import os
import sys
from pathlib import Path

M2_ROOT = str(Path(__file__).resolve().parents[1])
if M2_ROOT in sys.path:
    sys.path.remove(M2_ROOT)
sys.path.insert(1, M2_ROOT)

import yaml
from dotenv import load_dotenv

from harness import run


def cargar_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "PROJECT_ROOT no esta definido. Crear un .env con:\n"
            "  PROJECT_ROOT=/ruta/a/TopicosIA/Proyecto-Salud"
        )
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def main():
    parser = argparse.ArgumentParser(description="Harness de evaluacion M2")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config.yaml")
    parser.add_argument(
        "--solo",
        choices=["exact", "semantica", "judge", "scorecard"],
        default=None,
        help="Correr solo una dimension (util para debugging).",
    )
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    cfg["solo"] = args.solo
    project_root = resolver_project_root()
    return run(cfg, project_root)


if __name__ == "__main__":
    main()