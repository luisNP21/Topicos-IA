"""
run_tools.py
Entry point para correr la tool de normalizacion desde linea de comandos.
Mismo patron que run_corpus.py de M3/corpus: codigo versionado en git,
datos y outputs en Drive via PROJECT_ROOT (.env). config.yaml se resuelve
relativo a este archivo para que funcione igual sin importar el runtime.

Uso:
    python run_tools.py --config config.yaml
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from config_utils import cargar_config
from tool_normalizacion import run

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Tool de normalizacion terminologica -- M3, Luis"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al config.yaml (relativa a este script o absoluta)",
    )
    args = parser.parse_args()

    load_dotenv()
    project_root = os.environ.get("PROJECT_ROOT", "")

    config_path = SCRIPT_DIR / args.config
    yaml_cfg = cargar_config(str(config_path))

    cfg = {
        "config_yaml": str(config_path),
        "tool_normalizacion": yaml_cfg["tool_normalizacion"],
    }

    stats = run(cfg=cfg, project_root=project_root)
    print(stats)


if __name__ == "__main__":
    main()
