"""
Entry point para correr el pipeline de corpus desde linea de comandos.
Mismo patron que run_harness.py de M2: codigo versionado en git, datos
y outputs en Drive via PROJECT_ROOT (.env). config.yaml y fuentes.yaml
se resuelven relativo a este archivo (no a PROJECT_ROOT) para que
funcione igual sin importar en que runtime se clone el repo.
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from pipeline_corpus import run

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fuentes", default="fuentes.yaml")
    args = parser.parse_args()

    load_dotenv()
    project_root = os.environ["PROJECT_ROOT"]

    cfg = {
        "fuentes_yaml": str(SCRIPT_DIR / args.fuentes),
        "config_yaml": str(SCRIPT_DIR / args.config),
    }

    stats = run(cfg=cfg, project_root=project_root)
    print(stats)


if __name__ == "__main__":
    main()