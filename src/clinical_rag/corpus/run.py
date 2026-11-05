"""
Entry point para correr el pipeline de corpus desde linea de comandos.
Mismo patron que run_harness.py de M2: codigo versionado en git, datos
y outputs en Drive via PROJECT_ROOT (.env). config.yaml y fuentes.yaml
se resuelven relativo a este archivo (no a PROJECT_ROOT) para que
funcione igual sin importar en que runtime se clone el repo.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from clinical_rag.paths import CONFIG_DIR, project_root as default_project_root, resolve_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "corpus.yaml"))
    parser.add_argument("--fuentes", default=str(CONFIG_DIR / "sources.yaml"))
    args = parser.parse_args()

    load_dotenv()
    project_root = default_project_root()

    cfg = {
        "fuentes_yaml": str(resolve_config(args.fuentes)),
        "config_yaml": str(resolve_config(args.config)),
    }

    from clinical_rag.corpus.pipeline import run
    stats = run(cfg=cfg, project_root=project_root)
    print(stats)


if __name__ == "__main__":
    main()