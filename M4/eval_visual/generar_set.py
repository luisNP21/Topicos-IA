"""Entry point de la tarea 3a: genera el set sintetico de imagenes.

Uso:
    python generar_set.py                      # todos los documentos y niveles
    python generar_set.py --limite 3           # muestra pequena para revisar
    python generar_set.py --niveles limpio rotacion

Lee el gold set, renderiza cada documento como paginas de imagen, aplica las
degradaciones de config.yaml y escribe las imagenes y un manifiesto con los
parametros y semillas. El gold sigue siendo el texto original: cada imagen
conserva el doc_id que asigna gold_loader.cargar_gold_set, el mismo que usa el harness.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import PIL
import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Reutiliza el cargador del harness para que los doc_id coincidan siempre.
for extra in (REPO_ROOT / "M2", REPO_ROOT / "M2" / "harness"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import log  # noqa: E402  (M2/common.py)
from gold_loader import cargar_gold_set  # noqa: E402  (M2/harness/gold_loader.py)

from degradaciones import DEGRADACIONES  # noqa: E402
from render import cargar_fuente, renderizar_documento, resolver_fuente  # noqa: E402


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        log(f"PROJECT_ROOT no definido; se usa la raiz del repo: {REPO_ROOT}")
        return REPO_ROOT
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def semilla_de(seed_global: int, nivel: str, doc_id: str, pagina: int) -> int:
    """Semilla estable por (nivel, documento, pagina), independiente del orden de ejecucion."""
    clave = f"{seed_global}|{nivel}|{doc_id}|{pagina}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(clave).digest()[:4], "big")


def run(cfg: dict, project_root: Path, limite: int | None = None, niveles: list[str] | None = None) -> dict:
    seed_global = cfg["proyecto"]["seed_global"]
    niveles_cfg = cfg["niveles"]
    seleccion = niveles or list(niveles_cfg)
    desconocidos = [n for n in seleccion if n not in niveles_cfg or n not in DEGRADACIONES]
    if desconocidos:
        raise ValueError(f"Niveles no definidos: {desconocidos}. Disponibles: {list(niveles_cfg)}")

    gold_path = REPO_ROOT / cfg["entrada"]["gold_set"]
    eval_set = cargar_gold_set(gold_path)
    if limite is not None:
        eval_set = eval_set[:limite]

    imagenes_dir = project_root / cfg["salida"]["imagenes_dir"]
    manifest_path = project_root / cfg["salida"]["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fuente_ruta = resolver_fuente(cfg["render"], REPO_ROOT)
    fuente = cargar_fuente(fuente_ruta, cfg["render"]["tamano_fuente"])
    log(f"Fuente: {fuente_ruta}")

    # Perfiles de render por nivel: los niveles sin "render" propio usan el comun.
    perfiles = {}
    fuentes_usadas = {"comun": Path(fuente_ruta).name}
    for nivel in seleccion:
        override = niveles_cfg[nivel].get("render")
        if not override:
            perfiles[nivel] = (cfg["render"], fuente)
            continue
        perfil = {**cfg["render"], **override}
        ruta = Path(perfil["fuente_archivo"])
        if not ruta.is_absolute():
            ruta = REPO_ROOT / ruta
        if not ruta.exists():
            raise FileNotFoundError(f"Fuente del nivel {nivel} no encontrada: {ruta}")
        perfiles[nivel] = (perfil, cargar_fuente(str(ruta), perfil["tamano_fuente"]))
        fuentes_usadas[nivel] = ruta.name
        log(f"Fuente de {nivel}: {ruta}")
    log(f"Documentos: {len(eval_set)} | niveles: {seleccion}")

    documentos = []
    n_imagenes = 0
    for i, rec in enumerate(eval_set, start=1):
        doc_id, texto = rec["doc_id"], rec["text"]
        entrada = {
            "doc_id": doc_id,
            "n_caracteres": len(texto),
            "sha256_texto": hashlib.sha256(texto.encode("utf-8")).hexdigest(),
            "niveles": {},
        }

        paginas_comunes = None
        for nivel in seleccion:
            ncfg = niveles_cfg[nivel]
            ext = ncfg["formato"]
            perfil, fuente_nivel = perfiles[nivel]
            if ncfg.get("render"):
                paginas = renderizar_documento(
                    texto, fuente_nivel, perfil,
                    semilla_pagina=lambda k, n=nivel, d=doc_id: semilla_de(seed_global, n + "_render", d, k),
                )
            else:
                if paginas_comunes is None:
                    paginas_comunes = renderizar_documento(texto, fuente_nivel, perfil)
                paginas = paginas_comunes
            carpeta = imagenes_dir / nivel
            carpeta.mkdir(parents=True, exist_ok=True)
            registros = []
            for k, pagina in enumerate(paginas, start=1):
                semilla = semilla_de(seed_global, nivel, doc_id, k)
                rng = np.random.default_rng(semilla)
                img, usados = DEGRADACIONES[nivel](pagina["imagen"], rng, ncfg["params"])
                nombre = f"{doc_id}_p{k}.{ext}"
                ruta = carpeta / nombre
                if ext == "jpg":
                    img.convert("RGB").save(ruta, format="JPEG", quality=ncfg["params"].get("calidad_jpeg", 90))
                else:
                    img.save(ruta, format="PNG")
                registros.append({
                    "pagina": k,
                    "archivo": f"{nivel}/{nombre}",
                    "semilla": semilla,
                    "params_usados": usados,
                    "tamano_px": list(img.size),
                })
                n_imagenes += 1
            entrada["niveles"][nivel] = {
                "n_paginas": len(paginas),
                "texto_por_pagina": [p["texto"] for p in paginas],
                "paginas": registros,
            }

        documentos.append(entrada)
        if i % 10 == 0 or i == len(eval_set):
            log(f"  {i}/{len(eval_set)} documentos")

    manifest = {
        "descripcion": (
            "Set sintetico de imagenes derivado de los documentos del gold set. "
            "El gold es el texto original; las degradaciones son sinteticas y no "
            "equivalen a escaneos o fotos reales."
        ),
        "gold_set": cfg["entrada"]["gold_set"],
        "seed_global": seed_global,
        "fuentes": fuentes_usadas,
        "versiones": {"Pillow": PIL.__version__, "numpy": np.__version__},
        "render": cfg["render"],
        "niveles_config": {n: niveles_cfg[n] for n in seleccion},
        "n_documentos": len(documentos),
        "n_imagenes": n_imagenes,
        "documentos": documentos,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log(f"Listo: {n_imagenes} imagenes en {imagenes_dir}")
    log(f"Manifiesto: {manifest_path}")
    return {
        "n_documentos": len(documentos),
        "n_imagenes": n_imagenes,
        "imagenes_dir": str(imagenes_dir),
        "manifest_path": str(manifest_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Genera el set sintetico de imagenes (M4, 3a)")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--limite", type=int, default=None, help="Solo los primeros N documentos")
    parser.add_argument("--niveles", nargs="+", default=None, help="Subconjunto de niveles")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stats = run(cfg, resolver_project_root(), limite=args.limite, niveles=args.niveles)
    print(stats)


if __name__ == "__main__":
    main()
