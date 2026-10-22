"""Entry point de la visión (M4 · Agustín).

Misma interfaz que las demás piezas del proyecto:

    run(cfg, project_root) -> dict

Qué hace:
1. Lee un manifiesto de casos con imagen (CSV/JSON) o genera una nota sintética
   de demo.
2. Transcribe cada imagen con el backend elegido (VLM, Docling o Tesseract).
3. Escribe el cache de transcripciones y, si se pide, el cache NER compatible
   con `M2/predictions/` (formato `{doc_id: [entidades]}`).
4. Devuelve un resumen con latencias y fallos. NO calcula el delta del harness:
   eso es la Tarea 5b de Isabella.

Uso:
    python run_vision.py --config config.yaml --demo
    python run_vision.py --config config.yaml --manifest M4/eval_set/casos.csv
    python run_vision.py --config config.yaml --imagen foto.jpg --backend ocr_docling
    python run_vision.py --config config.yaml --comparar --manifest ... --ner
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from backends import crear_backend  # noqa: E402
from cache import guardar_cache_transcripciones, guardar_json  # noqa: E402
from demo import crear_nota_sintetica  # noqa: E402
from manifiesto import (  # noqa: E402
    cargar_manifiesto,
    caso_desde_imagen,
    resolver_ruta_imagen,
)
from ner_bridge import generar_cache_ner  # noqa: E402
from transcripcion import cargar_imagen, transcribir, transcribir_auto  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def cargar_config(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if root and Path(root).exists():
        return Path(root)
    # M4/vision -> M4 -> raiz del repo
    return DIR.parents[1]


def _preparar_casos(
    cfg: dict,
    project_root: Path,
    *,
    manifest: str | None,
    demo: bool,
    imagen: str | None,
    pregunta: str | None,
    entidad: str | None,
) -> list[dict]:
    if imagen:
        casos = [
            dict(
                caso_desde_imagen(
                    imagen=imagen,
                    pregunta=pregunta or "Transcribe la nota clínica de la imagen.",
                    doc_id="imagen_01",
                    entidad=entidad,
                )
            )
        ]
        for caso in casos:
            caso["_imagen_resuelta"] = caso["imagen"]
        return casos

    if demo:
        destino = project_root / "M4" / "vision" / "data" / "demo" / "nota_sintetica.png"
        ruta = crear_nota_sintetica(destino)
        return [
            {
                "id": "demo_01",
                "pregunta": pregunta or "¿Qué dice la nota clínica de la imagen?",
                "imagen": str(ruta),
                "entidad": entidad,
                "esperado": None,
                "criterio": None,
                "tipo": "normal",
                "_imagen_resuelta": str(ruta),
            }
        ]

    ruta_manifiesto = manifest or (cfg.get("rutas") or {}).get("manifest")
    if not ruta_manifiesto:
        raise ValueError(
            "No hay manifiesto de casos. Pase --manifest, use --demo o configure "
            "rutas.manifest en M4/vision/config.yaml."
        )
    ruta_manifiesto = Path(ruta_manifiesto)
    if not ruta_manifiesto.is_absolute():
        ruta_manifiesto = project_root / ruta_manifiesto

    imagenes_dir = (cfg.get("rutas") or {}).get("imagenes_dir")
    if imagenes_dir and not Path(imagenes_dir).is_absolute():
        imagenes_dir = project_root / imagenes_dir

    casos = [dict(c) for c in cargar_manifiesto(ruta_manifiesto)]
    for caso in casos:
        caso["_imagen_resuelta"] = resolver_ruta_imagen(
            caso, imagenes_dir, project_root
        )
    return casos


def _transcribir_caso(caso: dict, cfg: dict, backend_nombre: str) -> dict:
    ruta = caso.get("_imagen_resuelta") or caso["imagen"]
    imagen = cargar_imagen(ruta)
    instruccion = (cfg.get("vision") or {}).get("instruccion")

    if backend_nombre == "auto":
        return transcribir_auto(
            imagen,
            cfg,
            instruccion=instruccion,
            ruta=ruta,
            doc_id=caso["id"],
        )

    backend = crear_backend(backend_nombre, cfg)
    return transcribir(
        imagen,
        backend=backend,
        instruccion=instruccion,
        ruta=ruta,
        doc_id=caso["id"],
    )


def _resumen_backend(
    resultados: list[dict],
    backend_nombre: str,
    cfg: dict,
    cache_path: Path,
    ner_resumen: dict | None,
    salida_path: Path,
) -> dict:
    n = len(resultados)
    latencias = [float(t.get("segundos") or 0.0) for t in resultados]
    resultados_resumen = []
    for t in resultados:
        resultados_resumen.append(
            {
                "id": t.get("doc_id"),
                "imagen": t.get("imagen"),
                "backend": t.get("backend"),
                "segundos": t.get("segundos"),
                "valvula": t.get("valvula"),
                "error": t.get("error"),
                "sha256": t.get("sha256"),
                "texto_preview": str(t.get("texto") or "")[:300],
            }
        )

    return {
        "modulo": "vision",
        "responsable": "agustin",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "backend": backend_nombre,
        "modelo": (
            (cfg.get("vlm") or {}).get("model_id")
            if backend_nombre == "vlm"
            else (cfg.get("ocr") or {}).get(backend_nombre.replace("ocr_", ""), {})
        ),
        "n_casos": n,
        "n_ok": sum(1 for t in resultados if t.get("error") is None),
        "n_error": sum(1 for t in resultados if t.get("error") is not None),
        "n_valvula": sum(1 for t in resultados if t.get("valvula")),
        "latencia_media_s": round(sum(latencias) / n, 3) if n else None,
        "cache_transcripciones": str(cache_path),
        "cache_ner": ner_resumen,
        "salida": str(salida_path),
        "resultados": resultados_resumen,
    }


def run(
    cfg: dict,
    project_root: str | Path,
    *,
    backend: str | None = None,
    manifest: str | None = None,
    demo: bool = False,
    imagen: str | None = None,
    pregunta: str | None = None,
    entidad: str | None = None,
    comparar: bool = False,
) -> dict:
    """Ejecuta la transcripción sobre los casos y escribe los caches."""
    project_root = Path(project_root)
    rutas = cfg.get("rutas") or {}
    vision_cfg = cfg.get("vision") or {}

    if comparar:
        backends = vision_cfg.get("comparar", ["vlm", "ocr_docling"])
    elif backend:
        backends = [backend]
    else:
        backends = [vision_cfg.get("backend", "vlm")]

    casos = _preparar_casos(
        cfg,
        project_root,
        manifest=manifest,
        demo=demo,
        imagen=imagen,
        pregunta=pregunta,
        entidad=entidad,
    )
    log(f"Casos: {len(casos)} | backends: {', '.join(backends)} | raíz: {project_root}")

    resumenes = []
    for backend_nombre in backends:
        log(f"Transcribiendo con backend={backend_nombre} ...")
        resultados = []
        for i, caso in enumerate(casos, 1):
            t = _transcribir_caso(caso, cfg, backend_nombre)
            resultados.append(t)
            estado = "ERROR" if t.get("error") else ("VÁLVULA" if t.get("valvula") else "ok")
            log(
                f"  [{i}/{len(casos)}] {caso['id']} -> {estado} "
                f"({t.get('segundos')}s, {t.get('backend')})"
            )
            if t.get("error"):
                log(f"      error: {t['error'][:180]}")

        plantilla_cache = rutas.get(
            "transcripciones_cache", "M4/vision/predictions/transcripciones_{backend}.json"
        )
        cache_path = project_root / plantilla_cache.format(backend=backend_nombre)
        guardar_cache_transcripciones(resultados, cache_path)
        log(f"Cache de transcripciones: {cache_path}")

        ner_resumen = None
        if (cfg.get("ner") or {}).get("enabled"):
            plantilla_ner = rutas.get(
                "ner_cache", "M2/predictions/vision_{backend}.json"
            )
            ner_path = project_root / plantilla_ner.format(backend=backend_nombre)
            ner_resumen = generar_cache_ner(
                {t["doc_id"]: t for t in resultados},
                cfg["ner"],
                project_root,
                ner_path,
            )
            log(f"Cache NER para el harness: {ner_resumen['output_path']}")

        plantilla_salida = rutas.get(
            "salida", "M4/vision/outputs/resultado_vision_{backend}.json"
        )
        salida_path = project_root / plantilla_salida.format(backend=backend_nombre)
        resumen = _resumen_backend(
            resultados,
            backend_nombre,
            cfg,
            cache_path,
            ner_resumen,
            salida_path,
        )
        guardar_json(resumen, salida_path)
        log(f"Resumen: {salida_path}")
        resumenes.append(resumen)

    if len(resumenes) == 1:
        return resumenes[0]

    # Comparación lado a lado (VLM vs. OCR). No calcula CER/WER ni F1: eso es
    # Tarea 5a/5b. Solo deja los números de operación de cada backend.
    claves = (
        "backend",
        "n_ok",
        "n_error",
        "n_valvula",
        "latencia_media_s",
        "cache_transcripciones",
        "cache_ner",
    )
    comparacion = {
        "modulo": "vision",
        "responsable": "agustin",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_casos": len(casos),
        "backends": [{k: r.get(k) for k in claves} for r in resumenes],
    }
    ruta_comparacion = project_root / rutas.get(
        "comparacion", "M4/vision/outputs/comparacion_backends.json"
    )
    guardar_json(comparacion, ruta_comparacion)
    log(f"Comparación de backends: {ruta_comparacion}")
    return {
        "modulo": "vision",
        "archivo_comparacion": str(ruta_comparacion),
        "comparacion": resumenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visión M4 — transcripción de notas clínicas")
    parser.add_argument("--config", default=str(DIR / "config.yaml"))
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="vlm | ocr_docling | ocr_tesseract | auto | fake",
    )
    parser.add_argument("--manifest", default=None, help="CSV/JSON de casos con imagen")
    parser.add_argument("--imagen", default=None, help="Una sola imagen (modo rápido)")
    parser.add_argument("--pregunta", default=None)
    parser.add_argument("--entidad", default=None, help="Entidad gold opcional del caso")
    parser.add_argument("--demo", action="store_true", help="Genera una nota sintética")
    parser.add_argument("--comparar", action="store_true", help="Corre todos los backends de vision.comparar")
    parser.add_argument("--ner", action="store_true", help="Fuerza el puente NER -> cache M2")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    if args.ner:
        cfg.setdefault("ner", {})["enabled"] = True

    project_root = (
        Path(args.project_root) if args.project_root else resolver_project_root()
    )
    resultado = run(
        cfg,
        project_root,
        backend=args.backend,
        manifest=args.manifest,
        demo=args.demo,
        imagen=args.imagen,
        pregunta=args.pregunta,
        entidad=args.entidad,
        comparar=args.comparar,
    )
    print(yaml.safe_dump(resultado, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
