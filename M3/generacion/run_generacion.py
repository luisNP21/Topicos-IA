"""
Entry point de la pieza de generacion RAG  (Agustin / M3).

Mismo patron que M2: `run(cfg, project_root) -> dict`. Con
`generacion.usar_mocks: true` corre de punta a punta SIN depender de las
piezas de Pau (retrieval) ni de Luis (normalizacion).

Uso:
    python run_generacion.py --config config.yaml

Requiere .env con (opcional si usar_mocks=true y no se genera con LLM... pero
la generacion SI llama al LLM, asi que en la practica se necesita):
    PROJECT_ROOT=/ruta/a/proyecto
    GROQ_API_KEY=gsk_...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from contratos import NormalizeFn, RetrieveFn  # noqa: E402
from generacion import GroqGenerator, contextos_para_ragas, generar_respuesta  # noqa: E402
from orquestacion import resolver_query  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def cargar_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if root and Path(root).exists():
        return Path(root)
    # Respaldo: subir desde M3/generacion/ hasta la raiz del repo.
    return Path(__file__).resolve().parents[2]


def _construir_piezas(cfg: dict) -> tuple[RetrieveFn, NormalizeFn]:
    """
    Devuelve (retrieve_fn, normalizar_fn). Con usar_mocks=true usa los mocks;
    de lo contrario intenta importar las piezas reales del equipo y falla con
    un mensaje claro si todavia no existen.
    """
    if cfg["generacion"].get("usar_mocks", True):
        from mocks import crear_normalizador_mock, crear_retriever_mock
        _log("Usando MOCKS de retrieval (Pau) y normalizacion (Luis).")
        return crear_retriever_mock(), crear_normalizador_mock()

    # Cuando Pau y Luis publiquen sus modulos, se importan aqui.
    try:
        from retrieval import retrieve_advanced          # type: ignore  # pieza de Pau
        from normalizacion import normalizar_entidad      # type: ignore  # pieza de Luis
    except ImportError as e:
        raise RuntimeError(
            "usar_mocks=false pero no se pudieron importar las piezas reales "
            f"(retrieval / normalizacion): {e}"
        )
    return retrieve_advanced, normalizar_entidad


def run(cfg: dict, project_root: Path) -> dict:
    fijar_seeds = cfg["proyecto"].get("seed_global", 42)
    random.seed(fijar_seeds)

    gen_cfg = cfg["generacion"]
    orq_cfg = cfg["orquestacion"]
    rutas = cfg["rutas"]

    outputs_dir = project_root / rutas["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    entidades_path = project_root / rutas["entidades"]
    if not entidades_path.exists():
        repo_fallback = Path(__file__).resolve().parents[2] / rutas["entidades"]
        modulo_fallback = Path(__file__).resolve().parent / "entidades_ejemplo.json"
        if repo_fallback.exists():
            entidades_path = repo_fallback
        elif modulo_fallback.exists():
            entidades_path = modulo_fallback
    with open(entidades_path, "r", encoding="utf-8") as f:
        entidades = json.load(f)
    _log(f"Entidades de prueba: {len(entidades)} (desde {entidades_path})")

    retrieve_fn, normalizar_fn = _construir_piezas(cfg)

    generar_fn = GroqGenerator(
        model=gen_cfg["modelo"], temperature=gen_cfg["temperature"],
        max_tokens=gen_cfg["max_tokens"],
        pausa_entre_llamadas=gen_cfg.get("pausa_entre_llamadas", 3.0),
    )

    umbral = orq_cfg["umbral_score"]
    k = orq_cfg["k_fragments"]

    resultados = []
    _log(f"Corriendo generacion sobre {len(entidades)} entidades "
         f"(umbral={umbral}, k={k}, modelo={gen_cfg['modelo']})...")
    for i, entidad in enumerate(entidades, 1):
        q = resolver_query(entidad, retrieve_fn=retrieve_fn, normalizar_fn=normalizar_fn,
                           umbral=umbral, k=k)
        resp = generar_respuesta(q["query_final"], q["fragments"], generar_fn=generar_fn)
        _log(f"  [{i}/{len(entidades)}] {entidad!r} -> query={q['query_final']!r} | "
             f"tool={q['tool_invoked']} | fallback={resp['fallback_used']} | "
             f"fuentes={len(resp['sources_used'])}")
        resultados.append({
            "entidad": entidad,
            "query_final": q["query_final"],
            "tool_invoked": q["tool_invoked"],
            "tool_reason": q["tool_reason"],
            "fragments": [f["chunk_id"] for f in q["fragments"]],
            "scores": [f.get("score") for f in q["fragments"]],
            # `contexts` es lo que RAGAS consume (retrieval-side, dueno: Luis).
            "contexts": contextos_para_ragas(q["fragments"]),
            "answer": resp["answer"],
            "sources_used": resp["sources_used"],
            "fallback_used": resp["fallback_used"],
        })

    n = len(resultados) or 1
    resumen = {
        "modulo": "generacion_rag",
        "rol": "agustin",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "proveedor": gen_cfg["proveedor"],
        "modelo_generador": gen_cfg["modelo"],
        "usar_mocks": gen_cfg.get("usar_mocks", True),
        "orquestacion": {"umbral_score": umbral, "k_fragments": k},
        "n_entidades": len(resultados),
        "metricas_generacion": {
            "tool_invocada_pct": sum(r["tool_invoked"] for r in resultados) / n,
            "fallback_pct": sum(r["fallback_used"] for r in resultados) / n,
            "con_fuentes_pct": sum(bool(r["sources_used"]) for r in resultados) / n,
        },
        "resultados": resultados,
    }

    salida = outputs_dir / "resultado_generacion.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    _log(f"Guardado en: {salida}")
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generacion RAG (M3) -- Agustin")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    project_root = resolver_project_root()
    _log(f"PROJECT_ROOT: {project_root}")
    run(cfg, project_root)


if __name__ == "__main__":
    main()
