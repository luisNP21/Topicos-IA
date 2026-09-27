from __future__ import annotations

from pathlib import Path
from typing import Callable

from common import fijar_seeds, log

Sistema = Callable[[str], set[str]]


def harness(
    eval_set: list[dict],
    sistema: Sistema,
    cfg: dict,
    project_root: Path,
    solo: str | None = None,
) -> dict[str, float]:
    """Ejecuta las dimensiones usando el eval_set y sistema proporcionados."""
    fijar_seeds(cfg["proyecto"]["seed_global"])
    log(f"Seed global fijada: {cfg['proyecto']['seed_global']}")
    log(f"PROJECT_ROOT: {project_root}")

    import metrics_exact
    log("DIMENSION 1 -- Exact-match")
    resultado_exact = metrics_exact.run(cfg, project_root, eval_set, sistema)
    metricas = {
        f"exact_{key}": float(value)
        for key, value in resultado_exact["metrics"].items()
    }

    if solo in (None, "semantica"):
        import metrics_semantic
        log("DIMENSION 1b -- Similitud semantica")
        resultado_semantica = metrics_semantic.run(
            cfg, project_root, dim1_result=resultado_exact
        )
        for key, value in resultado_semantica["metrics"].items():
            if isinstance(value, (int, float)):
                metricas[f"semantic_{key}"] = float(value)

    if solo in (None, "judge"):
        import metrics_judge
        log("DIMENSION 3 -- LLM-as-judge")
        resultado_juez = metrics_judge.run(
            cfg,
            project_root,
            eval_set=eval_set,
            dim1_result=resultado_exact,
        )
        for key, value in resultado_juez["metrics"].items():
            if isinstance(value, (int, float)):
                metricas[f"judge_{key}"] = float(value)

    log("Harness completado.")
    return metricas
