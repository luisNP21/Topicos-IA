"""
Seccion 6 del harness: Dimension 1b -- similitud semantica por embeddings,
calibrada contra los casos boundary de la Dimension 1.
"""

import json
import random
import re
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from common import strip_chunk_suffix
from gold_loader import resolver_rutas


class EmbeddingSimilarity:
    """Encapsula el modelo de embeddings y el cache -- evita variables de modulo sueltas."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._cache = {}

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        nuevos = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if nuevos:
            vecs = self.model.encode(nuevos, batch_size=64, show_progress_bar=False,
                                      normalize_embeddings=True)
            for t, v in zip(nuevos, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def sim_matrix(self, preds: list[str], golds: list[str]) -> np.ndarray:
        P = self._get_embeddings(list(preds))
        G = self._get_embeddings(list(golds))
        return P @ G.T

    def sims_de_pares(self, pares: list[tuple]) -> np.ndarray:
        """Similitud por pares alineados (para calibracion), en un solo batch."""
        if not pares:
            return np.array([])
        A = [a for a, _ in pares]
        B = [b for _, b in pares]
        EA = self._get_embeddings(A)
        EB = self._get_embeddings(B)
        return np.sum(EA * EB, axis=1)


def match_hungaro(S: np.ndarray, threshold: float) -> list[tuple]:
    """Asignacion bipartita optima uno a uno que maximiza la similitud total."""
    PENALIZACION = -1e6
    if S.size == 0:
        return []
    S_pen = np.where(S >= threshold, S, PENALIZACION)
    filas, cols = linear_sum_assignment(-S_pen)
    return [(int(i), int(j), float(S[i, j])) for i, j in zip(filas, cols) if S[i, j] >= threshold]


def calibrar_umbral(casos_boundary: list[dict], true_by_doc: dict, pred_by_doc: dict,
                     emb_sim: EmbeddingSimilarity, n_pares_negativos: int, seed: int) -> dict:
    """Umbral que maximiza el indice de Youden sobre positivos (boundary) vs negativos (cross-doc)."""
    pares_pos = [(c["predicho"], c["gold"]) for c in casos_boundary if c["predicho"] != c["gold"]]

    doc_ids_orden = sorted(true_by_doc)
    rng = random.Random(seed)
    pares_neg = []
    for _ in range(n_pares_negativos):
        d1, d2 = rng.sample(doc_ids_orden, 2)
        p_pool = sorted(pred_by_doc.get(d1, set()))
        g_pool = sorted(true_by_doc.get(d2, set()))
        if not p_pool or not g_pool:
            continue
        a, b = rng.choice(p_pool), rng.choice(g_pool)
        if a != b:
            pares_neg.append((a, b))
    pares_neg = list(dict.fromkeys(pares_neg))

    sims_pos = emb_sim.sims_de_pares(pares_pos)
    sims_neg = emb_sim.sims_de_pares(pares_neg)

    grid = np.round(np.arange(0.30, 0.991, 0.01), 3)
    tabla_youden = [
        (t, float((sims_pos >= t).mean()), float((sims_neg >= t).mean()))
        for t in grid
    ]
    t_opt, tpr_opt, fpr_opt = max(tabla_youden, key=lambda r: r[1] - r[2])
    threshold = float(round(t_opt, 2))

    print(f"Umbral calibrado: {threshold}  (TPR={tpr_opt:.1%}, FPR={fpr_opt:.1%})")

    return {
        "threshold": threshold,
        "tpr_positivos": round(tpr_opt, 4),
        "fpr_negativos": round(fpr_opt, 4),
        "n_pares_pos": len(pares_pos),
        "n_pares_neg": len(pares_neg),
    }


def semantic_prf1_by_doc(true_by_doc: dict, pred_by_doc: dict, emb_sim: EmbeddingSimilarity,
                          threshold: float) -> tuple:
    tp = fp = fn = 0
    soft_tp = 0.0
    detalle, no_emparejados = {}, {}

    for doc_id in sorted(set(true_by_doc) | set(pred_by_doc)):
        orig_id = strip_chunk_suffix(doc_id)
        golds = sorted(true_by_doc.get(doc_id, set()))
        preds = sorted(pred_by_doc.get(doc_id, set()))

        if not golds or not preds:
            fp += len(preds)
            fn += len(golds)
            no_emparejados[orig_id] = {"fp": preds, "fn": golds}
            continue

        S = emb_sim.sim_matrix(preds, golds)
        pares = match_hungaro(S, threshold)

        tp += len(pares)
        fp += len(preds) - len(pares)
        fn += len(golds) - len(pares)
        soft_tp += sum(s for _, _, s in pares)

        usados_p = {i for i, _, _ in pares}
        usados_g = {j for _, j, _ in pares}
        detalle[orig_id] = [
            {"pred": preds[i], "gold": golds[j], "sim": round(s, 4), "exacto": preds[i] == golds[j]}
            for i, j, s in sorted(pares, key=lambda x: -x[2])
        ]
        no_emparejados[orig_id] = {
            "fp": [p for i, p in enumerate(preds) if i not in usados_p],
            "fn": [g for j, g in enumerate(golds) if j not in usados_g],
        }

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    soft_p = soft_tp / (tp + fp) if (tp + fp) else 0.0
    soft_r = soft_tp / (tp + fn) if (tp + fn) else 0.0
    soft_f1 = 2 * soft_p * soft_r / (soft_p + soft_r) if (soft_p + soft_r) else 0.0

    metrics = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
               "soft_precision": soft_p, "soft_recall": soft_r, "soft_f1": soft_f1}
    return metrics, detalle, no_emparejados


def _tokens(s: str) -> set:
    return set(re.findall(r"\w+", s.lower()))


def _normalizar_sin_puntuacion(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def categorizar(pred: str, gold: str) -> str:
    if pred == gold:
        return "exacto"
    if _normalizar_sin_puntuacion(pred) == _normalizar_sin_puntuacion(gold):
        return "variante de puntuacion/formato"
    if pred in gold or gold in pred:
        return "boundary (uno contiene al otro)"
    if _tokens(pred) & _tokens(gold):
        return "solapamiento parcial de palabras"
    return "sinonimo/parafrasis (sin solapamiento lexico)"


def analizar_aciertos(detalle_sem: dict, metrics_sem: dict) -> tuple[list, Counter, list]:
    nuevos, por_categoria = [], Counter()
    for doc_id, pares in detalle_sem.items():
        for par in pares:
            cat = categorizar(par["pred"], par["gold"])
            por_categoria[cat] += 1
            if not par["exacto"]:
                nuevos.append({"doc_id": doc_id, **par, "categoria": cat})

    sospechosos = [c for c in nuevos if c["categoria"] == "sinonimo/parafrasis (sin solapamiento lexico)"]
    return nuevos, por_categoria, sospechosos


def analizar_ambiguedad_gold(true_by_doc: dict, emb_sim: EmbeddingSimilarity, threshold: float) -> dict:
    pares_ambiguos, total_pares = 0, 0
    ejemplos = []
    for doc_id, golds in true_by_doc.items():
        golds = sorted(golds)
        if len(golds) < 2:
            continue
        S_gg = emb_sim.sim_matrix(golds, golds)
        for i in range(len(golds)):
            for j in range(i + 1, len(golds)):
                total_pares += 1
                if S_gg[i, j] >= threshold:
                    pares_ambiguos += 1
                    ejemplos.append((doc_id, golds[i], golds[j], float(S_gg[i, j])))

    fraccion = pares_ambiguos / total_pares if total_pares else 0.0
    return {
        "pares_ambiguos": pares_ambiguos,
        "pares_totales": total_pares,
        "fraccion": round(fraccion, 4),
        "ejemplos": [{"doc_id": d, "gold_a": a, "gold_b": b, "sim": round(s, 4)}
                     for d, a, b, s in sorted(ejemplos, key=lambda x: -x[3])[:50]],
    }


def run(cfg: dict, project_root) -> dict:
    rutas = resolver_rutas(cfg, project_root)

    assert rutas["dim1_path"].exists(), (
        f"No se encontro {rutas['dim1_path']}. Correr metrics_exact.py primero."
    )
    with open(rutas["dim1_path"], "r", encoding="utf-8") as f:
        dim1 = json.load(f)

    true_by_doc = {k: set(v) for k, v in dim1["true_by_doc"].items()}
    pred_by_doc = {k: set(v) for k, v in dim1["pred_by_doc"].items()}
    metrics_exact = dim1["metrics"]
    casos_boundary = dim1.get("casos_boundary", [])

    print(f"Predicciones cargadas: {len(true_by_doc)} documentos")

    dim1b_cfg = cfg["dimension1b_similitud_semantica"]
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_sim = EmbeddingSimilarity(dim1b_cfg["modelo_embeddings"], device)

    umbral_info = calibrar_umbral(
        casos_boundary, true_by_doc, pred_by_doc, emb_sim,
        dim1b_cfg["calibracion_umbral"]["n_pares_negativos"],
        dim1b_cfg["calibracion_umbral"]["seed"],
    )
    threshold = umbral_info["threshold"]

    metrics_sem, detalle_sem, no_emparejados = semantic_prf1_by_doc(
        true_by_doc, pred_by_doc, emb_sim, threshold
    )
    print("=== Metrica de similitud semantica ===")
    print(f"  F1 = {metrics_sem['f1']:.4f}  (exact-match F1 = {metrics_exact['f1']:.4f})")

    nuevos, por_categoria, sospechosos = analizar_aciertos(detalle_sem, metrics_sem)
    ambiguedad_gold = analizar_ambiguedad_gold(true_by_doc, emb_sim, threshold)

    resultado_dimension1b = {
        "dimension": "metrica_clasica_matching_semantico",
        "rol": "pau",
        "modelo": cfg["modelo"]["base_checkpoint"],
        "predicciones_origen": str(rutas["dim1_path"]),
        "gold_set": str(rutas["gold_set_path"]),
        "n_ejemplos_evaluados": len(true_by_doc),
        "configuracion": {
            "modelo_embeddings": dim1b_cfg["modelo_embeddings"],
            "funcion_similitud": dim1b_cfg["funcion_similitud"],
            "algoritmo_asignacion": dim1b_cfg["algoritmo_asignacion"],
            "restriccion": "uno a uno",
            "umbral": threshold,
            "criterio_umbral": "indice de Youden sobre positivos boundary vs negativos cross-document",
            "umbral_tpr_positivos": umbral_info["tpr_positivos"],
            "umbral_fpr_negativos": umbral_info["fpr_negativos"],
        },
        "metrics": metrics_sem,
        "metrics_exact_match_referencia": metrics_exact,
        "aciertos_por_categoria": dict(por_categoria),
        "aciertos_nuevos": nuevos,
        "emparejamientos_solo_embedding": sospechosos,
        "ambiguedad_gold": ambiguedad_gold,
        "detalle_emparejamientos": detalle_sem,
        "no_emparejados": no_emparejados,
    }

    with open(rutas["dim1b_path"], "w", encoding="utf-8") as f:
        json.dump(resultado_dimension1b, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {rutas['dim1b_path']}")
    print(f"  Aciertos recuperados: {metrics_sem['tp'] - metrics_exact['tp']}")

    return resultado_dimension1b