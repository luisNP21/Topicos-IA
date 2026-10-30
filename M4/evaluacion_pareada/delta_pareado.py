"""Delta M3 (texto) -> M4 (imagen) solo sobre el mismo gold.

Lee dos `resultado_dimension1.json` del harness (texto y visión), verifica que
evalúan los mismos doc_id con las mismas entidades gold y, solo entonces,
escribe `scorecard_m4.csv` con P/R/F1, delta y latencia por condición.

Uso:
    python M4/evaluacion_pareada/delta_pareado.py --texto RUTA_dim1_texto.json \
        --vision limpio=RUTA_dim1.json foto_realista=RUTA_dim1.json \
        [--meta limpio=RUTA.vision_meta.json ...] [--salida M4/scorecard_m4.csv]

Los `resultado_dimension1.json` salen de M2/harness (metrics_exact) corriendo el
mismo NER con fuente.tipo=texto y fuente.tipo=vision sobre el MISMO gold.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def cargar(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prf(true_by_doc: dict, pred_by_doc: dict) -> dict:
    tp = fp = fn = 0
    for doc, gold in true_by_doc.items():
        g, p = set(gold), set(pred_by_doc.get(doc, []))
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def diferencias_gold(a: dict, b: dict) -> list[str]:
    problemas = []
    solo_a, solo_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if solo_a or solo_b:
        problemas.append(f"doc_id distintos: solo texto={solo_a[:5]} solo visión={solo_b[:5]}")
    for doc in sorted(set(a) & set(b)):
        if set(a[doc]) != set(b[doc]):
            problemas.append(f"gold distinto en {doc}")
    return problemas


def pares(valores: list[str] | None) -> dict[str, str]:
    out = {}
    for v in valores or []:
        nombre, _, ruta = v.partition("=")
        if not ruta:
            raise SystemExit(f"Formato esperado nombre=ruta, vino: {v}")
        out[nombre] = ruta
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--texto", required=True)
    ap.add_argument("--vision", nargs="+", required=True)
    ap.add_argument("--meta", nargs="*")
    ap.add_argument("--latencia-texto", type=float, default=None)
    ap.add_argument("--salida", default="M4/scorecard_m4.csv")
    args = ap.parse_args(argv)

    texto = cargar(args.texto)
    base = prf(texto["true_by_doc"], texto["pred_by_doc"])
    metas = {k: cargar(v) for k, v in pares(args.meta).items()}
    filas = [{"condicion": "texto", "n_docs": len(texto["true_by_doc"]),
              "n_gold": base["tp"] + base["fn"], **base, "delta_f1": 0.0,
              "latencia_s_caso": args.latencia_texto}]

    errores = []
    for nombre, ruta in pares(args.vision).items():
        vis = cargar(ruta)
        problemas = diferencias_gold(texto["true_by_doc"], vis["true_by_doc"])
        if problemas:
            errores.append(f"{nombre}: " + "; ".join(problemas[:3]))
            continue
        m = prf(vis["true_by_doc"], vis["pred_by_doc"])
        filas.append({"condicion": nombre, "n_docs": len(vis["true_by_doc"]),
                      "n_gold": m["tp"] + m["fn"], **m, "delta_f1": m["f1"] - base["f1"],
                      "latencia_s_caso": metas.get(nombre, {}).get("latencia_media_s")})

    if errores:
        print("Delta NO comparable; no se escribe el scorecard:", file=sys.stderr)
        for e in errores:
            print("  -", e, file=sys.stderr)
        return 2

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        for fila in filas:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in fila.items()})
    print(f"Escrito {salida} con {len(filas)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
