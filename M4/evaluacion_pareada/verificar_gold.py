"""Chequeos de procedencia del gold antes de medir.

1. `--literal`: lista las entidades gold que no aparecen literalmente en el texto
   del caso (ningún lector de la imagen podría producirlas; bajan el recall de
   todas las condiciones por igual y conviene documentar de dónde salen).
2. `--comparar`: verifica que dos golds (p. ej. el del harness y el de la
   auditoría de transcripción) tienen los mismos doc_id y las mismas entidades.

Uso:
    python M4/evaluacion_pareada/verificar_gold.py --literal M4/eval_set/eval_set_casos.jsonl
    python M4/evaluacion_pareada/verificar_gold.py --comparar GOLD_A.jsonl GOLD_B.jsonl
Sale con código 1 si encuentra problemas.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(s.split())


def cargar(path: str) -> dict[str, dict]:
    """Mismo mapeo que M2/harness/gold_loader.py: doc_id o ex_{i}."""
    out = {}
    lineas = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, l in enumerate(lineas):
        ex = json.loads(l)
        doc = ex.get("doc_id", f"ex_{i}")
        out[doc] = {
            "text": ex.get("input", ex.get("text", "")),
            "entities": ex.get("esperado", ex.get("entities_gold", [])),
        }
    return out


def no_literales(gold: dict[str, dict]) -> list[tuple[str, str]]:
    faltan = []
    for doc, r in gold.items():
        texto = normalizar(r["text"])
        for e in r["entities"]:
            if normalizar(e) not in texto:
                faltan.append((doc, e))
    return faltan


def comparar(a: dict[str, dict], b: dict[str, dict]) -> list[str]:
    problemas = []
    if set(a) != set(b):
        problemas.append(f"doc_id distintos: solo A={sorted(set(a) - set(b))[:5]} solo B={sorted(set(b) - set(a))[:5]}")
    for doc in sorted(set(a) & set(b)):
        ea = {normalizar(e) for e in a[doc]["entities"]}
        eb = {normalizar(e) for e in b[doc]["entities"]}
        if ea != eb:
            problemas.append(f"{doc}: solo A={sorted(ea - eb)[:3]} solo B={sorted(eb - ea)[:3]}")
    return problemas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--literal")
    g.add_argument("--comparar", nargs=2)
    args = ap.parse_args(argv)
    if args.literal:
        gold = cargar(args.literal)
        faltan = no_literales(gold)
        total = sum(len(r["entities"]) for r in gold.values())
        print(f"{len(gold)} casos, {total} entidades gold, {len(faltan)} no literales en su texto")
        for doc, e in faltan:
            print(f"  - {doc}: «{e}»")
        return 1 if faltan else 0
    problemas = comparar(cargar(args.comparar[0]), cargar(args.comparar[1]))
    print("Golds idénticos" if not problemas else f"{len(problemas)} diferencias")
    for p in problemas[:20]:
        print("  -", p)
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
