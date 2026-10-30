"""Auditoría de entidades con el MISMO gold que el harness.

Clasifica cada entidad que el NER sacó de la transcripción:
- acierto: está en el gold del caso;
- fp_ner: no está en el gold, pero sí en el texto original (error del NER, no de la visión);
- inventada: no está en el texto original (la puso la lectura de la imagen).

Uso:
    python M4/evaluacion_pareada/auditar_entidades.py \
        --gold M4/eval_set/eval_set_casos.jsonl --ner limpio=CACHE_NER.json [foto_realista=...]
El cache NER es {doc_id: [entidades]} (por documento o por página <doc_id>__p<k>).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verificar_gold import cargar, normalizar  # noqa: E402


def por_documento(cache: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for k, v in cache.items():
        out[k.split("__p")[0]].update(normalizar(e) for e in v)
    return out


def auditar(gold: dict[str, dict], ner: dict[str, set[str]]) -> dict:
    conteo = {"predichas": 0, "acierto": 0, "fp_ner": 0, "inventada": 0}
    inventadas = []
    for doc, r in gold.items():
        g = {normalizar(e) for e in r["entities"]}
        texto = normalizar(r["text"])
        for e in sorted(ner.get(doc, set())):
            conteo["predichas"] += 1
            if e in g:
                conteo["acierto"] += 1
            elif e in texto:
                conteo["fp_ner"] += 1
            else:
                conteo["inventada"] += 1
                inventadas.append({"doc_id": doc, "entidad": e})
    return {**conteo, "inventadas": inventadas}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ner", nargs="+", required=True, help="nivel=ruta_cache_ner.json")
    args = ap.parse_args(argv)
    gold = cargar(args.gold)
    print("| Nivel | Predichas | Acierto | FP del NER | Inventada |")
    print("|---|---|---|---|---|")
    for par in args.ner:
        nivel, _, ruta = par.partition("=")
        r = auditar(gold, por_documento(json.loads(Path(ruta).read_text(encoding="utf-8"))))
        print(f"| {nivel} | {r['predichas']} | {r['acierto']} | {r['fp_ner']} | {r['inventada']} |")
        for x in r["inventadas"]:
            print(f"<!-- inventada {nivel}: {x['doc_id']} «{x['entidad']}» -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
