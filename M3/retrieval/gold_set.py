"""
Gold set del retrieval: carga, validación contra el índice y resumen.

El perfil del YAML indica las consultas (JSONL) y, opcionalmente, las anotaciones por chunk (CSV)
y gold_meta (JSON) con la descripción del corpus sobre el que se anotó. Como los chunk_id son
posicionales, una nueva ingesta puede cambiarlos; por eso, antes de medir, se comprueba que el
gold set corresponda al índice cargado y, si no, se detiene con un mensaje que dice qué falló.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from experimento_s08 import TIPOS_CONSULTA, cargar_consultas
from retrieval import IndiceRAG, info_indice

COMPROBACIONES_CORPUS = ("Huella del corpus", "Número de chunks", "Chunks por guía")
QUE_HACER_CORPUS = ("El corpus cargado no es el que se anotó. Use la versión del corpus que describe gold_meta.json "
                    "o anote de nuevo el gold set sobre el corpus actual y actualice gold_meta.json.")
QUE_HACER_GOLD = "Corrija el gold set (consultas, anotaciones o gold_meta.json) en M3/retrieval/data."


class GoldSetInvalido(RuntimeError):
    """El gold set no corresponde al índice cargado."""


def cargar_gold(cfg: dict) -> dict:
    """Consultas, anotaciones y metadatos del gold set del perfil activo."""
    from config_retrieval import rutas

    r = rutas(cfg)
    anotaciones = None
    if r.get("anotaciones"):
        with open(r["anotaciones"], encoding="utf-8-sig", newline="") as f:
            anotaciones = list(csv.DictReader(f))
    meta = json.loads(Path(r["gold_meta"]).read_text(encoding="utf-8")) if r.get("gold_meta") else None
    return {"consultas": cargar_consultas(r["consultas"]), "anotaciones": anotaciones, "meta": meta}


def _comprobacion(nombre: str, errores: list[str], detalle_ok: str = "") -> dict:
    if not errores:
        return {"Comprobación": nombre, "Resultado": "correcta", "Detalle": detalle_ok}
    extra = f" (y {len(errores) - 5} más)" if len(errores) > 5 else ""
    return {"Comprobación": nombre, "Resultado": "falla", "Detalle": "; ".join(errores[:5]) + extra}


def validar_gold(indice: IndiceRAG, gold: dict, detener: bool = True) -> list[dict]:
    """Compara el gold set con el índice. Devuelve una fila por comprobación y, si alguna falla
    y detener es True, lanza GoldSetInvalido con todas las fallas y qué hacer."""
    guia_de = {c["chunk_id"]: c["doc_id"] for c in indice.chunks}
    guias = set(guia_de.values())
    consultas, anotaciones, meta = gold["consultas"], gold.get("anotaciones"), gold.get("meta")
    filas = []

    if meta:
        c, info = meta["corpus"], info_indice(indice)
        por_guia = Counter(guia_de.values())
        filas.append(_comprobacion("Huella del corpus", [] if info["huella_sha256"] == c["huella_sha256"] else
                                   [f"índice {info['huella_sha256']}, gold_meta {c['huella_sha256']}"],
                                   info["huella_sha256"]))
        filas.append(_comprobacion("Número de chunks", [] if len(guia_de) == c["n_chunks"] else
                                   [f"índice {len(guia_de)}, gold_meta {c['n_chunks']}"], str(len(guia_de))))
        filas.append(_comprobacion("Chunks por guía", [
            f"{d}: índice {por_guia.get(d, 0)}, gold_meta {c['chunks_por_guia'].get(d, 0)}"
            for d in sorted(set(por_guia) | set(c["chunks_por_guia"]))
            if por_guia.get(d, 0) != c["chunks_por_guia"].get(d, 0)], f"{len(por_guia)} guías"))
        esperado = meta.get("consultas", {})
        dentro = sum(not q.get("fuera_de_corpus") for q in consultas)
        filas.append(_comprobacion("Número de consultas", [
            f"{nombre}: archivo {real}, gold_meta {esperado[clave]}"
            for clave, nombre, real in (("total", "total", len(consultas)),
                                        ("dentro_del_corpus", "dentro del corpus", dentro),
                                        ("fuera_del_corpus", "fuera del corpus", len(consultas) - dentro))
            if clave in esperado and esperado[clave] != real], str(len(consultas))))

    filas.append(_comprobacion("Chunks relevantes existen en el índice", [
        f"{x} (consulta '{q['consulta']}')" for q in consultas for x in q.get("chunks_relevantes") or []
        if x not in guia_de]))
    filas.append(_comprobacion("Chunks relevantes pertenecen a una guía relevante", [
        f"{x} es de {guia_de[x]}, no de {q['relevantes']} (consulta '{q['consulta']}')"
        for q in consultas for x in q.get("chunks_relevantes") or []
        if x in guia_de and guia_de[x] not in q["relevantes"]]))
    filas.append(_comprobacion("Guías relevantes existen en el índice", [
        f"{d} (consulta '{q['consulta']}')" for q in consultas for d in q["relevantes"] if d not in guias]))
    filas.append(_comprobacion("Consultas dentro del corpus con chunks relevantes", [
        f"'{q['consulta']}' no tiene chunks_relevantes" for q in consultas
        if not q.get("fuera_de_corpus") and not q.get("chunks_relevantes")]))

    if anotaciones is not None:
        anotados = {a["chunk_id"]: a["doc_id"] for a in anotaciones}
        filas.append(_comprobacion("Anotaciones cubren el índice", [
            *(f"{x} anotado pero no está en el índice" for x in anotados if x not in guia_de),
            *(f"{x} está en el índice pero no está anotado" for x in guia_de if x not in anotados),
            *(f"{x} anotado como {d}, en el índice es de {guia_de[x]}" for x, d in anotados.items()
              if x in guia_de and d != guia_de[x])], f"{len(anotados)} chunks"))

    fallas = [f for f in filas if f["Resultado"] == "falla"]
    if fallas and detener:
        que_hacer = QUE_HACER_CORPUS if any(f["Comprobación"] in COMPROBACIONES_CORPUS for f in fallas) else QUE_HACER_GOLD
        raise GoldSetInvalido("El gold set no corresponde al índice cargado:\n"
                              + "\n".join(f"  - {f['Comprobación']}: {f['Detalle']}" for f in fallas)
                              + f"\n{que_hacer}")
    return filas


def resumen_gold(gold: dict) -> dict:
    """Tablas del gold set: consultas por tipo, por guía y chunks relevantes por consulta."""
    consultas, anotaciones, meta = gold["consultas"], gold.get("anotaciones") or [], gold.get("meta") or {}
    por_tipo = Counter(q.get("tipo", "") for q in consultas)
    tipos = [{"Tipo de consulta": TIPOS_CONSULTA.get(t, t), "Consultas": n} for t, n in por_tipo.items()]

    guias = sorted({d for q in consultas for d in q["relevantes"]})
    por_guia = []
    for d in guias:
        de_guia = [q for q in consultas if d in q["relevantes"]]
        fila = {"Guía": d, "Consultas": len(de_guia),
                "Chunks relevantes distintos": len({x for q in de_guia for x in q.get("chunks_relevantes") or []})}
        if anotaciones:
            de_d = [a for a in anotaciones if a["doc_id"] == d]
            fila["Chunks de la guía"] = len(de_d)
            fila["Chunks anotados de tratamiento"] = sum(a.get("relevante_tratamiento_general") == "1" for a in de_d)
        por_guia.append(fila)

    por_consulta = [{"Consulta": q["consulta"], "Tipo de consulta": TIPOS_CONSULTA.get(q.get("tipo"), q.get("tipo")),
                     "Guía": ", ".join(q["relevantes"]) or "fuera del corpus",
                     "Chunks relevantes": len(q.get("chunks_relevantes") or [])} for q in consultas]
    anotador = meta.get("anotador") or ", ".join(sorted({q.get("anotador") for q in consultas if q.get("anotador")}))
    return {"por_tipo": tipos, "por_guia": por_guia, "por_consulta": por_consulta, "anotador": anotador or None}
