"""Lectura de los caches de M4/vision y union de paginas por documento (tarea 5a).

El set de eval_visual tiene una imagen por pagina y los CSV de casos usan
id = "<doc_id>__p<k>". El cache de transcripciones de M4/vision sale entonces con una
entrada por pagina. Antes del NER y de cualquier metrica por documento hay que unir las
paginas de cada doc_id en orden, porque una entidad puede quedar partida entre dos paginas.

`unir_paginas` produce un cache por documento con la misma forma que una Transcripcion
de M4/vision (texto, backend, valvula, error, ...), asi que se puede pasar tal cual a
`M4/vision/ner_bridge.generar_cache_ner` para obtener un cache NER con los doc_id del harness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePath

_RE_ID_PAGINA = re.compile(r"^(?P<doc>.+)__p(?P<pag>\d+)$")
_RE_ARCHIVO = re.compile(r"^(?P<doc>.+)_p(?P<pag>\d+)\.(?:png|jpe?g)$", re.IGNORECASE)
MENSAJE_SIN_TEXTO = "No se aprecia texto legible en la imagen."   # igual que M4/vision/prompts.py


def cargar_json(ruta: Path):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _identificar(clave: str, registro: dict) -> tuple[str, int | None, str | None]:
    """(doc_id, pagina, nivel) de un registro del cache. pagina=None si ya es por documento."""
    imagen = str(registro.get("imagen") or "")
    nivel = None
    doc, pag = None, None
    if imagen:
        p = PurePath(imagen.replace("\\", "/"))
        m = _RE_ARCHIVO.match(p.name)
        if m:
            doc, pag = m.group("doc"), int(m.group("pag"))
            nivel = p.parent.name or None
    m = _RE_ID_PAGINA.match(str(registro.get("doc_id") or clave))
    if m:
        doc, pag = m.group("doc"), int(m.group("pag"))
    if doc is None:
        doc = str(registro.get("doc_id") or clave)
    return doc, pag, nivel


def texto_util(registro: dict) -> str:
    """Texto transcrito; vacio si hubo error o el backend uso la valvula (no vio texto)."""
    if registro.get("error") or registro.get("valvula"):
        return ""
    texto = str(registro.get("texto") or "")
    return "" if texto.strip() == MENSAJE_SIN_TEXTO else texto


def unir_paginas(cache: dict, manifest_docs: dict, nivel_forzado: str | None = None) -> tuple[str, dict, dict]:
    """Cache de M4/vision (por pagina o por documento) -> (nivel, cache por documento, avisos).

    `manifest_docs` es {doc_id: entrada del manifiesto}. Solo se aceptan documentos y
    paginas que existan en el manifiesto para ese nivel.
    """
    niveles = set()
    por_doc: dict[str, dict] = {}
    for clave, reg in cache.items():
        doc, pag, nivel = _identificar(clave, reg)
        if nivel:
            niveles.add(nivel)
        por_doc.setdefault(doc, {})[pag] = reg

    if nivel_forzado:
        nivel = nivel_forzado
    elif len(niveles) == 1:
        nivel = niveles.pop()
    elif not niveles:
        raise ValueError("No se pudo deducir el nivel de las rutas de imagen; indicalo con 'nivel' en la entrada.")
    else:
        raise ValueError(f"El cache mezcla niveles {sorted(niveles)}; se espera un cache por nivel.")

    avisos = {"documentos_fuera_del_manifiesto": [], "paginas_faltantes": [], "paginas_sobrantes": []}
    unido = {}
    for doc, paginas in sorted(por_doc.items()):
        info = (manifest_docs.get(doc) or {}).get("niveles", {}).get(nivel)
        if info is None:
            avisos["documentos_fuera_del_manifiesto"].append(doc)
            continue
        n = info["n_paginas"]
        if None in paginas:                       # cache ya por documento
            reg = paginas[None]
            textos, regs = [texto_util(reg)], [reg]
        else:
            textos, regs = [], []
            for k in range(1, n + 1):
                reg = paginas.get(k)
                if reg is None:
                    avisos["paginas_faltantes"].append(f"{doc}__p{k}")
                textos.append(texto_util(reg) if reg else "")
                regs.append(reg or {})
            avisos["paginas_sobrantes"] += [f"{doc}__p{k}" for k in paginas if k is not None and k > n]
        texto = "\n".join(t for t in textos if t)
        unido[doc] = {
            "doc_id": doc,
            "texto": texto if texto else MENSAJE_SIN_TEXTO,
            "valvula": not texto,
            "error": "; ".join(r["error"] for r in regs if r.get("error")) or None,
            "backend": next((r.get("backend") for r in regs if r.get("backend")), None),
            "segundos": round(sum(float(r.get("segundos") or 0) for r in regs), 3),
            "nivel": nivel,
            "n_paginas": len(regs),
            "paginas_valvula": sum(1 for r in regs if r.get("valvula")),
            "paginas_error": sum(1 for r in regs if r.get("error")),
            "paginas_faltantes": sum(1 for r in regs if not r),
            "imagenes": [r.get("imagen") for r in regs],
        }
    return nivel, unido, avisos
