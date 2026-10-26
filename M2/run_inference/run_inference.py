import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HARNESS_DIR = ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import yaml
from dotenv import load_dotenv

from common import aggregate_entities_by_original_doc, chunk_words, log
from gold_loader import cargar_gold_set
from model_loader import cargar_modelo, sanity_check
from inference import run_inference


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "PROJECT_ROOT no esta definido. Crear un .env con:\n"
            "  PROJECT_ROOT=/ruta/al/proyecto/en/Drive"
        )
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def _fuente_texto(cfg: dict, project_root: Path, eval_set: list[dict]) -> tuple[dict[str, str], dict | None]:
    """Decide de dónde sale el texto que alimenta el NER.

    Devuelve `(textos_por_doc, vision_meta)`:
    - `fuente.tipo == "texto"` (default): usa `rec["text"]` del gold set. El
      comportamiento es idéntico al histórico; `vision_meta` es None.
    - `fuente.tipo == "vision"`: transcribe las imágenes del manifiesto de M4
      con el backend configurado y usa ESE texto. `vision_meta` queda con el
      backend, latencias, válvulas y errores para que el harness/scorecard los
      registre.

    Los doc_id de los textos resultantes deben coincidir con los del gold set;
    si no, el harness fallará explícitamente aguas abajo.
    """
    fuente = cfg.get("fuente") or {}
    tipo = str(fuente.get("tipo", "texto")).lower()

    if tipo == "texto":
        return {rec["doc_id"]: rec["text"] for rec in eval_set}, None

    if tipo != "vision":
        raise ValueError(
            f"fuente.tipo desconocido: '{tipo}'. Use 'texto' o 'vision'."
        )

    # Fuente de visión: se transcribe la imagen de cada caso y ese texto entra
    # al mismo NER. No se reimplementa la visión aquí: se reutiliza M4/vision.
    # El codigo y el config de M4/vision viven en el repo clonado, no en Drive
    # (PROJECT_ROOT solo tiene datos). Este archivo esta en <repo>/M2/run_inference/.
    repo_root = Path(__file__).resolve().parents[2]
    vision_dir = repo_root / "M4" / "vision"
    if str(vision_dir) not in sys.path:
        sys.path.insert(0, str(vision_dir))
    try:
        from manifiesto import cargar_manifiesto, resolver_ruta_imagen
        from transcripcion import cargar_imagen, transcribir, transcribir_auto
        from backends import crear_backend
    except Exception as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "No se pudo importar M4/vision. Verificar que el módulo exista y que "
            "sus dependencias estén instaladas (pip install -r M4/vision/requirements.txt): "
            f"{exc}"
        ) from exc

    vision_cfg_path = fuente.get("vision_config", "M4/vision/config.yaml")
    with open(repo_root / vision_cfg_path, "r", encoding="utf-8") as f:
        vision_cfg = yaml.safe_load(f)

    backend_nombre = fuente.get("backend") or (vision_cfg.get("vision") or {}).get("backend", "vlm")
    instruccion = (vision_cfg.get("vision") or {}).get("instruccion")

    # El manifiesto de imágenes de M4 (lo construyen Luis/Pau). El doc_id de
    # cada caso debe coincidir con el del gold set.
    manifest = fuente.get("manifest") or (vision_cfg.get("rutas") or {}).get("manifest")
    if not manifest:
        raise ValueError(
            "fuente.tipo=vision requiere un manifiesto de imágenes. Configure "
            "fuente.manifest o rutas.manifest en M4/vision/config.yaml."
        )
    manifest_path = project_root / manifest
    # `fuente.imagenes_dir` permite apuntar a la muestra versionada
    # (M4/eval_set/ejemplos_finales) sin cambiar el default del módulo de visión.
    imagenes_dir = fuente.get("imagenes_dir") or (vision_cfg.get("rutas") or {}).get(
        "imagenes_dir"
    )
    if imagenes_dir:
        imagenes_dir = project_root / imagenes_dir

    # El set sintético combinado repite `id` entre niveles; se filtra por
    # `fuente.nivel`/`fuente.grupo` antes de validar duplicados.
    nivel_filtro = fuente.get("nivel")
    if manifest_path.stem.startswith("casos_"):
        nivel_filtro = manifest_path.stem.replace("casos_", "")

    casos = [
        dict(c)
        for c in cargar_manifiesto(
            manifest_path,
            nivel=nivel_filtro,
            grupo=fuente.get("grupo"),
        )
    ]
    for caso in casos:
        caso["_imagen_resuelta"] = resolver_ruta_imagen(caso, imagenes_dir, project_root)

    # Un documento puede venir partido en varias paginas (eval_visual escribe
    # id = "<doc_id>__p<k>"). Se juntan en orden de pagina antes del NER para
    # que el texto quede indexado por el doc_id del gold set.
    # Si las transcripciones ya fueron generadas por M4/vision, se reutilizan del cache.
    sufijo_manifest = Path(manifest).stem.replace("casos_", "")
    candidatos_cache = [
        project_root / f"M4/vision/predictions/transcripciones_{backend_nombre}_{sufijo_manifest}.json",
        project_root / f"M4/vision/predictions/transcripciones_{backend_nombre}.json",
        repo_root / f"M4/vision/predictions/transcripciones_{backend_nombre}_{sufijo_manifest}.json",
        repo_root / f"M4/vision/predictions/transcripciones_{backend_nombre}.json",
    ]
    cache_existente: dict[str, dict] = {}
    for c_path in candidatos_cache:
        if c_path.exists():
            try:
                raw_c = json.loads(c_path.read_text(encoding="utf-8"))
                if isinstance(raw_c, dict) and raw_c:
                    cache_existente = raw_c
                    print(f"Reutilizando transcripciones de cache: {c_path.name} ({len(cache_existente)} items)")
                    break
            except Exception:
                pass

    paginas_por_doc: dict[str, list[tuple[int, str]]] = {}
    transcripciones: list[dict] = []
    backend = None  # se crea una sola vez si se necesita transcribir
    for caso in casos:
        cid = str(caso["id"])
        if cid in cache_existente and cache_existente[cid].get("texto") is not None:
            t = cache_existente[cid]
        else:
            imagen = cargar_imagen(caso["_imagen_resuelta"])
            if backend_nombre == "auto":
                t = transcribir_auto(
                    imagen, vision_cfg, instruccion=instruccion,
                    ruta=caso["_imagen_resuelta"], doc_id=caso["id"],
                )
            else:
                if backend is None:
                    backend = crear_backend(backend_nombre, vision_cfg)
                t = transcribir(
                    imagen, backend=backend, instruccion=instruccion,
                    ruta=caso["_imagen_resuelta"], doc_id=caso["id"],
                )
        doc_id, _, pag = cid.partition("__p")
        paginas_por_doc.setdefault(doc_id, []).append(
            (int(pag) if pag.isdigit() else 0, t.get("texto") or "")
        )
        transcripciones.append(t)

    latencias = [float(t.get("segundos") or 0.0) for t in transcripciones]
    n = len(transcripciones)
    vision_meta = {
        "fuente": "vision",
        "backend": backend_nombre,
        "modelo": (vision_cfg.get("vlm") or {}).get("model_id") if backend_nombre == "vlm" else backend_nombre,
        "manifest": str(manifest_path),
        "n_casos": n,
        "n_ok": sum(1 for t in transcripciones if t.get("error") is None),
        "n_error": sum(1 for t in transcripciones if t.get("error") is not None),
        "n_valvula": sum(1 for t in transcripciones if t.get("valvula")),
        "latencia_media_s": round(sum(latencias) / n, 3) if n else None,
        "latencia_total_s": round(sum(latencias), 3),
        "por_doc": {
            t["doc_id"]: {
                "segundos": t.get("segundos"),
                "valvula": t.get("valvula"),
                "error": t.get("error"),
                "sha256": t.get("sha256"),
            }
            for t in transcripciones
        },
    }
    textos = {
        doc_id: "\n".join(texto for _, texto in sorted(paginas))
        for doc_id, paginas in paginas_por_doc.items()
    }
    vision_meta["n_docs"] = len(textos)
    return textos, vision_meta


def run(cfg: dict, project_root: Path) -> dict:
    model_cfg = cfg["modelo"]
    model, tokenizer, id2label = cargar_modelo(model_cfg, project_root)
    sanity_check(model, tokenizer, id2label)

    eval_path = project_root / cfg["eval_set_path"]
    eval_set = cargar_gold_set(eval_path)

    output_path = project_root / cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # El texto que alimenta el NER puede venir del gold (texto) o de transcribir
    # imágenes (vision). El resto del pipeline (chunking, NER, agregación) es
    # idéntico en ambos casos.
    textos_por_doc, vision_meta = _fuente_texto(cfg, project_root, eval_set)

    def chunk_fn(words):
        return chunk_words(words, cfg["chunking"]["window_words"], cfg["chunking"]["overlap_words"])

    inference_records = []
    for rec in eval_set:
        doc_id = rec["doc_id"]
        if doc_id not in textos_por_doc:
            raise ValueError(
                f"No hay texto para el doc_id '{doc_id}'. Si fuente.tipo=vision, "
                "revise que el manifiesto de imágenes cubra todos los casos del gold set."
            )
        text = textos_por_doc[doc_id]
        words = text.split()
        chunks = chunk_fn(words)
        collected = []
        for chunk_idx, chunk_words_list in chunks:
            import torch
            from transformers import AutoTokenizer
            inputs = tokenizer(
                chunk_words_list,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
            ).to(model.device)
            word_ids = inputs.word_ids(batch_index=0)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1)[0].tolist()
            word_preds, seen = [], set()
            for idx, wid in zip(pred_ids, word_ids):
                if wid is None or wid in seen:
                    continue
                word_preds.append(id2label[idx])
                seen.add(wid)
            from common import bio_to_entity_set
            entity_set = bio_to_entity_set(chunk_words_list, word_preds)
            collected.append((f"{doc_id}_chunk{chunk_idx}" if len(chunks) > 1 else doc_id, entity_set))
        inference_records.extend(collected)

    pred_doc_ids = [doc_id for doc_id, _ in inference_records]
    pred_entity_sets = [ents for _, ents in inference_records]
    grouped = aggregate_entities_by_original_doc(pred_doc_ids, pred_entity_sets)
    serialized = {doc_id: sorted(ents) for doc_id, ents in grouped.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    resultado = {
        "sistema": cfg["sistema"],
        "n_documentos": len(serialized),
        "output_path": str(output_path),
    }

    # Si el texto vino de la visión, se deja un sidecar de metadatos junto al
    # cache de predicciones. El harness lo lee para registrarlo en el scorecard.
    if vision_meta is not None:
        meta_path = output_path.with_name(output_path.stem + ".vision_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(vision_meta, f, ensure_ascii=False, indent=2)
        resultado["vision_meta_path"] = str(meta_path)
        resultado["vision_meta"] = {
            k: vision_meta[k]
            for k in ("backend", "modelo", "n_casos", "n_ok", "n_error", "n_valvula", "latencia_media_s")
        }
        log(
            f"run_inference (vision/{vision_meta['backend']}) completado: "
            f"{len(serialized)} documentos | latencia media {vision_meta['latencia_media_s']}s | "
            f"válvulas {vision_meta['n_valvula']} | errores {vision_meta['n_error']}"
        )
    else:
        log(f"run_inference completado: {len(serialized)} documentos cacheados en {output_path}")

    return resultado


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inferencia NER (texto o vision)")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project_root = resolver_project_root()
    run(cfg, project_root)