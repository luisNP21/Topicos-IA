"""
Retrieval avanzado sobre el índice Chroma de guías clínicas.

Parte de la implementación del laboratorio S08 (BM25 con rank_bm25, fusión RRF con k=60 y
reranker cross-encoder mMARCO) y la adapta al corpus del proyecto:

  * El índice es una colección Chroma persistente ("guias_clinicas") con embeddings
    intfloat/multilingual-e5-base indexados con el prefijo "passage: ". Las consultas se
    embeben con el mismo modelo y el prefijo "query: ".
  * La consulta que llega es una entidad extraída ("diabetes mellitus tipo 2"). Como toda la
    guía menciona la enfermedad, sin más contexto el retrieval no distingue la sección de
    tratamiento de la de diagnóstico o epidemiología. Los modos de intención reescriben la
    consulta para expresar qué información se busca (query transformation).

Modos disponibles (ver MODOS):
  densa                  búsqueda densa (el RAG ingenuo de S07)
  hibrida                BM25 + densa fusionadas con RRF
  rerank                 híbrida -> cross-encoder con la entidad como consulta
  rerank_pregunta        híbrida -> cross-encoder con la pregunta de tratamiento
  intencion_sin_rerank   varias consultas de tratamiento fusionadas con RRF, sin reranker
  intencion              varias consultas de tratamiento -> cross-encoder con la pregunta
  intencion_seccion      como "intencion", reordenando solo chunks de secciones de tratamiento
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

EMBED_MODEL = "intfloat/multilingual-e5-base"
PREFIJO_QUERY, PREFIJO_PASSAGE = "query: ", "passage: "
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHROMA_DIR, COLECCION = "data/chroma_guias", "guias_clinicas"
MODOS = ("densa", "hibrida", "rerank", "rerank_pregunta", "intencion_sin_rerank", "intencion",
         "intencion_seccion")
TIPO_SCORE = {"densa": "cosine", "hibrida": "rrf", "intencion_sin_rerank": "rrf"}   # el resto: reranker_prob

PLANTILLAS_INTENCION = ("tratamiento de {e}", "manejo farmacológico de {e}", "{e}: recomendaciones terapéuticas")
PREGUNTA_INTENCION = "¿Cuál es el tratamiento de {e}?"

PATRON_TRATAMIENTO = re.compile(r"tratamient|manejo|terap|farmacol|intervenc|insulin|dosis|posolog")
PATRON_OTRA_SECCION = re.compile(r"diagnost|epidemiolog|definic|introduc|descripc|etiolog|fisiopatol|"
                                 r"pronost|metodolog|alcance|referencia|anexo|glosario|autor|tamiz|cribado")
PROTOTIPOS_SECCION = {
    "tratamiento": "recomendaciones de tratamiento farmacológico y no farmacológico, medicamentos, "
                   "dosis y manejo terapéutico del paciente",
    "diagnostico": "criterios diagnósticos, pruebas de laboratorio, estudios de imagen y evaluación "
                   "para confirmar el diagnóstico",
    "descripcion": "definición, epidemiología, causas, factores de riesgo y manifestaciones clínicas",
    "otro": "introducción, alcance, metodología, autores, conflictos de interés y referencias de la guía",
}

DensaFn = Callable[[str, int], list[tuple[str, float]]]


class Embedder(Protocol):
    def encode(self, textos: list[str], **kw: Any) -> Any: ...


class Reranker(Protocol):
    def predict(self, pares: list[tuple[str, str]]) -> Any: ...


def configurar_intencion(plantillas: list[str] | tuple[str, ...], pregunta: str) -> None:
    """Reemplaza las reformulaciones y la pregunta de tratamiento (por ejemplo, desde el YAML)."""
    global PLANTILLAS_INTENCION, PREGUNTA_INTENCION
    PLANTILLAS_INTENCION, PREGUNTA_INTENCION = tuple(plantillas), pregunta


def pregunta_intencion(entidad: str) -> str:
    """Pregunta que recibe el reranker. La generación debe usar la misma, para que el LLM
    responda sobre lo mismo que se buscó."""
    return PREGUNTA_INTENCION.format(e=entidad)


# ---------------------------------------------------------------------------
# Texto y secciones
# ---------------------------------------------------------------------------
def quitar_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin tildes y sin puntuación: las entidades extraídas traen erratas y tildes faltantes."""
    return re.findall(r"\w+", quitar_tildes(texto.lower()))


def _normalizar(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def clasificar_secciones(chunks: list[dict], emb_chunks: Any = None,
                         embedder: Embedder | None = None) -> tuple[list[str], list[str]]:
    """Clasifica cada chunk como 'tratamiento', 'otra' o 'desconocida'.

    Primero por el título de la sección; si el título no es concluyente (por ejemplo "5.2" o
    el nombre de un fármaco), por la descripción de referencia más parecida a su embedding.
    Devuelve también el origen de cada decisión ('titulo', 'similitud' o '-')."""
    tipos, origenes, pendientes = [], [], []
    for i, c in enumerate(chunks):
        titulo = quitar_tildes((c.get("seccion") or "").lower())
        if PATRON_TRATAMIENTO.search(titulo):
            tipos.append("tratamiento")
            origenes.append("titulo")
        elif PATRON_OTRA_SECCION.search(titulo):
            tipos.append("otra")
            origenes.append("titulo")
        else:
            tipos.append("desconocida")
            origenes.append("-")
            pendientes.append(i)
    if pendientes and emb_chunks is not None and embedder is not None:
        nombres = list(PROTOTIPOS_SECCION)
        proto = _normalizar(np.asarray(
            embedder.encode([PREFIJO_QUERY + PROTOTIPOS_SECCION[n] for n in nombres]), dtype=float))
        emb = _normalizar(np.asarray(emb_chunks, dtype=float)[pendientes])
        for i, fila in zip(pendientes, emb @ proto.T):
            tipos[i] = "tratamiento" if nombres[int(np.argmax(fila))] == "tratamiento" else "otra"
            origenes[i] = "similitud"
    return tipos, origenes


# ---------------------------------------------------------------------------
# Lectura del corpus
# ---------------------------------------------------------------------------
def chunks_desde_chroma(coleccion) -> list[dict]:
    todo = coleccion.get(include=["documents", "metadatas"])
    return [{"chunk_id": cid, "doc_id": md["doc_id"], "texto": doc, "seccion": md.get("seccion") or None}
            for cid, doc, md in zip(todo["ids"], todo["documents"], todo["metadatas"])]


def embeddings_desde_chroma(coleccion, chunk_ids: list[str]) -> np.ndarray:
    todo = coleccion.get(ids=chunk_ids, include=["embeddings"])
    por_id = dict(zip(todo["ids"], todo["embeddings"]))
    return np.asarray([por_id[cid] for cid in chunk_ids], dtype=float)


def cargar_chunks_json(ruta: str | Path) -> list[dict]:
    """Lee una lista JSON de chunks o una carpeta con un chunk por archivo."""
    ruta = Path(ruta)
    if ruta.is_dir():
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ruta.glob("*.json"))]
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_manifest(ruta: str | Path | None) -> dict[str, dict]:
    if not ruta or not Path(ruta).exists():
        return {}
    return {d["doc_id"]: d for d in json.loads(Path(ruta).read_text(encoding="utf-8"))}


# ---------------------------------------------------------------------------
# Búsqueda densa
# ---------------------------------------------------------------------------
def densa_chroma(coleccion, embedder: Embedder) -> DensaFn:
    """Chroma con espacio coseno devuelve distancias; la similitud es 1 - distancia."""
    def buscar(consulta: str, n: int) -> list[tuple[str, float]]:
        q = embedder.encode([PREFIJO_QUERY + consulta], normalize_embeddings=True)
        r = coleccion.query(query_embeddings=np.asarray(q).tolist(), n_results=min(n, coleccion.count()),
                            include=["distances"])
        return [(cid, 1.0 - float(d)) for cid, d in zip(r["ids"][0], r["distances"][0])]
    return buscar


def densa_en_memoria(chunks: list[dict], embedder: Embedder) -> DensaFn:
    ids = [c["chunk_id"] for c in chunks]
    emb = _normalizar(np.asarray(embedder.encode([PREFIJO_PASSAGE + c["texto"] for c in chunks]), dtype=float))

    def buscar(consulta: str, n: int) -> list[tuple[str, float]]:
        q = _normalizar(np.asarray(embedder.encode([PREFIJO_QUERY + consulta]), dtype=float))[0]
        sims = emb @ q
        return [(ids[i], float(sims[i])) for i in np.argsort(-sims)[:n]]
    return buscar


# ---------------------------------------------------------------------------
# Índice
# ---------------------------------------------------------------------------
class IndiceRAG:
    def __init__(self, chunks: list[dict], densa_fn: DensaFn, reranker: Reranker | None = None, *,
                 manifest: dict[str, dict] | None = None, enriquecer: bool = True, krrf: int = 60,
                 n_candidatos: int = 30, aplicar_sigmoide: bool | None = None,
                 embedder: Embedder | None = None, emb_chunks: Any = None):
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("El corpus está vacío.")
        self.chunks = list(chunks)
        self.manifest = manifest or {}
        self.densa_fn, self.reranker = densa_fn, reranker
        self.krrf, self.n_candidatos = krrf, n_candidatos
        self._pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        # BM25 y el reranker leen "título. sección. texto": un chunk de tratamiento puede no
        # nombrar la enfermedad, pero el título de su guía sí.
        self._texto = [self._texto_busqueda(c) if enriquecer else c["texto"] for c in self.chunks]
        self._bm25 = BM25Okapi([tokenizar(t) for t in self._texto])

        self.aplicar_sigmoide = aplicar_sigmoide
        if reranker is not None and aplicar_sigmoide is None:
            sonda = np.asarray(reranker.predict([(self._texto[0], self._texto[0]), ("zzz qqq", self._texto[0])]))
            self.aplicar_sigmoide = bool(sonda.min() < 0 or sonda.max() > 1)

        if emb_chunks is None and embedder is not None:
            emb_chunks = embedder.encode([PREFIJO_PASSAGE + c["texto"] for c in self.chunks])
        self.seccion_tipo, self.seccion_origen = clasificar_secciones(self.chunks, emb_chunks, embedder)

    def _titulo(self, doc_id: str) -> str:
        return self.manifest.get(doc_id, {}).get("titulo") or doc_id

    def _texto_busqueda(self, c: dict) -> str:
        return ". ".join(p for p in (self._titulo(c["doc_id"]), c.get("seccion") or "", c["texto"]) if p)

    # Cada buscador devuelve [(posición del chunk, score)] de mayor a menor.
    def densa(self, consulta: str, k: int) -> list[tuple[int, float]]:
        return [(self._pos[cid], s) for cid, s in self.densa_fn(consulta, k) if cid in self._pos]

    def bm25(self, consulta: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenizar(consulta))
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i] > 0]

    def _rrf(self, consultas: list[str], k: int) -> list[tuple[int, float]]:
        """RRF sobre densa y BM25 de una o varias consultas: 1 / (krrf + puesto)."""
        n = max(k, self.n_candidatos)
        puntos: dict[int, float] = {}
        for q in consultas:
            for lista in (self.densa(q, n), self.bm25(q, n)):
                for puesto, (idx, _) in enumerate(lista, 1):
                    puntos[idx] = puntos.get(idx, 0.0) + 1.0 / (self.krrf + puesto)
        return sorted(puntos.items(), key=lambda x: -x[1])[:k]

    def _rerank(self, candidatos: list[int], pregunta: str, k: int) -> list[tuple[int, float]]:
        if self.reranker is None:
            raise RuntimeError("Este modo necesita un reranker.")
        if not candidatos:
            return []
        scores = np.asarray(self.reranker.predict([(pregunta, self._texto[i]) for i in candidatos]), dtype=float)
        if self.aplicar_sigmoide:
            scores = 1.0 / (1.0 + np.exp(-scores))
        return [(candidatos[i], float(scores[i])) for i in np.argsort(-scores)[:k]]

    @staticmethod
    def _variantes(consulta: str) -> list[str]:
        return [p.format(e=consulta) for p in PLANTILLAS_INTENCION]

    def hibrida(self, consulta: str, k: int) -> list[tuple[int, float]]:
        return self._rrf([consulta], k)

    def rerank(self, consulta: str, k: int) -> list[tuple[int, float]]:
        return self._rerank([i for i, _ in self._rrf([consulta], self.n_candidatos)], consulta, k)

    def rerank_pregunta(self, consulta: str, k: int) -> list[tuple[int, float]]:
        return self._rerank([i for i, _ in self._rrf([consulta], self.n_candidatos)],
                            pregunta_intencion(consulta), k)

    def intencion_sin_rerank(self, consulta: str, k: int) -> list[tuple[int, float]]:
        return self._rrf(self._variantes(consulta), k)

    def intencion(self, consulta: str, k: int) -> list[tuple[int, float]]:
        candidatos = [i for i, _ in self._rrf(self._variantes(consulta), self.n_candidatos)]
        return self._rerank(candidatos, pregunta_intencion(consulta), k)

    def intencion_seccion(self, consulta: str, k: int) -> list[tuple[int, float]]:
        """Si ningún candidato es de una sección de tratamiento, usa todos."""
        candidatos = [i for i, _ in self._rrf(self._variantes(consulta), self.n_candidatos)]
        de_tratamiento = [i for i in candidatos if self.seccion_tipo[i] == "tratamiento"]
        return self._rerank(de_tratamiento or candidatos, pregunta_intencion(consulta), k)

    def buscar(self, consulta: str, k: int, modo: str = "intencion") -> list[dict]:
        """Devuelve una lista de fragmentos: chunk_id, doc_id, texto, score, score_tipo, fuente,
        seccion y seccion_tipo. Solo los scores 'reranker_prob' (en [0, 1]) admiten umbrales."""
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}")
        tipo = TIPO_SCORE.get(modo, "reranker_prob")
        salida = []
        for i, s in getattr(self, modo)(consulta, k):
            c = self.chunks[i]
            salida.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "texto": c["texto"],
                           "score": round(s, 6), "score_tipo": tipo, "fuente": self._titulo(c["doc_id"]),
                           "seccion": c.get("seccion"), "seccion_tipo": self.seccion_tipo[i]})
        return salida


# ---------------------------------------------------------------------------
# Construcción desde Chroma y funciones públicas del pipeline
# ---------------------------------------------------------------------------
def construir_indice(persist_dir: str | None = None, manifest_path: str | None = None,
                     modelo_reranker: str = MODELO_RERANKER, modelo_embeddings: str = EMBED_MODEL,
                     **kw: Any) -> IndiceRAG:
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    try:
        from corpus_utils import EMBED_MODEL as MODELO_DEL_INDICE
        if MODELO_DEL_INDICE != modelo_embeddings:
            raise RuntimeError(f"El índice usa {MODELO_DEL_INDICE} y el retrieval {modelo_embeddings}; deben coincidir.")
    except ImportError:
        pass

    persist_dir = persist_dir or os.environ.get("M3_CHROMA_DIR", CHROMA_DIR)
    if not Path(persist_dir).exists():
        raise RuntimeError(f"No existe el índice Chroma en {persist_dir}.")
    coleccion = chromadb.PersistentClient(path=persist_dir).get_collection(COLECCION)
    embedder = SentenceTransformer(modelo_embeddings)
    chunks = chunks_desde_chroma(coleccion)
    indice = IndiceRAG(chunks, densa_chroma(coleccion, embedder), CrossEncoder(modelo_reranker, max_length=512),
                       manifest=cargar_manifest(manifest_path or os.environ.get("M3_MANIFEST")),
                       embedder=embedder,
                       emb_chunks=embeddings_desde_chroma(coleccion, [c["chunk_id"] for c in chunks]), **kw)
    indice.origen, indice.modelo_reranker, indice.modelo_embeddings = str(persist_dir), modelo_reranker, modelo_embeddings
    return indice


def info_indice(indice: IndiceRAG) -> dict:
    """Describe el índice evaluado. La huella cambia si cambia cualquier chunk, lo que permite
    saber si dos corridas se hicieron sobre el mismo corpus."""
    contenido = "\n".join(f"{c['chunk_id']}\t{c['texto']}"
                          for c in sorted(indice.chunks, key=lambda c: c["chunk_id"]))
    docs = sorted({c["doc_id"] for c in indice.chunks})
    return {"origen": getattr(indice, "origen", None), "n_chunks": len(indice.chunks),
            "n_documentos": len(docs), "documentos": docs,
            "huella_sha256": hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:16],
            "modelo_embeddings": getattr(indice, "modelo_embeddings", EMBED_MODEL), "modelo_reranker": getattr(indice, "modelo_reranker", None),
            "plantillas_intencion": list(PLANTILLAS_INTENCION), "pregunta_intencion": PREGUNTA_INTENCION,
            "chunks_de_tratamiento": indice.seccion_tipo.count("tratamiento"),
            "secciones_clasificadas_por_similitud": indice.seccion_origen.count("similitud"),
            "fecha": datetime.now().isoformat(timespec="seconds")}


_INDICE: IndiceRAG | None = None
_MODO_PIPELINE = "intencion"


def configurar(indice: IndiceRAG, modo: str | None = None) -> None:
    global _INDICE, _MODO_PIPELINE
    _INDICE = indice
    if modo is not None:
        _MODO_PIPELINE = modo


def _indice() -> IndiceRAG:
    """Si M3_RETRIEVAL_CONFIG apunta a un YAML, el índice se construye con esa configuración."""
    if _INDICE is None:
        if os.environ.get("M3_RETRIEVAL_CONFIG"):
            from config_retrieval import cargar_config, configurar_pipeline
            configurar_pipeline(cargar_config())
        else:
            configurar(construir_indice())
    return _INDICE


def retrieve_naive(consulta: str, k: int) -> list[dict]:
    return _indice().buscar(consulta, k, "densa")


def retrieve_advanced(consulta: str, k: int) -> list[dict]:
    indice = _indice()
    return indice.buscar(consulta, k, os.environ.get("M3_MODO_RETRIEVAL", _MODO_PIPELINE))
