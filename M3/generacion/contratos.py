"""
Contratos de datos del pipeline M3.

Estos tipos son la INTERFAZ entre las piezas del equipo. Se declaran con
TypedDict (documentan la forma del dato en tiempo de lectura) y Protocol
(permiten inyectar la implementacion real o un mock, para desarrollo en
paralelo).

IMPORTANTE: los contratos de Corpus y Retrieval NO son de esta pieza; se
declaran aqui solo como referencia del acuerdo del equipo. Su dueno es quien
los produce (Isa: corpus, Pau: retrieval). Si cambian un nombre de campo, basta
con actualizarlo aqui y los Protocol avisan en el resto del codigo.

Pieza de esta carpeta (Agustin): NormalizationResult se consume, el resto de la
parte de generacion/orquestacion se define al final.
"""

from __future__ import annotations

from typing import NotRequired, Optional, Protocol, TypedDict, runtime_checkable


# ---------------------------------------------------------------------------
# Corpus de guias clinicas  (dueno: Isa)
# ---------------------------------------------------------------------------
class Chunk(TypedDict):
    """Un pedazo de guia clinica: la unidad de recuperacion."""
    chunk_id: str
    doc_id: str            # FK al manifest
    texto: str
    # 'fuente' citables: S07/S08 de la profesora SIEMPRE la llevan en el chunk
    # (el SYSTEM_RAG pide mencionar la fuente). Si el corpus no la trae, la
    # generacion cae a doc_id como respaldo.
    fuente: NotRequired[str]
    seccion: NotRequired[Optional[str]]


class ManifestEntry(TypedDict):
    """Procedencia y responsabilidad de un documento del corpus."""
    doc_id: str
    titulo: str
    fuente_url: str
    licencia: str
    fecha_publicacion: str   # ISO 8601
    fecha_indexado: str      # ISO 8601
    responsable: str         # quien valida esa fuente en el equipo


# ---------------------------------------------------------------------------
# Retrieval avanzado  (dueno: Pau)
# ---------------------------------------------------------------------------
class Fragment(TypedDict):
    """Fragmento recuperado, listo para aumentar el prompt de generacion."""
    chunk_id: str
    doc_id: str
    texto: str
    score: float
    # Aclaracion del contrato: 'score' debe ser comparable con el umbral de la
    # orquestacion. S08 fusiona rankings con RRF, que NO tiene score comparable;
    # se recomienda declarar de donde viene el score con 'score_tipo' (o
    # normalizarlo a [0, 1]). La orquestacion solo asume "mayor = mejor".
    score_tipo: NotRequired[str]   # "cosine" | "bm25" | "rrf" | "reranker"
    fuente: NotRequired[str]


@runtime_checkable
class RetrieveFn(Protocol):
    """Firma del retrieval de Pau: consulta en texto -> fragmentos ordenados."""
    def __call__(self, consulta: str, k: int) -> list[Fragment]: ...


# ---------------------------------------------------------------------------
# Tool de normalizacion terminologica  (dueno: Luis)
# ---------------------------------------------------------------------------
class NormalizationResult(TypedDict):
    entidad_original: str
    entidad_normalizada: str
    source_terminology: Optional[str]   # "SNOMED CT" | "UMLS" | None
    normalization_failed: bool          # explicito: nunca fallback silencioso


@runtime_checkable
class NormalizeFn(Protocol):
    def __call__(self, entidad: str) -> NormalizationResult: ...


# ---------------------------------------------------------------------------
# Generacion final y orquestacion  (dueno: Agustin)
# ---------------------------------------------------------------------------
class RespuestaRAG(TypedDict):
    """Salida de generar_respuesta: respuesta + trazabilidad."""
    answer: str
    sources_used: list[str]   # chunk_ids en que se APOYA la respuesta (vacío si fallback)
    fallback_used: bool       # True si no habia evidencia suficiente


class QueryResuelta(TypedDict):
    """Salida de resolver_query: que se busco, si se invoco la tool y por que."""
    query_final: str
    tool_invoked: bool
    tool_reason: str
    fragments: list[Fragment]   # los fragmentos YA recuperados (evita doble retrieval)


@runtime_checkable
class GenerateFn(Protocol):
    """Firma del generador: system + user -> texto (patron de S07/S10)."""
    def __call__(self, system: str, user: str) -> str: ...
