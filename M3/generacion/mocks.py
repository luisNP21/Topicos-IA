"""
Mocks de las piezas de Pau (retrieval) y Luis (normalizacion).

Permiten construir y probar generacion + orquestacion SIN depender de sus
modulos reales, que es justo lo que habilita el trabajo en paralelo por
contratos (acuerdo del equipo). Cuando sus piezas esten, se apunta
`run_generacion.py` a las reales y estos mocks quedan solo para los tests.

Los mocks respetan EXACTAMENTE los contratos de `contratos.py`.
"""

from __future__ import annotations

from contratos import Fragment, NormalizeFn, RetrieveFn

# Corpus de juguete del dominio (no es el corpus real de Isa: es solo para
# poder correr el pipeline de generacion de punta a punta mientras tanto).
CORPUS_MOCK: list[dict] = [
    {"chunk_id": "gpc_neumonia_c0", "doc_id": "gpc_neumonia", "fuente": "GPC Neumonía (2023)",
     "texto": ("La neumonía adquirida en la comunidad se trata empíricamente con amoxicilina "
               "en el paciente ambulatorio sano; en el paciente hospitalizado se considera "
               "ceftriaxona más un macrólido según gravedad.")},
    {"chunk_id": "gpc_diabetes_c0", "doc_id": "gpc_diabetes", "fuente": "GPC Diabetes tipo 2 (2024)",
     "texto": ("La diabetes mellitus tipo 2 se maneja con metformina como primera línea, "
               "más intervención en estilo de vida; se escala a otros antidiabéticos si no "
               "se alcanza la meta de HbA1c.")},
    {"chunk_id": "gpc_sepsis_c0", "doc_id": "gpc_sepsis", "fuente": "GPC Sepsis (2021)",
     "texto": ("La sepsis requiere antibiótico de amplio espectro en la primera hora y "
               "reanimación con cristaloides guiada por lactato y presión arterial media.")},
    {"chunk_id": "gpc_guillain_c0", "doc_id": "gpc_guillain",
     "fuente": "GPC Guillain-Barré (2020)",
     "texto": ("El síndrome de Guillain-Barré se trata con inmunoglobulina intravenosa o "
               "plasmaféresis; la ventilación se vigila por capacidad vital decreciente.")},
]


def crear_retriever_mock(corpus: list[dict] | None = None) -> RetrieveFn:
    """
    Retrieval de juguete: solapamiento de términos normalizado a [0, 1].
    Da scores comparables con el umbral (score_tipo="overlap"), suficiente para
    ejercitar el criterio de invocacion de la tool.
    """
    corpus = corpus if corpus is not None else CORPUS_MOCK

    def retrieve(consulta: str, k: int) -> list[Fragment]:
        terminos_consulta = set(consulta.lower().split())
        resultados = []
        for c in corpus:
            terminos_chunk = set(c["texto"].lower().split())
            score = len(terminos_consulta & terminos_chunk) / max(1, len(terminos_consulta))
            resultados.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                "texto": c["texto"], "score": round(float(score), 4),
                "score_tipo": "overlap", "fuente": c.get("fuente", c["doc_id"]),
            })
        resultados.sort(key=lambda x: -x["score"])
        return resultados[:k]

    return retrieve


# Mapa de normalizacion de juguete: entidad cruda -> termino controlado.
MAPA_NORMALIZACION_MOCK: dict[str, str] = {
    "epoc": "enfermedad pulmonar obstructiva crónica",
    "dm2": "diabetes mellitus tipo 2",
    "acv": "accidente cerebrovascular",
    "guillain barre": "síndrome de Guillain-Barré",
    "sindrome de guillain-barre": "síndrome de Guillain-Barré",
}


def crear_normalizador_mock(mapa: dict[str, str] | None = None) -> NormalizeFn:
    """
    Tool de normalizacion de juguete. Si la entidad no esta en el mapa devuelve
    `normalization_failed=True` (fallo explicito, no inventa un termino).
    """
    mapa = mapa if mapa is not None else MAPA_NORMALIZACION_MOCK

    def normalizar(entidad: str):
        clave = entidad.strip().lower()
        if clave in mapa:
            return {"entidad_original": entidad, "entidad_normalizada": mapa[clave],
                    "source_terminology": "SNOMED CT (mock)", "normalization_failed": False}
        return {"entidad_original": entidad, "entidad_normalizada": entidad,
                "source_terminology": None, "normalization_failed": True}

    return normalizar
