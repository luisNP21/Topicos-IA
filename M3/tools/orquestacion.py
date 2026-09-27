"""
orquestacion.py
Decide CUANDO invocar tool_normalizacion -- M3

Regla de negocio de invocacion: si el retrieval con la entidad cruda ya
devuelve un score suficiente, no se invoca la tool (es caro llamar a una
API externa sin necesidad). Solo se normaliza cuando el retrieval falla,
y solo se usa la entidad normalizada si la normalizacion tuvo exito.
"""

from tool_normalizacion import normalizar_entidad


def resolver_query(entidad: str, retrieve_fn, umbral: float) -> dict:
    """
    Decide si invocar la tool de normalizacion en funcion del score de retrieval.

    entidad     : texto crudo del encoder (salida de M1/M2)
    retrieve_fn : funcion de retrieval del compañero encargado de esa parte.
                  Firma esperada: (query: str, k: int) -> list[dict]
                  Cada dict debe tener al menos {"score": float}.
                  Se recibe como parametro para no acoplarse a un modulo
                  de retrieval que puede no existir todavia.
    umbral      : cfg["tool_normalizacion"]["umbral_score_retrieval"]

    Devuelve:
    {
        "query_final"  : str,   -- query que se debe usar para retrieval final
        "tool_invoked" : bool,  -- si se invoco la tool
        "tool_reason"  : str    -- razon legible de la decision
    }
    """
    resultados = retrieve_fn(entidad, k=5)
    mejor_score = resultados[0]["score"] if resultados else 0.0

    if mejor_score >= umbral:
        return {
            "query_final": entidad,
            "tool_invoked": False,
            "tool_reason": f"score suficiente ({mejor_score:.2f}) con entidad cruda",
        }

    norm = normalizar_entidad(entidad)
    if norm["normalization_failed"]:
        return {
            "query_final": entidad,
            "tool_invoked": True,
            "tool_reason": (
                f"score bajo ({mejor_score:.2f}), normalizacion sin match "
                "-- se mantiene entidad cruda"
            ),
        }

    return {
        "query_final": norm["entidad_normalizada"],
        "tool_invoked": True,
        "tool_reason": (
            f"score bajo ({mejor_score:.2f}) -- "
            f"normalizada via {norm['source_terminology']}"
        ),
    }



# Funciones mock para smoke-test sin modulo de retrieval real


def _retrieve_mock_bueno(query: str, k: int) -> list[dict]:
    return [{"chunk_id": "c1", "score": 0.8, "texto": "..."}]


def _retrieve_mock_malo(query: str, k: int) -> list[dict]:
    return [{"chunk_id": "c1", "score": 0.1, "texto": "..."}]


if __name__ == "__main__":
    print(resolver_query("diabetes tipo 2", _retrieve_mock_bueno, umbral=0.5))
    print(resolver_query("diabetes tipo 2", _retrieve_mock_malo, umbral=0.5))
    print(resolver_query("una enfermedad rarisima inventada", _retrieve_mock_malo, umbral=0.5))
