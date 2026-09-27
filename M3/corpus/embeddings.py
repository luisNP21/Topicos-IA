from sentence_transformers import SentenceTransformer


def cargar_modelo_embeddings(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embeber_query(query: str, modelo_embed: SentenceTransformer = None) -> list[float]:
    """
    Para uso de retrieval: convierte una query transformada en vector.
    Prefijo "query: " es obligatorio con e5 — asimétrico respecto al "passage: " de embeber_chunks.
    Si no se pasa modelo_embed, lo carga (útil para uso standalone fuera del pipeline).
    """
    if modelo_embed is None:
        modelo_embed = cargar_modelo_embeddings()
    return modelo_embed.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()


def embeber_chunks(chunks: list[dict], modelo_embed: SentenceTransformer = None) -> list[list[float]]:
    textos = [f"passage: {c['texto']}" for c in chunks]
    return modelo_embed.encode(textos, normalize_embeddings=True, show_progress_bar=True).tolist()