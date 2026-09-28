import re
from transformers import AutoTokenizer


def cargar_tokenizer(model_name: str) -> AutoTokenizer:
    """Se llama una sola vez en run(); mismo model_name que embeddings.py, ambos vienen de config.yaml."""
    return AutoTokenizer.from_pretrained(model_name)


def chunk_seccion(doc_id: str, seccion: dict, tokenizer, max_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Ventanas por tokens con overlap medido en tokens. Se detiene al llegar al final
    de la seccion (sin cola de chunks-sufijo). Los tokens por palabra se calculan una sola vez.
    """
    palabras = seccion["texto"].split()
    if not palabras:
        return []
    tok_por_palabra = [len(tokenizer.encode(p, add_special_tokens=False)) for p in palabras]
    n = len(palabras)

    chunks, i, idx = [], 0, 0
    while i < n:
        j, n_tokens = i, 0
        while j < n and (n_tokens + tok_por_palabra[j] <= max_tokens or j == i):
            n_tokens += tok_por_palabra[j]
            j += 1

        chunks.append({
            "chunk_id": f"{doc_id}_chunk{idx}",
            "doc_id": doc_id,
            "texto": " ".join(palabras[i:j]),
            "seccion": seccion["titulo"],
        })
        idx += 1

        if j >= n:  # el chunk llego al final de la seccion: no generar cola
            break

        # retroceder hasta cubrir overlap_tokens (en tokens, no en palabras)
        k, ov = j, 0
        while k > i + 1 and ov + tok_por_palabra[k - 1] <= overlap_tokens:
            ov += tok_por_palabra[k - 1]
            k -= 1
        i = k  # k > i siempre, hay progreso garantizado

    return chunks


def renumerar_chunks(doc_id: str, chunks: list[dict]) -> list[dict]:
    """Asigna chunk_id secuencial unico a nivel de documento completo, no por seccion."""
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}_chunk{i}"
    return chunks


def clasificar_seccion(titulo: str, patrones_categoria: dict) -> str:
    for categoria, patron in patrones_categoria.items():
        if patron.search(titulo or ""):
            return categoria
    return "otro"