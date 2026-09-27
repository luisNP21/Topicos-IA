import re
from transformers import AutoTokenizer


def cargar_tokenizer(model_name: str) -> AutoTokenizer:
    """Se llama una sola vez en run(); mismo model_name que embeddings.py, ambos vienen de config.yaml."""
    return AutoTokenizer.from_pretrained(model_name)


def chunk_seccion(doc_id: str, seccion: dict, tokenizer: AutoTokenizer, max_tokens: int, overlap_tokens: int) -> list[dict]:
    """Contrato de salida: Chunk = {chunk_id, doc_id, texto, seccion} -- ver README de corpus."""
    palabras = seccion["texto"].split()
    chunks, i, idx = [], 0, 0
    while i < len(palabras):
        acumulado, n_tokens = [], 0
        for palabra in palabras[i:]:
            n_tok = len(tokenizer.encode(palabra, add_special_tokens=False))
            if n_tokens + n_tok > max_tokens:
                break
            acumulado.append(palabra)
            n_tokens += n_tok
        chunks.append({
            "chunk_id": f"{doc_id}_chunk{idx}",
            "doc_id": doc_id,
            "texto": " ".join(acumulado),
            "seccion": seccion["titulo"]
        })
        i += max(len(acumulado) - overlap_tokens, 1)
        idx += 1
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