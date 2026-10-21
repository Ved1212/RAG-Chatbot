"""Step 2: split pages into overlapping chunks so each fits a focused search."""


def chunk_pages(pages: list[dict], size: int = 800, overlap: int = 150) -> list[dict]:
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append({"text": piece, "source": p["source"], "page": p["page"]})
            start += size - overlap
    return chunks
