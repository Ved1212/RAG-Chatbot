"""Step 3 + 4: embed chunks and store/search them in a vector database."""
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # small, free, runs locally
    return _model


class VectorStore:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.col = self.client.get_or_create_collection(
            f"docs_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict]) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = get_model().encode(texts, normalize_embeddings=True).tolist()
        self.col.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )

    def search(self, query: str, k: int = 4) -> list[dict]:
        emb = get_model().encode([query], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=emb, n_results=k)
        return [
            {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
