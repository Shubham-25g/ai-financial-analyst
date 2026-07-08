"""
Thin FAISS wrapper: build an index over the news corpus once, query it at
request time. Swap for Qdrant by replacing the three methods below (build,
save/load, search) — the rest of the app only depends on `search()`.
"""
from __future__ import annotations
import json
import numpy as np

from app import config
from app.rag.news_corpus import all_documents

_model = None  # lazy-loaded singleton
_using_fallback = False


class _HashingFallbackEmbedder:
    """
    Offline fallback used only when the real sentence-transformer model can't
    be downloaded (e.g. no internet / huggingface.co unreachable, as in this
    sandbox). Uses feature hashing over words — much weaker semantics than a
    real transformer, but keeps the RAG pipeline runnable end-to-end without
    network access. On a machine with normal internet access this path is
    never taken; `sentence-transformers` loads normally instead.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        import numpy as np
        import re
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for w in words:
                h = hash(w) % self.dim
                vecs[i, h] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vecs[i]) + 1e-8
                vecs[i] /= norm
        return vecs


def _get_embedder():
    global _model, _using_fallback
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as e:
            print(f"[vector_store] Could not load {config.EMBEDDING_MODEL} ({e}); "
                  f"falling back to offline hashing embedder.")
            _model = _HashingFallbackEmbedder()
            _using_fallback = True
    return _model


def build_index():
    import faiss

    docs = all_documents()
    embedder = _get_embedder()
    texts = [f"{d['headline']}. {d['text']}" for d in docs]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized dot product
    index.add(embeddings)

    faiss.write_index(index, str(config.VECTOR_INDEX_PATH))
    config.VECTOR_META_PATH.write_text(json.dumps(docs, indent=2))
    print(f"Built FAISS index with {len(docs)} documents -> {config.VECTOR_INDEX_PATH}")


def search(query: str, k: int = 5) -> list[dict]:
    import faiss

    if not config.VECTOR_INDEX_PATH.exists():
        build_index()

    index = faiss.read_index(str(config.VECTOR_INDEX_PATH))
    docs = json.loads(config.VECTOR_META_PATH.read_text())

    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    scores, idxs = index.search(q_emb, min(k, len(docs)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        doc = dict(docs[idx])
        doc["relevance_score"] = round(float(score), 4)
        results.append(doc)
    return results


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build_index()
    else:
        for r in search("AI chip export controls semiconductor risk", k=3):
            print(r["headline"], r["relevance_score"])
