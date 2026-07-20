"""
Step 4: Vector Representation
--------------------------------
Builds the hybrid retriever: BM25 (lexical) + sentence-transformer
embeddings (semantic), combined as:

    hybrid_score = (1 - ALPHA) * BM25   +   ALPHA * embeddings
"""
from importlib import import_module

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

preprocessing = import_module("02_preprocessing")
chunking = import_module("03_chunking")

ALPHA = 0.6
MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(chunks=None):
    chunks = chunks if chunks is not None else chunking.build_chunks()

    tokenized_chunks = [chunk["search_text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(
        [chunk["search_text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return {
        "chunks": chunks,
        "bm25": bm25,
        "model": model,
        "chunk_embeddings": chunk_embeddings,
    }


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(index, query, k=4, company=None):
    chunks = index["chunks"]
    clean_query = preprocessing.preprocess_text(query)

    bm25_scores = index["bm25"].get_scores(clean_query.split())
    query_embedding = index["model"].encode(
        [clean_query], convert_to_numpy=True, normalize_embeddings=True
    )
    embedding_scores = cosine_similarity(query_embedding, index["chunk_embeddings"]).flatten()

    hybrid_scores = (1 - ALPHA) * min_max_normalize(bm25_scores) + ALPHA * min_max_normalize(
        embedding_scores
    )

    order = np.argsort(hybrid_scores)[::-1]

    results = []
    for i in order:
        chunk = chunks[i]
        if company is not None and chunk["company"] != company:
            continue
        results.append({**chunk, "score": float(hybrid_scores[i])})
        if len(results) == k:
            break

    return results


if __name__ == "__main__":
    index = build_index()
    for row in hybrid_search(index, "Why do employee desks have wheels?", k=3, company="Valve"):
        print(round(row["score"], 3), row["title"], "->", row["chunk_text"][:80])
