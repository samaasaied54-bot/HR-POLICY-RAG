"""
Step 5: Build Index
----------------------
Wraps steps 1-4 into a single reusable function that downloads the
documents, chunks them, and builds the hybrid (BM25 + embeddings) index
once. The Streamlit app calls this and caches the result with
st.cache_resource so it only runs once per app session, not on every
question.
"""
from importlib import import_module

documents_module = import_module("01_documents")
chunking = import_module("03_chunking")
vectors = import_module("04_vector_representation")


def build_full_index():
    documents = documents_module.build_documents()
    chunks = chunking.build_chunks(documents=documents)
    index = vectors.build_index(chunks=chunks)
    index["documents"] = documents
    return index


def get_platform_list(index):
    """Returns the list of companies/platforms the user can choose from."""
    seen = []
    for doc in index["documents"]:
        if doc["company"] not in seen:
            seen.append(doc["company"])
    return seen


if __name__ == "__main__":
    idx = build_full_index()
    print("Platforms available:", get_platform_list(idx))
    print("Total chunks indexed:", len(idx["chunks"]))
