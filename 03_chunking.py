"""
Step 3: Chunking
------------------
Splits each document into overlapping word chunks so that retrieval can
work on small, focused pieces of text instead of whole documents.
"""
from importlib import import_module

documents_module = import_module("01_documents")
preprocess_text = import_module("02_preprocessing").preprocess_text


def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def build_chunks(documents=None):
    documents = documents if documents is not None else documents_module.build_documents()
    rows = []

    for document in documents:
        for chunk_index, chunk in enumerate(chunk_text(document["text"])):
            rows.append(
                {
                    "chunk_id": f"{document['id']}_{chunk_index}",
                    "document_id": document["id"],
                    "company": document["company"],
                    "title": document["title"],
                    "chunk_text": chunk,
                    "search_text": preprocess_text(f"{document['title']} {chunk}"),
                }
            )

    return rows


if __name__ == "__main__":
    chunks = build_chunks()
    print("Total chunks:", len(chunks))
    print(chunks[0])
