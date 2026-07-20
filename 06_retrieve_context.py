"""
Step 6: Retrieve Context
--------------------------
Given a question and the platform (company) the user selected, retrieves
the best matching chunks and formats them into a single context string
that will be inserted into the LLM prompt.
"""
from importlib import import_module

vectors = import_module("04_vector_representation")


def build_context(index, question, company, k=8, max_sources=4):
    rows = vectors.hybrid_search(index, question, k=k, company=company)
    rows = [row for row in rows if row["score"] > 0][:max_sources]

    context_lines = []
    for source_number, row in enumerate(rows, start=1):
        context_lines.append(f"[Source {source_number}] {row['title']}\n{row['chunk_text']}")

    context_text = "\n\n".join(context_lines)
    return context_text, rows


if __name__ == "__main__":
    build_index = import_module("05_build_index").build_full_index
    index = build_index()
    context, sources = build_context(
        index, "Why do employee desks have wheels?", company="Valve"
    )
    print(context)
