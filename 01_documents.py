"""
Step 1: Documents
------------------
Downloads real corporate HR / code-of-conduct PDF documents from a public
GitHub repository and extracts their raw text with PyMuPDF.

Source repository:
https://github.com/Heps-akint/policy-rag-chatbot
(documents are published by the companies themselves and are used here
for internal / educational purposes)
"""
import re

import fitz  # PyMuPDF
import requests

BASE_URL = "https://raw.githubusercontent.com/Heps-akint/policy-rag-chatbot/master/data/raw"

# Each entry is one "platform" the user can pick in the app.
SOURCE_FILES = {
    0: {"company": "Google", "title": "Google Code of Conduct", "filename": "google_code_of_conduct.pdf"},
    1: {"company": "Vanderbloemen", "title": "Remote Work Policy", "filename": "vanderbloemen_remote_work_policy.pdf"},
    2: {"company": "Boeing", "title": "Ethical Business Conduct Guidelines", "filename": "boeing_conduct_guidelines.pdf"},
    3: {"company": "Walmart", "title": "Code of Conduct", "filename": "walmart_code_of_conduct.pdf"},
    4: {"company": "UnitedHealth Group", "title": "Code of Conduct", "filename": "unitedhealth_group_code_of_conduct.pdf"},
    5: {"company": "Valve", "title": "Handbook for New Employees", "filename": "valve_employee_handbook.pdf"},
}


def download_and_extract_text(filename):
    url = f"{BASE_URL}/{filename}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf_document = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text() + "\n"
    pdf_document.close()

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_documents():
    documents = []
    for doc_id, meta in SOURCE_FILES.items():
        full_text = download_and_extract_text(meta["filename"])
        documents.append(
            {
                "id": doc_id,
                "company": meta["company"],
                "title": f"{meta['company']} — {meta['title']}",
                "source_url": f"{BASE_URL}/{meta['filename']}",
                "text": full_text,
            }
        )
    return documents


if __name__ == "__main__":
    docs = build_documents()
    for d in docs:
        print(f"{d['company']:20s} | {len(d['text']):>7,} chars | {d['title']}")
