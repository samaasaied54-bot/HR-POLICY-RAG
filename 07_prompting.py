"""
Step 7: Prompting + Generation via OpenRouter
------------------------------------------------
Builds a grounded prompt from the retrieved context and sends it to an
LLM through OpenRouter (https://openrouter.ai).
"""
import os
from importlib import import_module

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

retrieve = import_module("06_retrieve_context")

load_dotenv()

# Get API key safely from Streamlit secrets or environment
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))


def build_prompt(question, company, context_text):
    return f"""You are a careful corporate-policy assistant for {company}.

STRICT RULES:
- Use ONLY the information inside the Context section below. Never use outside knowledge.
- If the Context does not contain enough information, respond exactly:
  "The provided policy documents do not contain enough information to answer this question."
- Always cite the sources you used, e.g. [Source 1].

Question:
{question}

Context:
{context_text}

Answer:"""


def ask_openrouter(prompt, model=None):
    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY in Streamlit Secrets."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    candidate_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "deepseek/deepseek-r1:free",
        "qwen/qwen-2.5-7b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ]
    
    if model:
        candidate_models.insert(0, model)

    last_error = None
    for m in candidate_models:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    return f"Error contacting OpenRouter models: {last_error}"


def answer_question(index, question, company, model=None):
    context_text, sources = retrieve.build_context(index, question, company=company)
    prompt = build_prompt(question, company, context_text)

    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY. Add it to your .env file or Streamlit secrets.", sources

    answer = ask_openrouter(prompt, model=model)
    return answer, sources


if __name__ == "__main__":
    build_index = import_module("05_build_index").build_full_index
    index = build_index()
    answer, sources = answer_question(index, "Why do employee desks have wheels?", company="Valve")
    print(answer)
