"""
Step 7: Prompting + Generation via OpenRouter
------------------------------------------------
Builds a grounded prompt from the retrieved context and sends it to an
LLM through OpenRouter (https://openrouter.ai), which gives access to
many models (OpenAI, Anthropic, Google, Meta, ...) behind one API key.
"""
import os
from importlib import import_module

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st  # تم إضافة ستريمليت لجلب المفتاح بأمان

retrieve = import_module("06_retrieve_context")

load_dotenv()

# 1. جلب المفتاح بشكل ديناميكي وآمن من الـ Secrets أو ملف الـ .env (تم مسح المفتاح القديم المخترق)
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))

# 2. الموديل الافتراضي المستقر والمجاني حالياً على أوبن راوتر
OPENROUTER_MODEL = "google/gemini-2.5-flash:free"


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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    # 🎯 إجبار الكود على استخدام موديل شغال ومجاني حالياً لتفادي خطأ 404
    # حتى لو ملف streamlit_app.py بيبعت اسم موديل قديم ومحذوف
    chosen_model = "google/gemini-2.5-flash:free"
    
    response = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


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
