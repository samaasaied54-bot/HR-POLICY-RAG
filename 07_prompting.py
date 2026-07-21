"""
Step 7: Prompting + Generation via OpenRouter (100% Dynamic & Free)
-------------------------------------------------------------------
Builds a grounded prompt from the retrieved context and sends it to an
LLM through OpenRouter (https://openrouter.ai) using ONLY free models.
"""
import os
import requests
from importlib import import_module

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

retrieve = import_module("06_retrieve_context")

load_dotenv()

# جلب المفتاح بأمان من Streamlit Secrets أو ملف البيئة
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


def get_available_free_models():
    """
    دالة ذكية تجلب قائمة الموديلات المجانية المتاحة والنشطة حالياً
    من OpenRouter مباشرة لمنع خطأ 404 للأبد.
    """
    fallback_free_models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
        "google/gemini-2.0-flash-lite-001:free",
    ]
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            # تصفية الموديلات التي ينتهي اسمها بـ :free فقط
            online_free_models = [m["id"] for m in data if m["id"].endswith(":free")]
            if online_free_models:
                return online_free_models
    except Exception:
        pass
    
    return fallback_free_models


def ask_openrouter(prompt, model=None):
    if not OPENROUTER_API_KEY:
        return "⚠️ مفتاح OPENROUTER_API_KEY غير موجود في Streamlit Secrets."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    # جلب الموديلات المجانية المتاحة لايف من السيرفر
    free_models = get_available_free_models()

    last_error = None
    # التجربة في الموديلات المجانية المتوفرة واحد تلو الآخر
    for m in free_models:
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

    return f"⚠️ تعذر الاتصال بموديلات OpenRouter المجانية: {last_error}"


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
