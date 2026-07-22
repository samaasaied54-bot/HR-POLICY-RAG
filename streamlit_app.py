"""
Corporate Policy RAG Chatbot — Streamlit UI
---------------------------------------------
Step 1: the user picks the company/platform they want to ask about.
Step 2: the user asks questions and gets grounded answers with sources,
        restricted to that company's policy document.

Run locally:
    streamlit run streamlit_app.py

Deploy: push this whole folder to GitHub, then deploy on
        https://share.streamlit.io and set OPENROUTER_API_KEY in
        the app's Secrets.
"""
from importlib import import_module

import streamlit as st

preprocessing = import_module("02_preprocessing")
build_index_module = import_module("05_build_index")
rag = import_module("07_prompting")

st.set_page_config(page_title="Policy RAG Chatbot", page_icon="📄")


# ---- Load API key from Streamlit secrets if not in the environment ----
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


@st.cache_resource(show_spinner="Downloading policy documents and building the index...")
def load_index():
    preprocessing.ensure_nltk_data()
    return build_index_module.build_full_index()


index = load_index()
platforms = build_index_module.get_platform_list(index)

st.title("📄 Corporate Policy RAG Chatbot")
st.caption("Ask questions about a company's official HR / code-of-conduct policy.")

# ---------- Step 1: choose the platform / company ----------
selected_company = st.selectbox(
    "1) Choose the company whose policy you want to ask about:",
    platforms,
)

if "messages" not in st.session_state:
    st.session_state.messages = {}
if selected_company not in st.session_state.messages:
    st.session_state.messages[selected_company] = []

st.divider()
st.subheader(f"2) Ask a question about {selected_company}'s policy")

for message in st.session_state.messages[selected_company]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input(f"Ask something about {selected_company}'s policy...")

if user_question:
    st.session_state.messages[selected_company].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating an answer..."):
            answer, sources = rag.answer_question(index, user_question, company=selected_company)

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**[Source {i}] {source['title']}**")
                    st.caption(source["chunk_text"][:300] + "...")

    st.session_state.messages[selected_company].append({"role": "assistant", "content": answer})

import streamlit as st

# 🎯 تصميم مخصص بـ CSS
st.markdown("""
    <style>
    /* تغيير نوع الخط في الصفحة كلها */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* جعل الحواف دائرية وإضافة ظل خفيف للـ Selectbox والـ Input */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        border-radius: 12px !important;
        border: 1px solid #3B82F6 !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* تغيير شكل زرار الإرسال */
    .stButton>button {
        border-radius: 10px;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
    }
    
    /* إضافة كارت أنيق حول إجابات البوت */
    .stChatMessage {
        border-radius: 15px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)
