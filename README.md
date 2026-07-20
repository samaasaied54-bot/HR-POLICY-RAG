# Corporate Policy RAG Chatbot

RAG chatbot بيسأل عن سياسات شركات حقيقية (Google, Boeing, Walmart, UnitedHealth Group,
Valve, Vanderbloemen) — المستخدم بيختار الشركة (platform) الأول، وبعدين يسأل أي سؤال
عن سياستها، والإجابة بتيجي مبنية على المصادر الحقيقية فقط (grounded) عن طريق OpenRouter.

## هيكل الملفات (نفس تقسيمة simple_rag_lab)

```text
01_documents.py            -> تنزيل الـ PDFs الحقيقية واستخراج النص
02_preprocessing.py        -> تنظيف النص (lowercase, stopwords, lemmatization)
03_chunking.py             -> تقسيم كل مستند لقطع (chunks) متداخلة
04_vector_representation.py-> hybrid retrieval: BM25 + all-MiniLM-L6-v2 embeddings
05_build_index.py          -> يجمع الخطوات 1-4 في index واحد قابل لإعادة الاستخدام
06_retrieve_context.py     -> يجيب أفضل الـ chunks المتعلقة بالسؤال + الشركة المختارة
07_prompting.py            -> يبني الـ prompt ويكلم OpenRouter عشان يطلع الإجابة
streamlit_app.py           -> الواجهة: (1) اختيار الشركة -> (2) السؤال
requirements.txt
.env.example
```

الـ hybrid retrieval:

```text
hybrid_score = 0.4 * BM25 + 0.6 * all-MiniLM-L6-v2 embeddings
```

---

## الخطوات بالتفصيل

### 1) تشغيل المشروع على جهازك (تجربة محلية)

```bash
cd policy-rag-app
python -m venv .venv
source .venv/bin/activate        # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

افتحي ملف `.env` وحطي مفتاح OpenRouter الحقيقي بتاعك:

```text
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

طريقة الحصول على المفتاح:
1. روحي على https://openrouter.ai وسجلي حساب (بالإيميل أو Google).
2. من الداشبورد اختاري **Keys** ثم **Create Key**.
3. انسخي المفتاح (يبدأ بـ `sk-or-v1-...`) والصقيه في `.env`.
4. فيه موديلات مجانية زي `meta-llama/llama-3.1-8b-instruct:free` أو
   `google/gemini-2.0-flash-exp:free` — تقدري تشوفي القائمة كاملة في
   https://openrouter.ai/models

### 2) تجربة كل خطوة لوحدها (اختياري، للفهم)

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_build_index.py
python 06_retrieve_context.py
python 07_prompting.py
```

### 3) تشغيل الواجهة محليًا

```bash
streamlit run streamlit_app.py
```

هيفتح المتصفح، هتلاقي:
- Dropdown لاختيار الشركة (platform).
- تحت منه صندوق شات تكتبي فيه سؤالك عن سياسة الشركة دي، وهيردّ عليكي
  مع ذكر المصادر (Sources) اللي جاب منها الإجابة.

### 4) رفع المشروع على GitHub

```bash
cd policy-rag-app
git init
git add .
git commit -m "Corporate Policy RAG chatbot with OpenRouter"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

> استبدلي `USERNAME/REPO_NAME` باسم اليوزر بتاعك واسم الريبو اللي عملتيه على
> GitHub (لازم تعمليه فاضي الأول من على الموقع نفسه، بدون README حتى
> يبقى الـ push سهل).

⚠️ ملف `.env` متسجلش في Git خالص (موجود في `.gitignore`) عشان متسربيش
المفتاح بتاعك بالغلط.

### 5) النشر على Streamlit Community Cloud

1. روحي على https://share.streamlit.io وسجلي دخول بحساب GitHub.
2. دوسي **New app**.
3. اختاري الريبو اللي رفعتيه، والـ branch (`main`)، وملف الدخول
   `streamlit_app.py`.
4. قبل ما تدوسي Deploy، افتحي **Advanced settings -> Secrets** وحطي:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```

5. دوسي **Deploy**. أول تشغيل هياخد شوية وقت لأنه هيحمّل موديل
   الـ embeddings وينزل الـ PDFs، بعدها هيبقى فيه لينك عام (public URL)
   تقدري تبعتيه لأي حد.

---

## ملاحظات مهمة

- المستندات بتتنزل كل مرة الأبلكيشن يقوم (لأن Streamlit Cloud مساحته
  مؤقتة/ephemeral)، لكن `st.cache_resource` بيمنع إعادة بنائها مع كل
  سؤال — بس مرة واحدة لكل جلسة سيرفر.
- لو حابة تضيفي شركات تانية، زودي عليها في `SOURCE_FILES` جوه
  `01_documents.py`.
- لو غيرتي الموديل في OpenRouter، تأكدي إن اسمه مطابق تمامًا للي في
  https://openrouter.ai/models
