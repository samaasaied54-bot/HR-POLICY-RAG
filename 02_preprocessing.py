"""
Step 2: Preprocessing
----------------------
Cleans and normalizes text before it is used for lexical (BM25) search.
Semantic embeddings use the raw chunk text instead, so this step mainly
helps the keyword side of the hybrid retriever.
"""
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
translator = str.maketrans("", "", string.punctuation)
protected_negation_words = {"no", "not", "nor", "never"}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = {"the", "is", "and", "a", "an", "of", "to", "in", "for", "with", "on"}


def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)


def safe_lemmatize(token, pos="v"):
    token = token.lower()
    try:
        return lemmatizer.lemmatize(token, pos=pos)
    except LookupError:
        return token


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = safe_word_tokenize(text)
    tokens = [
        token
        for token in tokens
        if token not in stop_words or token in protected_negation_words
    ]
    tokens = [safe_lemmatize(token, pos="v") for token in tokens]
    return " ".join(tokens)


def ensure_nltk_data():
    """Download the small NLTK resources needed for tokenizing/lemmatizing."""
    import nltk

    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


if __name__ == "__main__":
    ensure_nltk_data()
    sample = "Employees are NOT allowed to accept gifts. Dropping a class does not create a refund."
    print(preprocess_text(sample))
