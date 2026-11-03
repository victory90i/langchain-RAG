import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================================
# Config
# =====================================================================
DOC_PATH = Path("data/document.pdf")          # .pdf or .txt
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Free LOCAL embedding model (no API, no quota)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For French / multilingual documents:
# EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GEN_MODEL = "gemini-2.5-flash"                # swap for a current free Flash model from AI Studio
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

# Answer cache settings
CACHE_DB = CACHE_DIR / "answers.db"
SEMANTIC_CACHE = True          # also match *similar* questions, not just identical ones
SEMANTIC_THRESHOLD = 0.92      # higher = stricter (fewer false hits); tune on your own questions

INDEX_DIR = CACHE_DIR / f"faiss_{re.sub(r'[^a-zA-Z0-9]+', '_', EMBED_MODEL)}"

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing API key. Add GEMINI_API_KEY to .streamlit/secrets.toml.")
        st.stop()


# =====================================================================
# Models + index  (cached in memory by Streamlit, index also cached on disk)
# =====================================================================
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embeddings = get_embeddings()
llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0.2, max_retries=3)


@st.cache_resource(show_spinner="Indexing document...")
def get_vector_store() -> FAISS:
    if INDEX_DIR.exists():
        # Safe: we created this pickle ourselves
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    if not DOC_PATH.exists():
        st.error(f"Document not found: {DOC_PATH}")
        st.stop()
    loader = (
        PyPDFLoader(str(DOC_PATH))
        if DOC_PATH.suffix.lower() == ".pdf"
        else TextLoader(str(DOC_PATH), encoding="utf-8")
    )
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    ).split_documents(loader.load())
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    return store


# =====================================================================
# Answer cache (SQLite): exact match first, then semantic match
# =====================================================================
def _doc_fingerprint() -> str:
    s = DOC_PATH.stat()
    return f"{s.st_size}-{s.st_mtime_ns}"


# Anything that changes the answer must be in the version, or you get stale hits
CACHE_VERSION = "|".join(
    [EMBED_MODEL, GEN_MODEL, str(TOP_K), str(CHUNK_SIZE), str(OVERLAP), _doc_fingerprint()]
)


@st.cache_resource
def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(CACHE_DB, check_same_thread=False)
    db.execute(
        """CREATE TABLE IF NOT EXISTS answers (
               key TEXT PRIMARY KEY,
               version TEXT,
               question TEXT,
               embedding BLOB,
               answer TEXT,
               sources TEXT,
               created REAL)"""
    )
    db.commit()
    return db


def normalize_question(q: str) -> str:
    return " ".join(q.lower().split()).strip(" ?!.")


def cache_key(q: str) -> str:
    return hashlib.sha256((CACHE_VERSION + "||" + normalize_question(q)).encode()).hexdigest()


def cache_lookup(question: str, q_vec: np.ndarray):
    """Return (answer, sources, how) or None."""
    db = get_db()

    # 1) Exact (normalized) match
    row = db.execute(
        "SELECT answer, sources FROM answers WHERE key = ?", (cache_key(question),)
    ).fetchone()
    if row:
        return row[0], json.loads(row[1]), "exact"

    # 2) Semantic match: cosine similarity against cached questions (vectors are normalized)
    if SEMANTIC_CACHE:
        rows = db.execute(
            "SELECT embedding, answer, sources FROM answers WHERE version = ?",
            (CACHE_VERSION,),
        ).fetchall()
        if rows:
            mat = np.vstack([np.frombuffer(r[0], dtype="float32") for r in rows])
            scores = mat @ q_vec
            best = int(np.argmax(scores))
            if scores[best] >= SEMANTIC_THRESHOLD:
                return rows[best][1], json.loads(rows[best][2]), f"similar ({scores[best]:.2f})"
    return None


def cache_store(question: str, q_vec: np.ndarray, answer: str, sources: list[dict]):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            cache_key(question),
            CACHE_VERSION,
            question,
            q_vec.astype("float32").tobytes(),
            answer,
            json.dumps(sources),
            time.time(),
        ),
    )
    db.commit()


def cache_clear():
    get_db().execute("DELETE FROM answers")
    get_db().commit()


# =====================================================================
# RAG chain
# =====================================================================
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know. Cite sources like [1], [2].

Context:
{context}

Question: {question}
Answer:"""
)
chain = prompt | llm | StrOutputParser()


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def show_sources(sources: list[dict]):
    with st.expander("Retrieved context"):
        for i, s in enumerate(sources, 1):
            label = f"page {s['page'] + 1}" if s["page"] is not None else "n/a"
            st.markdown(f"**[{i}]** {label}\n\n{s['text']}")


# =====================================================================
# Streamlit UI
# =====================================================================
st.set_page_config(page_title="Chat with your document", page_icon="📄")
st.title("📄 Chat with your document (LangChain + Gemini)")

store = get_vector_store()

with st.sidebar:
    use_cache = st.toggle("Use answer cache", value=True)
    if st.button("Clear answer cache"):
        cache_clear()
        st.success("Cache cleared.")
    n = get_db().execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    st.caption(f"{n} cached answers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # Embed the question ONCE (local, cheap) and reuse it for cache + retrieval
            q_vec = np.array(embeddings.embed_query(question), dtype="float32")

            hit = cache_lookup(question, q_vec) if use_cache else None
            if hit:
                answer, sources, how = hit
                st.markdown(answer)
                st.caption(f"⚡ From cache ({how}), no Gemini call")
            else:
                docs = store.similarity_search_by_vector(q_vec.tolist(), k=TOP_K)
                sources = [
                    {"page": d.metadata.get("page"), "text": d.page_content} for d in docs
                ]
                answer = st.write_stream(
                    chain.stream({"context": format_docs(docs), "question": question})
                )
                if use_cache and answer.strip():
                    cache_store(question, q_vec, answer, sources)
            show_sources(sources)
        except Exception as e:
            # Errors are never cached
            answer = f"⚠️ Request failed: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})