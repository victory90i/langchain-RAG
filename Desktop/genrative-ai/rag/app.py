import os
import re
from pathlib import Path

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
DOC_PATH = Path("data/next-js.pdf")          # .pdf or .txt
CACHE_DIR = Path(".cache")

# Free LOCAL embedding model (runs on your machine, no API, no quota).
# English, small and fast:
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For French / multilingual documents, use this instead:
# EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GEN_MODEL = "gemini-2.5-flash"                # swap for a current free Flash model from AI Studio

# MiniLM models only read ~128-256 tokens, so keep chunks small (long chunks get truncated)
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

# Index folder is named after the embedding model, so switching models
# never loads an incompatible old index.
INDEX_DIR = CACHE_DIR / f"faiss_{re.sub(r'[^a-zA-Z0-9]+', '_', EMBED_MODEL)}"

# API key (used only for the Gemini chat model now)
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing API key. Add GEMINI_API_KEY to .streamlit/secrets.toml.")
        st.stop()


# =====================================================================
# Models
# =====================================================================
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine-equivalent ranking
    )


embeddings = get_embeddings()
llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0.2, max_retries=3)


# =====================================================================
# Indexing
# =====================================================================
@st.cache_resource(show_spinner="Indexing document...")
def get_vector_store() -> FAISS:
    # Finished index exists -> load it (safe: we created this pickle ourselves)
    if INDEX_DIR.exists():
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

    store = FAISS.from_documents(chunks, embeddings)   # local, so no rate limits
    CACHE_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    return store


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


# =====================================================================
# Streamlit UI
# =====================================================================
st.set_page_config(page_title="Chat with your document", page_icon="📄")
st.title("📄 Chat with your document (LangChain + Gemini)")

store = get_vector_store()
retriever = store.as_retriever(search_kwargs={"k": TOP_K})
st.caption(f"Embeddings: {EMBED_MODEL} (local) · Answers: {GEN_MODEL}")

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
            docs = retriever.invoke(question)
            answer = st.write_stream(
                chain.stream({"context": format_docs(docs), "question": question})
            )
            with st.expander("Retrieved context"):
                for i, d in enumerate(docs, 1):
                    page = d.metadata.get("page")
                    label = f"page {page + 1}" if page is not None else "n/a"
                    st.markdown(f"**[{i}]** {label}\n\n{d.page_content}")
        except Exception as e:
            answer = f"⚠️ Request failed: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})