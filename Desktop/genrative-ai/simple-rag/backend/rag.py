"""RAG core: local embeddings + FAISS + Gemini + SQLite answer cache. No web framework in here."""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =====================================================================
# Config
# =====================================================================
BASE_DIR = Path(__file__).parent
DOC_PATH = BASE_DIR / os.getenv("DOC_PATH", "data/document.pdf")   # .pdf or .txt
CACHE_DIR = BASE_DIR / ".cache"

# Free LOCAL embedding model (no API, no quota)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For French / multilingual documents:
# EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")  # use a current free Flash model from AI Studio
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

CACHE_DB = CACHE_DIR / "answers.db"
SEMANTIC_CACHE = True
SEMANTIC_THRESHOLD = 0.92

INDEX_DIR = CACHE_DIR / f"faiss_{re.sub(r'[^a-zA-Z0-9]+', '_', EMBED_MODEL)}"

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know. Cite sources like [1], [2].

Context:
{context}

Question: {question}
Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


class RAG:
    def __init__(self):
        # langchain-google-genai reads GOOGLE_API_KEY; accept GEMINI_API_KEY too
        if not os.getenv("GOOGLE_API_KEY"):
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY in backend/.env")
            os.environ["GOOGLE_API_KEY"] = key

        CACHE_DIR.mkdir(exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = self._load_or_build_index()
        llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0.2, max_retries=3)
        self.chain = PROMPT | llm | StrOutputParser()

        self.version = "|".join(
            [EMBED_MODEL, GEN_MODEL, str(TOP_K), str(CHUNK_SIZE), str(OVERLAP), self._doc_fingerprint()]
        )
        self._db_lock = threading.Lock()
        self.db = sqlite3.connect(CACHE_DB, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS answers (
                   key TEXT PRIMARY KEY, version TEXT, question TEXT,
                   embedding BLOB, answer TEXT, sources TEXT, created REAL)"""
        )
        self.db.commit()

    # ---------- index ----------
    @staticmethod
    def _doc_fingerprint() -> str:
        s = DOC_PATH.stat()
        return f"{s.st_size}-{s.st_mtime_ns}"

    def _load_or_build_index(self) -> FAISS:
        if INDEX_DIR.exists():
            # Safe: we created this pickle ourselves
            return FAISS.load_local(
                str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True
            )
        if not DOC_PATH.exists():
            raise FileNotFoundError(f"Document not found: {DOC_PATH}")
        loader = (
            PyPDFLoader(str(DOC_PATH))
            if DOC_PATH.suffix.lower() == ".pdf"
            else TextLoader(str(DOC_PATH), encoding="utf-8")
        )
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
        ).split_documents(loader.load())
        store = FAISS.from_documents(chunks, self.embeddings)
        store.save_local(str(INDEX_DIR))
        return store

    @property
    def chunk_count(self) -> int:
        return self.store.index.ntotal

    # ---------- answer cache ----------
    @staticmethod
    def _normalize(q: str) -> str:
        return " ".join(q.lower().split()).strip(" ?!.")

    def _key(self, q: str) -> str:
        return hashlib.sha256((self.version + "||" + self._normalize(q)).encode()).hexdigest()

    def _cache_lookup(self, question: str, q_vec: np.ndarray):
        with self._db_lock:
            row = self.db.execute(
                "SELECT answer, sources FROM answers WHERE key = ?", (self._key(question),)
            ).fetchone()
            if row:
                return row[0], json.loads(row[1]), "exact"
            if SEMANTIC_CACHE:
                rows = self.db.execute(
                    "SELECT embedding, answer, sources FROM answers WHERE version = ?",
                    (self.version,),
                ).fetchall()
            else:
                rows = []
        if rows:
            mat = np.vstack([np.frombuffer(r[0], dtype="float32") for r in rows])
            scores = mat @ q_vec  # vectors are normalized -> cosine similarity
            best = int(np.argmax(scores))
            if scores[best] >= SEMANTIC_THRESHOLD:
                return rows[best][1], json.loads(rows[best][2]), f"similar {scores[best]:.2f}"
        return None

    def _cache_store(self, question, q_vec, answer, sources):
        with self._db_lock:
            self.db.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self._key(question), self.version, question,
                    q_vec.astype("float32").tobytes(), answer,
                    json.dumps(sources), time.time(),
                ),
            )
            self.db.commit()

    def clear_cache(self):
        with self._db_lock:
            self.db.execute("DELETE FROM answers")
            self.db.commit()

    # ---------- main entry point ----------
    def ask(self, question: str, use_cache: bool = True):
        """Generator of events: meta -> token* -> done."""
        # Embed once (local, cheap); reuse for cache lookup and retrieval
        q_vec = np.array(self.embeddings.embed_query(question), dtype="float32")

        hit = self._cache_lookup(question, q_vec) if use_cache else None
        if hit:
            answer, sources, how = hit
            yield {"type": "meta", "cached": True, "how": how, "sources": sources}
            yield {"type": "token", "text": answer}
            yield {"type": "done"}
            return

        docs = self.store.similarity_search_by_vector(q_vec.tolist(), k=TOP_K)
        sources = [{"page": d.metadata.get("page"), "text": d.page_content} for d in docs]
        yield {"type": "meta", "cached": False, "how": None, "sources": sources}

        parts: list[str] = []
        for token in self.chain.stream({"context": _format_docs(docs), "question": question}):
            parts.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(parts)
        if use_cache and answer.strip():  # errors raise before this point, so they are never cached
            self._cache_store(question, q_vec, answer, sources)
        yield {"type": "done"}