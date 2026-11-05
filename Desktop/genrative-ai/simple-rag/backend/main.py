"""FastAPI wrapper around the RAG core. Run: uvicorn main:app --reload --port 8000"""
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import RAG

log = logging.getLogger("rag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAG()  # loads the embedding model and builds/loads the index once
    yield


app = FastAPI(title="Simple RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_cache: bool = True


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "chunks": request.app.state.rag.chunk_count}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    rag: RAG = request.app.state.rag

    def event_stream():
        try:
            for event in rag.ask(req.question.strip(), req.use_cache):
                yield json.dumps(event) + "\n"          # NDJSON: one JSON event per line
        except Exception as e:
            log.exception("chat failed")
            yield json.dumps({"type": "error", "message": str(e)[:300]}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/cache")
def clear_cache(request: Request):
    request.app.state.rag.clear_cache()
    return {"status": "cleared"}