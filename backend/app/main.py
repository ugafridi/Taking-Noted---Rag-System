"""FastAPI entry point. Run:  uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import BACKEND_DIR, settings
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from rag.pipeline import RAGSystem

FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGSystem.load()
    yield


app = FastAPI(
    title="Course Notes RAG API",
    description="Grounded question answering over the 13 Week 1-8 course guides.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rag(request: Request) -> RAGSystem:
    return request.app.state.rag


if not FRONTEND_DIST.exists():
    @app.get("/")
    def root():
        return {"name": "Course Notes RAG API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    return rag(request).health()


@app.get("/api/documents", response_model=list[DocumentInfo])
def documents(request: Request):
    return rag(request).documents()


@app.post("/api/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request):
    """Retrieval only, no LLM call. Use this to inspect scores cheaply."""
    return rag(request).search(payload.question, payload.k, payload.mode)


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    return rag(request).ask(payload.question, payload.k, payload.mode, payload.threshold)


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest, request: Request):
    """Server-sent events: meta -> token* -> done."""
    system = rag(request)

    def events():
        for event in system.ask_stream(
            payload.question, payload.k, payload.mode, payload.threshold
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request, x_admin_key: str | None = Header(None)):
    """Rebuild the index (e.g. after adding a document or changing chunk size)."""
    if x_admin_key != settings.ingest_admin_key:
        raise HTTPException(status_code=403, detail="Invalid X-Admin-Key")
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be < chunk_size")
    return rag(request).rebuild(payload.chunk_size, payload.chunk_overlap)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
