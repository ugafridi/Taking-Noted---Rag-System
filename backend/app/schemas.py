"""Request / response contracts shared by the API and the React client."""
from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["semantic", "hybrid"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=10)
    mode: Mode = "semantic"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "semantic"


class Hit(BaseModel):
    rank: int
    score: float
    source: str
    chunk: str


class AskResponse(BaseModel):
    answer: str
    hits: list[Hit]
    top_score: float
    refused: bool
    model: str
    mode: Mode
    k: int
    latency_ms: float


class SearchResponse(BaseModel):
    question: str
    hits: list[Hit]
    top_score: float
    mode: Mode


class DocumentInfo(BaseModel):
    name: str
    characters: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks: int
    documents: int
    embedding_model: str
    embedding_dim: int
    index_backend: str
    llm_provider: str
    llm_model: str
    threshold: float


class IngestRequest(BaseModel):
    chunk_size: int = Field(default=300, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class IngestResponse(BaseModel):
    documents: int
    characters: int
    chunks: int
    average_chunk_length: float
    longest_chunk: int
    shortest_chunk: int
    dimension: int
    index_backend: str
