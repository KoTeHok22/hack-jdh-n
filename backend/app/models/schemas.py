from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    processing = "processing"
    indexed = "indexed"
    error = "error"


class DocumentType(str, Enum):
    textbook = "textbook"
    paper = "paper"
    report = "report"
    template = "template"
    excel = "excel"


class DocumentUpload(BaseModel):
    metadata: Optional[dict] = None


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    source: str
    doc_type: DocumentType
    authors: list[str]
    status: DocumentStatus
    created_at: datetime
    chunks_count: int = 0


class ChunkResponse(BaseModel):
    id: str
    document_id: UUID
    content: str
    page: Optional[int]
    section: Optional[str]
    score: float


class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    top_k: int = 10


class SearchResult(BaseModel):
    chunk: ChunkResponse
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ProblemCreate(BaseModel):
    statement: str
    target_kpi: Optional[str] = None
    target_delta: Optional[str] = None
    constraints: Optional[dict] = None
    domain: Optional[str] = None
    document_ids: Optional[list[UUID]] = None


class ProblemResponse(BaseModel):
    id: UUID
    statement: str
    target_kpi: Optional[str]
    target_delta: Optional[str]
    constraints: Optional[dict]
    domain: Optional[str]
    status: str
    created_at: datetime
    hypotheses: list["HypothesisResponse"] = []


class HypothesisResponse(BaseModel):
    id: UUID
    problem_id: UUID
    statement: str
    mechanism: str
    citations: list[str]
    novelty: float
    feasibility: float
    impact: float
    risk: float
    confidence: float
    composite_score: float
    reasoning_trace: Optional[str] = None
    risks: list[str]
    verification_plan: Optional[str] = None
    created_at: datetime


class HypothesisDetail(HypothesisResponse):
    source_chunks: list[ChunkResponse] = []


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class ExportFormat(str, Enum):
    json = "json"
    csv = "csv"
    pdf = "pdf"
