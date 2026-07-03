"""Pydantic schemas shared across the API."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class ChunkResult(BaseModel):
    index: int
    text: str
    start_char: int
    end_char: int
    perplexity: float
    burstiness: float
    repetition_score: float
    ai_probability: float  # 0-100, this chunk only
    verdict: str  # "human", "mixed", "ai"


class DocumentSummary(BaseModel):
    filename: str
    total_chars: int
    total_chunks: int
    word_count: int
    overall_ai_probability: float  # 0-100
    overall_verdict: str
    mean_perplexity: float
    mean_burstiness: float
    flagged_chunk_count: int
    processing_seconds: float


class AnalysisResponse(BaseModel):
    job_id: str
    summary: DocumentSummary
    chunks: List[ChunkResult]


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "extracting", "scoring", "done", "error"
    progress: float  # 0-1
    message: Optional[str] = None
