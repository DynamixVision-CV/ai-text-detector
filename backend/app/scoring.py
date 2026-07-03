"""Combine perplexity + statistical heuristics into a single AI-probability
score per chunk, and aggregate into a document-level summary.

IMPORTANT — read before trusting the output:
This score is a *heuristic estimate*, not a forensic proof. It should never
be the sole basis for an academic integrity decision. See README.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .chunking import chunk_document
from .heuristics import burstiness, repetition_score, lexical_diversity
from .perplexity import compute_perplexity
from .models import ChunkResult, DocumentSummary, AnalysisResponse

# Calibration constants (heuristic, tuned by hand — not a scientific ground truth)
PPL_LOW = 12.0   # perplexity at/below this -> strongly AI-like
PPL_HIGH = 55.0  # perplexity at/above this -> strongly human-like
BURSTINESS_LOW = 0.35   # low variation -> AI-like
BURSTINESS_HIGH = 0.85  # high variation -> human-like

WEIGHT_PERPLEXITY = 0.55
WEIGHT_BURSTINESS = 0.25
WEIGHT_REPETITION = 0.20


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _perplexity_to_ai_score(ppl: float | None) -> float:
    """Map perplexity to an AI-likelihood in [0, 1]. Lower perplexity -> higher score."""
    if ppl is None:
        return 0.5  # unknown, neutral
    if ppl <= PPL_LOW:
        return 1.0
    if ppl >= PPL_HIGH:
        return 0.0
    # linear interpolation between the two anchors
    return _clip01(1.0 - (ppl - PPL_LOW) / (PPL_HIGH - PPL_LOW))


def _burstiness_to_ai_score(b: float) -> float:
    if b <= BURSTINESS_LOW:
        return 1.0
    if b >= BURSTINESS_HIGH:
        return 0.0
    return _clip01(1.0 - (b - BURSTINESS_LOW) / (BURSTINESS_HIGH - BURSTINESS_LOW))


def _verdict(ai_probability: float) -> str:
    if ai_probability >= 70:
        return "ai"
    if ai_probability >= 40:
        return "mixed"
    return "human"


def score_chunk(index: int, raw_text: str, start_char: int, end_char: int) -> ChunkResult:
    ppl = compute_perplexity(raw_text)
    b = burstiness(raw_text)
    rep = repetition_score(raw_text)

    ppl_score = _perplexity_to_ai_score(ppl)
    burst_score = _burstiness_to_ai_score(b)
    rep_score = _clip01(rep)

    combined = (
        WEIGHT_PERPLEXITY * ppl_score
        + WEIGHT_BURSTINESS * burst_score
        + WEIGHT_REPETITION * rep_score
    )
    ai_probability = round(combined * 100, 1)

    return ChunkResult(
        index=index,
        text=raw_text,
        start_char=start_char,
        end_char=end_char,
        perplexity=round(ppl, 2) if ppl is not None else -1.0,
        burstiness=round(b, 3),
        repetition_score=round(rep, 3),
        ai_probability=ai_probability,
        verdict=_verdict(ai_probability),
    )


def analyze_document(job_id: str, filename: str, text: str) -> AnalysisResponse:
    started = time.time()

    raw_chunks = chunk_document(text)
    chunk_results = [
        score_chunk(i, c.text, c.start_char, c.end_char)
        for i, c in enumerate(raw_chunks)
    ]

    if chunk_results:
        mean_ppl = sum(c.perplexity for c in chunk_results if c.perplexity >= 0) / max(
            1, len([c for c in chunk_results if c.perplexity >= 0])
        )
        mean_burst = sum(c.burstiness for c in chunk_results) / len(chunk_results)
        overall_ai = sum(c.ai_probability for c in chunk_results) / len(chunk_results)
        flagged = len([c for c in chunk_results if c.verdict == "ai"])
    else:
        mean_ppl, mean_burst, overall_ai, flagged = 0.0, 0.0, 0.0, 0

    summary = DocumentSummary(
        filename=filename,
        total_chars=len(text),
        total_chunks=len(chunk_results),
        word_count=len(text.split()),
        overall_ai_probability=round(overall_ai, 1),
        overall_verdict=_verdict(overall_ai),
        mean_perplexity=round(mean_ppl, 2),
        mean_burstiness=round(mean_burst, 3),
        flagged_chunk_count=flagged,
        processing_seconds=round(time.time() - started, 2),
    )

    return AnalysisResponse(job_id=job_id, summary=summary, chunks=chunk_results)
