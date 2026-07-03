"""Extract raw text from uploaded documents.

Supports .pdf, .docx and .txt. Designed to scale to ~100-page documents by
streaming pages/paragraphs rather than loading everything into memory twice.
"""
from __future__ import annotations

import io
from pathlib import Path

import pdfplumber
from docx import Document


class UnsupportedFileType(Exception):
    pass


def extract_text(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(raw_bytes)
    if suffix == ".docx":
        return _extract_docx(raw_bytes)
    if suffix == ".txt":
        return raw_bytes.decode("utf-8", errors="ignore")

    raise UnsupportedFileType(f"Format non supporté : {suffix or 'inconnu'}")


def _extract_pdf(raw_bytes: bytes) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _extract_docx(raw_bytes: bytes) -> str:
    doc = Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)
