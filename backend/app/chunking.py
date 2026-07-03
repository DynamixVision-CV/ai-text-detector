"""Split a long document into overlapping-free chunks suitable for scoring.

For a ~100-page document we don't want one giant perplexity pass (too slow,
too coarse) nor per-sentence passes (too noisy, too many model calls).
We chunk by sentence groups targeting ~120-180 words per chunk, which keeps
the local-model context small and gives a stable per-chunk score while still
being fine-grained enough to localize AI-flagged sections in the UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9\"'])")

TARGET_WORDS_PER_CHUNK = 150
MIN_WORDS_PER_CHUNK = 60


@dataclass
class RawChunk:
    text: str
    start_char: int
    end_char: int


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def chunk_document(text: str) -> list[RawChunk]:
    """Group sentences into chunks of roughly TARGET_WORDS_PER_CHUNK words,
    tracking character offsets in the original text so the UI can highlight
    the exact span later.
    """
    chunks: list[RawChunk] = []
    cursor = 0
    buffer_sentences: list[str] = []
    buffer_words = 0
    buffer_start = None

    sentences = split_sentences(text)

    for sentence in sentences:
        # Locate this sentence in the original text starting from cursor
        idx = text.find(sentence, cursor)
        if idx == -1:
            idx = cursor  # fallback, shouldn't normally happen
        end_idx = idx + len(sentence)

        if buffer_start is None:
            buffer_start = idx

        buffer_sentences.append(sentence)
        buffer_words += len(sentence.split())
        cursor = end_idx

        if buffer_words >= TARGET_WORDS_PER_CHUNK:
            chunk_text = " ".join(buffer_sentences)
            chunks.append(RawChunk(text=chunk_text, start_char=buffer_start, end_char=end_idx))
            buffer_sentences = []
            buffer_words = 0
            buffer_start = None

    # Flush remainder; merge into last chunk if too small to be meaningful
    if buffer_sentences:
        chunk_text = " ".join(buffer_sentences)
        if chunks and buffer_words < MIN_WORDS_PER_CHUNK:
            last = chunks.pop()
            merged_text = last.text + " " + chunk_text
            chunks.append(RawChunk(text=merged_text, start_char=last.start_char, end_char=cursor))
        else:
            chunks.append(RawChunk(text=chunk_text, start_char=buffer_start or 0, end_char=cursor))

    return chunks
