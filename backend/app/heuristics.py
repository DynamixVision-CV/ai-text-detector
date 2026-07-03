"""Non-ML statistical heuristics used alongside perplexity.

These are cheap, deterministic, and run even if the local language model
fails to load - so the app degrades gracefully rather than crashing.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from .chunking import split_sentences

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def burstiness(text: str) -> float:
    """Coefficient of variation of sentence length (in words).

    Human writing tends to alternate short and long sentences (high
    burstiness). AI-generated text is often more uniform (low burstiness).
    Returns a value roughly in [0, 2]; higher = more human-like variation.
    """
    sentences = split_sentences(text)
    lengths = [len(_WORD_RE.findall(s)) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 1.0  # not enough data, neutral value

    mean_len = statistics.mean(lengths)
    if mean_len == 0:
        return 1.0
    stdev_len = statistics.pstdev(lengths)
    return stdev_len / mean_len


def repetition_score(text: str) -> float:
    """Measures n-gram repetition / low lexical diversity, a common tell of
    templated AI phrasing. Returns 0 (diverse) to 1 (highly repetitive).
    """
    words = _words(text)
    if len(words) < 8:
        return 0.0

    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return min(1.0, repeated / max(1, len(trigrams)))


def lexical_diversity(text: str) -> float:
    """Type-token ratio: unique words / total words. Lower can indicate
    more formulaic (AI-like) phrasing, though this is a weak signal alone.
    """
    words = _words(text)
    if not words:
        return 1.0
    return len(set(words)) / len(words)
