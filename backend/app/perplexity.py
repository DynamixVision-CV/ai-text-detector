"""Perplexity scoring via a small local causal language model.

We use GPT-2 (or distilgpt2) purely as a *scoring* model: we never generate
text with it, we only compute how "surprised" it is by an already-written
passage. Low perplexity => the text closely matches patterns an LM would
itself produce => more likely AI-generated. High perplexity => more
"surprising" word choices => more likely human-written.

The model is loaded lazily and cached as a singleton so repeated requests
don't reload weights. If loading fails (no internet on first run, no torch,
etc.) `compute_perplexity` returns None and the app falls back to the
non-ML heuristics only (see heuristics.py) - the app never crashes because
of this.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

MODEL_NAME = os.environ.get("PERPLEXITY_MODEL", "distilgpt2")
MAX_TOKENS = 512  # keep each chunk within a single forward pass


class _ModelUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # torch/transformers not installed
        raise _ModelUnavailable(str(exc)) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device, torch
    except Exception as exc:  # network unavailable, disk issue, etc.
        raise _ModelUnavailable(str(exc)) from exc


def perplexity_model_ready() -> bool:
    try:
        _load_model()
        return True
    except _ModelUnavailable:
        return False


def compute_perplexity(text: str) -> Optional[float]:
    """Returns perplexity as a float, or None if the model is unavailable."""
    try:
        tokenizer, model, device, torch = _load_model()
    except _ModelUnavailable:
        return None

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    )
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item()

    return float(math.exp(neg_log_likelihood))
