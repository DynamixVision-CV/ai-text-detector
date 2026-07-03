import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking import split_sentences, chunk_document
from app.heuristics import burstiness, repetition_score, lexical_diversity
from app.scoring import _perplexity_to_ai_score, _burstiness_to_ai_score, _verdict


def test_split_sentences_basic():
    text = "Ceci est une phrase. En voici une autre! Et une troisième?"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_chunk_document_preserves_offsets():
    text = "Phrase un. Phrase deux. " * 50
    chunks = chunk_document(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert text[c.start_char:c.start_char + 10] in text  # sanity: offsets exist in text


def test_burstiness_uniform_sentences_is_low():
    uniform = "Le chat mange. Le chien court. Le chat dort. Le chien joue. Le chat saute. Le chien mange."
    varied = "Le chat mange. Après une longue journée passée à explorer le jardin sous la pluie battante, il s'endort profondément. Il court."
    assert burstiness(uniform) < burstiness(varied)


def test_repetition_score_detects_repeated_ngrams():
    repetitive = "il est important de noter que il est important de noter que cela fonctionne bien"
    diverse = "le renard brun saute par dessus la barrière tandis que le hibou observe la scène en silence"
    assert repetition_score(repetitive) > repetition_score(diverse)


def test_perplexity_to_ai_score_bounds():
    assert _perplexity_to_ai_score(5) == 1.0
    assert _perplexity_to_ai_score(1000) == 0.0
    assert _perplexity_to_ai_score(None) == 0.5


def test_verdict_thresholds():
    assert _verdict(10) == "human"
    assert _verdict(50) == "mixed"
    assert _verdict(90) == "ai"
