"""Unit tests for MaritacaSemanticCache."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_maritaca.cache import MaritacaSemanticCache


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings used only for tests.

    Maps hardcoded strings to fixed vectors; any unseen string falls back to
    a hash-based pseudo-random but deterministic vector of length 4.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.mapping:
            return list(self.mapping[text])
        seed = sum(ord(c) for c in text)
        return [((seed * i) % 97) / 97.0 for i in range(1, 5)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _gen(text: str) -> list[Generation]:
    return [ChatGeneration(message=AIMessage(content=text))]


class TestSemanticCacheMiss:
    def test_lookup_returns_none_when_empty(self) -> None:
        cache = MaritacaSemanticCache(
            embeddings=FakeEmbeddings(), similarity_threshold=0.95, max_entries=10
        )
        result = cache.lookup(prompt="anything", llm_string="{}")
        assert result is None
