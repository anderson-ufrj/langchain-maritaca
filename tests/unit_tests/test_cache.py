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


class TestSemanticCacheLookupAndUpdate:
    def test_update_then_lookup_exact_prompt_hits(self) -> None:
        emb = FakeEmbeddings(mapping={"hello": [1.0, 0.0, 0.0, 0.0]})
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=10
        )
        cache.update("hello", "{}", _gen("world"))
        result = cache.lookup("hello", "{}")
        assert result is not None
        assert result[0].message.content == "world"  # type: ignore[union-attr]

    def test_lookup_hits_when_similarity_above_threshold(self) -> None:
        emb = FakeEmbeddings(
            mapping={
                "Qual a capital do Brasil?": [1.0, 0.0, 0.0, 0.0],
                "Qual é a capital do Brasil?": [0.99, 0.01, 0.0, 0.0],
            }
        )
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=10
        )
        cache.update("Qual a capital do Brasil?", "{}", _gen("Brasília"))
        result = cache.lookup("Qual é a capital do Brasil?", "{}")
        assert result is not None
        assert result[0].message.content == "Brasília"  # type: ignore[union-attr]

    def test_lookup_misses_when_similarity_below_threshold(self) -> None:
        emb = FakeEmbeddings(
            mapping={
                "hello": [1.0, 0.0, 0.0, 0.0],
                "xyzzy": [0.0, 1.0, 0.0, 0.0],  # orthogonal -> cosine = 0
            }
        )
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=10
        )
        cache.update("hello", "{}", _gen("world"))
        assert cache.lookup("xyzzy", "{}") is None

    def test_different_llm_string_is_a_different_scope(self) -> None:
        emb = FakeEmbeddings(mapping={"hello": [1.0, 0.0, 0.0, 0.0]})
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=10
        )
        cache.update("hello", "scope-a", _gen("response-a"))
        assert cache.lookup("hello", "scope-a") is not None
        assert cache.lookup("hello", "scope-b") is None

    def test_clear_removes_all_entries(self) -> None:
        emb = FakeEmbeddings(mapping={"hello": [1.0, 0.0, 0.0, 0.0]})
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=10
        )
        cache.update("hello", "{}", _gen("world"))
        cache.clear()
        assert cache.lookup("hello", "{}") is None


class TestSemanticCacheLRU:
    def test_oldest_entry_is_evicted_when_max_entries_exceeded(self) -> None:
        emb = FakeEmbeddings(
            mapping={
                "one": [1.0, 0.0, 0.0, 0.0],
                "two": [0.0, 1.0, 0.0, 0.0],
                "three": [0.0, 0.0, 1.0, 0.0],
            }
        )
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=2
        )
        cache.update("one", "{}", _gen("g-one"))
        cache.update("two", "{}", _gen("g-two"))
        cache.update("three", "{}", _gen("g-three"))  # evicts "one"

        assert cache.lookup("one", "{}") is None
        result_two = cache.lookup("two", "{}")
        result_three = cache.lookup("three", "{}")
        assert result_two is not None
        assert result_two[0].message.content == "g-two"  # type: ignore[union-attr]
        assert result_three is not None
        assert result_three[0].message.content == "g-three"  # type: ignore[union-attr]

    def test_hit_bumps_entry_and_protects_it_from_eviction(self) -> None:
        emb = FakeEmbeddings(
            mapping={
                "one": [1.0, 0.0, 0.0, 0.0],
                "two": [0.0, 1.0, 0.0, 0.0],
                "three": [0.0, 0.0, 1.0, 0.0],
            }
        )
        cache = MaritacaSemanticCache(
            embeddings=emb, similarity_threshold=0.95, max_entries=2
        )
        cache.update("one", "{}", _gen("g-one"))
        cache.update("two", "{}", _gen("g-two"))
        # Touch "one" so it moves to the LRU head
        assert cache.lookup("one", "{}") is not None
        cache.update("three", "{}", _gen("g-three"))  # should evict "two"

        assert cache.lookup("two", "{}") is None
        assert cache.lookup("one", "{}") is not None
        assert cache.lookup("three", "{}") is not None
