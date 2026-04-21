"""Semantic cache for langchain-maritaca.

Plugs into LangChain's ``set_llm_cache()`` mechanism via the standard
``BaseCache`` interface. Matches new prompts against previously cached
prompts by cosine similarity over embeddings, scoped per ``llm_string`` so
entries never leak across model or config boundaries.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache  # noqa: F401
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class MaritacaSemanticCache(BaseCache):
    """Semantic cache keyed by prompt embedding similarity.

    Args:
        embeddings: Any ``Embeddings`` implementation. Reuse
            ``DeepInfraEmbeddings`` from this package for the recommended
            Portuguese-capable model.
        similarity_threshold: Cosine similarity in ``[0, 1]`` above which a
            cached entry is considered a hit. Defaults to ``0.95`` (strict).
        max_entries: LRU upper bound per ``llm_string`` scope. Defaults to
            ``1000``.
        fail_silently: If ``True`` (default), embedding failures degrade to
            cache misses without raising, so the caller falls through to the
            model. If ``False``, re-raise.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
        fail_silently: bool = True,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            msg = (
                f"similarity_threshold must be in [0, 1], got {similarity_threshold!r}"
            )
            raise ValueError(msg)
        if max_entries < 1:
            msg = f"max_entries must be >= 1, got {max_entries!r}"
            raise ValueError(msg)
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.fail_silently = fail_silently
        # Per-scope storage: llm_string -> OrderedDict[prompt -> (vec, generations)]
        self._store: dict[
            str, OrderedDict[str, tuple[np.ndarray, RETURN_VAL_TYPE]]
        ] = {}
        self._lock = threading.Lock()

    def _embed(self, prompt: str) -> np.ndarray | None:
        try:
            vec = np.asarray(self.embeddings.embed_query(prompt), dtype=np.float32)
        except Exception:
            if self.fail_silently:
                logger.warning(
                    "MaritacaSemanticCache embedding failed; treating as miss.",
                    exc_info=True,
                )
                return None
            raise
        return vec

    def _cosine(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # matrix: shape (n, d); query: shape (d,)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero for all-zero rows
        safe = matrix_norms > 0
        scores = np.zeros(matrix.shape[0], dtype=np.float32)
        scores[safe] = (matrix[safe] @ query) / (matrix_norms[safe] * query_norm)
        return scores

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        with self._lock:
            bucket = self._store.get(llm_string)
            if not bucket:
                return None

            query_vec = self._embed(prompt)
            if query_vec is None:
                return None

            prompts = list(bucket.keys())
            matrix = np.stack([bucket[p][0] for p in prompts])
            scores = self._cosine(query_vec, matrix)
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) < self.similarity_threshold:
                return None

            best_prompt = prompts[best_idx]
            entry = bucket.pop(best_prompt)
            bucket[best_prompt] = entry  # LRU bump
            return entry[1]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        with self._lock:
            query_vec = self._embed(prompt)
            if query_vec is None:
                return

            bucket = self._store.setdefault(llm_string, OrderedDict())
            if prompt in bucket:
                bucket.pop(prompt)
            bucket[prompt] = (query_vec, return_val)
            # LRU eviction; Task 8 verifies the boundary
            while len(bucket) > self.max_entries:
                bucket.popitem(last=False)

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._store.clear()

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        # Delegate to sync. The embedding call and numpy work are CPU/IO-bound
        # but already fast; the lock keeps thread safety. Subclasses can
        # override if they need to back off to aembed_query.
        return self.lookup(prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        self.update(prompt, llm_string, return_val)
