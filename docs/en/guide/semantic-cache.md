# Semantic Cache

`MaritacaSemanticCache` is a drop-in replacement for LangChain's exact-match caches. It matches prompts by cosine similarity over embeddings, so paraphrases of previously answered questions reuse the cached response.

## Quickstart

```python
from langchain_core.globals import set_llm_cache
from langchain_maritaca import (
    ChatMaritaca,
    DeepInfraEmbeddings,
    MaritacaSemanticCache,
)

cache = MaritacaSemanticCache(
    embeddings=DeepInfraEmbeddings(),
    similarity_threshold=0.95,
    max_entries=1000,
)
set_llm_cache(cache)

model = ChatMaritaca()
model.invoke("Qual a capital do Brasil?")    # miss -> API
model.invoke("Qual é a capital do Brasil?")  # hit (cosine >= 0.95)
```

## Scope isolation

Cache entries are bucketed by `llm_string`, the serialized model config LangChain already passes to `BaseCache`. Two calls with different `temperature`, `model`, or tool bindings will never share entries — the cache protects you from accidental cross-config hits.

## Tuning the threshold

- `0.95` (default) is strict — only close paraphrases hit.
- `0.90` is a common middle ground for FAQ workloads.
- Values below `0.85` risk surprising users with "similar but wrong" answers.

Measure precision on your own dataset before lowering the threshold.

## Failure modes

`fail_silently=True` (default) turns embedding outages into cache misses. The model call proceeds and your pipeline keeps running. Set `fail_silently=False` if caching is load-bearing (batch cost experiments, eval replays) and you prefer loud failures.

```python
cache = MaritacaSemanticCache(
    embeddings=DeepInfraEmbeddings(),
    similarity_threshold=0.92,
    fail_silently=False,  # raise on embedding errors
)
```

## LRU eviction

The `max_entries` bound applies per `llm_string` scope. When the bound is exceeded, the least-recently-accessed entry is evicted. Cache hits count as access, so frequently used entries stay alive.

## Async

`alookup` and `aupdate` are supported and delegate to the sync versions, which keeps the numpy-backed similarity search in-thread (it is already fast for typical bucket sizes).

## What this cache is not

- **Not persistent** — entries live in memory and disappear on process restart.
- **Not for multi-process sharing** — for distributed cache, combine with an external vector store in a future release.
