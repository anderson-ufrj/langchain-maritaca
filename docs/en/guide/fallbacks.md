# Smart Fallback Chains

`ChatMaritaca` ships a pre-configured fallback helper that wraps the underlying LangChain `Runnable.with_fallbacks()` machinery with a curated model order and a transient-only error filter.

## Basic use

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(model="sabia-3.1", temperature=0.3).with_smart_fallbacks()
response = model.invoke("Explique o artigo 5 da Constituição brasileira.")
```

The call above tries `sabia-3.1` first and falls through to `sabiazinho-4` then `sabiazinho-3.1` only on transient failures (`429`, `502`, `503`, `504`, or timeouts). Authentication errors, bad requests, and other 4xx statuses propagate immediately so you notice configuration bugs.

## Building from scratch

```python
from langchain_maritaca import ChatMaritaca

chain = ChatMaritaca.with_smart_fallbacks_from_primary(
    primary="sabia-3.1",
    api_key="...",
    temperature=0.3,
)
```

All keyword arguments are passed to the primary `ChatMaritaca` and inherited by the fallback siblings.

## Customizing the chain

```python
model = ChatMaritaca(model="sabia-3.1").with_smart_fallbacks(
    fallbacks=["sabiazinho-3.1"],
)
```

You can also override the exception filter if you need broader retry behavior:

```python
import httpx

model = ChatMaritaca(model="sabia-3.1").with_smart_fallbacks(
    exceptions_to_handle=(httpx.TimeoutException, httpx.HTTPStatusError),
)
```

## Default chain

| Primary | Fallback order |
|---|---|
| `sabia-3.1` | `sabiazinho-4` → `sabiazinho-3.1` |
| `sabiazinho-4` | `sabia-3.1` → `sabiazinho-3.1` |
| `sabiazinho-3.1` | `sabia-3.1` → `sabiazinho-4` |

Unknown primary models (for example, future Sabiá variants not yet known to the package) require an explicit `fallbacks=[...]` list, otherwise the helper raises a `ValueError` at construction time.

## When not to use it

- If you only need retries on rate limits against the *same* model, use `retry_if_rate_limited` instead — a fallback would switch models unnecessarily.
- If the downstream system requires strict model pinning (for evaluation reproducibility, for instance), skip fallbacks so behavior is deterministic.
