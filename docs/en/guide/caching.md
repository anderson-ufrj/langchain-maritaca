# Caching

ChatMaritaca supports LangChain's native caching mechanism, which can significantly reduce API costs and improve response times for repeated queries.

## Why Use Caching?

- **Cost Reduction**: Avoid paying for duplicate API calls
- **Faster Responses**: Cached responses are returned instantly
- **Rate Limit Protection**: Fewer API calls means less chance of hitting rate limits

## Basic Usage

### In-Memory Cache

The simplest way to enable caching is with `InMemoryCache`:

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

# Enable caching globally
set_llm_cache(InMemoryCache())

model = ChatMaritaca()

# First call - hits the API
response1 = model.invoke("What is the capital of Brazil?")
print(response1.content)  # "The capital of Brazil is Brasília."

# Second call with same input - uses cache (instant!)
response2 = model.invoke("What is the capital of Brazil?")
print(response2.content)  # Same response, no API call
```

### SQLite Cache

For persistent caching across sessions:

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

# Cache persists to disk
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

### Redis Cache

For production environments with distributed caching:

```python
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache
import redis

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379)
set_llm_cache(RedisCache(redis_client))
```

## How Cache Keys Work

Cache keys are generated based on:

1. **The prompt/messages**: Exact text of the input
2. **Model parameters**: temperature, max_tokens, top_p, etc.
3. **Model name**: sabia-3.1, sabiazinho-3.1, etc.

This means:

```python
model1 = ChatMaritaca(temperature=0.5)
model2 = ChatMaritaca(temperature=0.9)

# These will have DIFFERENT cache keys (different temperature)
model1.invoke("Hello")  # Cache miss
model2.invoke("Hello")  # Cache miss (different key)

# This will be a cache HIT (same model, same input)
model1.invoke("Hello")  # Uses cached response
```

## Disabling Cache

To disable caching:

```python
from langchain_core.globals import set_llm_cache

# Disable global cache
set_llm_cache(None)
```

## Best Practices

1. **Use caching in development**: Speeds up iteration and reduces costs
2. **Consider cache invalidation**: Cached responses don't reflect model updates
3. **Use persistent cache for production**: SQLite or Redis for durability
4. **Monitor cache hit rate**: Track how effective your caching strategy is

## Example: Cost Tracking with Cache

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

set_llm_cache(InMemoryCache())
model = ChatMaritaca()

api_calls = 0

def tracked_invoke(prompt: str) -> str:
    global api_calls
    # Check if this would be a cache hit
    # (simplified - actual implementation would check cache directly)
    response = model.invoke(prompt)
    if response.response_metadata.get("from_cache"):
        print(f"Cache HIT: {prompt[:30]}...")
    else:
        api_calls += 1
        print(f"Cache MISS: {prompt[:30]}... (API call #{api_calls})")
    return response.content

# First call - API
tracked_invoke("What is Python?")

# Second call - Cache
tracked_invoke("What is Python?")

print(f"Total API calls: {api_calls}")  # 1
```

## Limitations

- **Streaming**: Cache works with non-streaming calls only
- **Tool Calling**: Tool call responses are cached based on the full conversation
- **Memory**: In-memory cache is lost when the process ends

## See Also

- [LangChain Caching Documentation](https://python.langchain.com/docs/how_to/llm_caching/)
- [Basic Usage](basic-usage.md)
- [Configuration](../getting-started/configuration.md)
