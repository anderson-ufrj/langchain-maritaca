# langchain-maritaca Roadmap

Consolidated roadmap with detailed implementation steps for each planned feature.

> **Last Updated:** December 2025
> **Current Version:** v0.2.4

---

## Summary

| Feature | Priority | Complexity | Status |
|---------|----------|------------|--------|
| ~~Cache Integration~~ | High | Low | **IMPLEMENTED** |
| ~~Configurable Retry Logic~~ | High | Low | **IMPLEMENTED** |
| ~~Enhanced Callbacks~~ | Medium | Low | **IMPLEMENTED** |
| Token Counter | Medium | Medium | Planned |
| Batch Optimization | Low | Medium | Planned |
| Multimodal/Vision Support | Low | High | Blocked (API) |

---

## HIGH PRIORITY

### 1. ~~Cache Integration~~ ✅ IMPLEMENTED

**Goal:** Enable LangChain's native caching to reduce API costs for repeated queries.

**Result:** LangChain's caching works natively with ChatMaritaca! No code changes needed.

**Usage:**
```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

set_llm_cache(InMemoryCache())
model = ChatMaritaca()

# First call hits API, second call uses cache
response1 = model.invoke("Hello")  # API call
response2 = model.invoke("Hello")  # Cache hit (instant!)
```

**Supported cache backends:**
- `InMemoryCache` - Simple in-memory caching
- `SQLiteCache` - Persistent file-based caching
- `RedisCache` - Distributed caching for production

**Status:** ✅ IMPLEMENTED
**Complexity:** Low (native LangChain support)
**Impact:** Medium - Reduces costs for repeated queries

---

### 2. ~~Configurable Retry Logic~~ ✅ IMPLEMENTED

**Goal:** Expose retry parameters for production resilience customization.

**Implemented Parameters:**
```python
model = ChatMaritaca(
    retry_if_rate_limited=True,   # Auto-retry on HTTP 429
    retry_delay=1.0,              # Initial delay (seconds)
    retry_max_delay=60.0,         # Maximum delay (seconds)
    retry_multiplier=2.0,         # Exponential backoff multiplier
    max_retries=2,                # Maximum retry attempts
)
```

**Features:**
- Exponential backoff with configurable multiplier
- Delay capped at `retry_max_delay`
- Rate limit retry can be disabled with `retry_if_rate_limited=False`
- Validation for all retry parameters
- Works for both sync and async requests

**Status:** ✅ IMPLEMENTED
**Complexity:** Low
**Impact:** Medium - Better resilience in production

---

## MEDIUM PRIORITY

### 3. ~~Enhanced Callbacks~~ ✅ IMPLEMENTED

**Goal:** Provide granular callbacks for observability (cost tracking, latency, token streaming).

**Implemented Callbacks:**
- `CostTrackingCallback` - Track token usage and estimate API costs
- `LatencyTrackingCallback` - Measure response times with percentile statistics
- `TokenStreamingCallback` - Monitor streaming token rates
- `CombinedCallback` - Combined cost and latency tracking

**Usage:**
```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback, LatencyTrackingCallback

cost_cb = CostTrackingCallback()
latency_cb = LatencyTrackingCallback()

model = ChatMaritaca(callbacks=[cost_cb, latency_cb])
model.invoke("Hello!")

print(f"Cost: ${cost_cb.total_cost:.6f}")
print(f"Tokens: {cost_cb.total_tokens}")
print(f"Latency: {latency_cb.average_latency:.2f}s")
print(f"P95: {latency_cb.p95_latency:.2f}s")
```

**Features:**
- Cost tracking based on Maritaca AI pricing (sabia-3.1, sabiazinho-3.1)
- Latency statistics with P50, P95, P99 percentiles
- Token streaming rate calculation
- Easy integration with monitoring tools (Prometheus, logging)

**Status:** ✅ IMPLEMENTED
**Complexity:** Low
**Impact:** Medium - Better observability

---

### 4. Token Counter

**Goal:** Implement `get_num_tokens()` for cost estimation before API calls.

**Implementation Steps:**

1. **Research tokenization options**
   - Check if Maritaca publishes their tokenizer
   - Evaluate `tiktoken` as approximation
   - Consider `sentencepiece` for Portuguese

2. **Choose tokenization strategy**
   - Option A: Use tiktoken cl100k_base as approximation
   - Option B: Implement character-based estimation
   - Option C: Call Maritaca API for exact count (if available)

3. **Implement `get_num_tokens()` method**
   - File: `langchain_maritaca/chat_models.py`
   - Handle different input types (str, list of messages)
   - Return token count estimate

4. **Implement `get_num_tokens_from_messages()`**
   - Handle message formatting overhead
   - Account for system/user/assistant prefixes

5. **Add cost estimation utility**
   ```python
   def estimate_cost(self, messages: list, max_output_tokens: int = 1000) -> float:
       input_tokens = self.get_num_tokens_from_messages(messages)
       # Calculate based on Maritaca pricing
   ```

6. **Add tests**
   - Test token counting accuracy
   - Test cost estimation
   - Compare with actual API usage

7. **Add documentation**
   - Usage examples
   - Accuracy disclaimers

**Files to modify:**
- `langchain_maritaca/chat_models.py`
- `pyproject.toml` (add tiktoken dependency, optional)
- `tests/unit_tests/test_token_counter.py` (new)
- `docs/en/guide/cost-estimation.md` (new)

---

## LOW PRIORITY

### 5. Batch Optimization

**Goal:** Use batch API endpoint for improved throughput on multiple requests.

**Implementation Steps:**

1. **Research Maritaca batch API**
   - Check if Maritaca offers batch endpoints
   - Document API differences from single request

2. **If batch API exists:**
   - Implement `_generate_batch` method
   - Override `batch()` method to use batch API
   - Handle batch size limits

3. **If no batch API:**
   - Implement client-side batching with async
   - Use `asyncio.gather` for parallel requests
   - Add rate limiting to avoid API throttling

4. **Add benchmarks**
   - Compare batch vs sequential performance
   - Measure throughput improvements

5. **Add documentation**
   - Batch usage examples
   - Performance tuning guide

**Files to modify:**
- `langchain_maritaca/chat_models.py`
- `tests/unit_tests/test_batch.py` (new)
- `tests/benchmarks/test_batch_performance.py` (new)

---

### 6. Multimodal/Vision Support

**Goal:** Support image inputs when Maritaca API adds vision capabilities.

**Status:** BLOCKED - Waiting for Maritaca API to support image inputs.

**Preparation Steps:**

1. **Monitor Maritaca API updates**
   - Check API documentation periodically
   - Subscribe to Maritaca announcements

2. **Research LangChain multimodal patterns**
   - Study how other integrations handle images
   - Understand `ImageBlock` and `HumanMessageChunk`

3. **When API available:**
   - Implement image content handling
   - Add base64 encoding utilities
   - Support URL and local file inputs

4. **Testing considerations:**
   - Mock image API for unit tests
   - Create integration tests with real images

**Files to modify (when ready):**
- `langchain_maritaca/chat_models.py`
- `tests/unit_tests/test_vision.py` (new)
- `tests/integration_tests/test_vision.py` (new)
- `docs/en/guide/vision.md` (new)

---

## Completed Features

| Feature | Version | Date |
|---------|---------|------|
| Embeddings Support (DeepInfra) | v0.2.2 | Dec 2025 |
| Structured Output | v0.2.2 | Dec 2025 |
| Tool Calling / Function Calling | v0.2.0 | Dec 2025 |
| Coverage Badge | v0.2.3 | Dec 2025 |
| Bilingual Documentation | v0.2.3 | Dec 2025 |
| Configurable Retry Logic | v0.2.3 | Dec 2025 |
| Cache Integration | v0.2.4 | Dec 2025 |
| Enhanced Callbacks | v0.2.4 | Dec 2025 |

---

## Contributing

Want to work on one of these features? Here's how:

1. **Pick a feature** from the roadmap above
2. **Open an issue** to discuss your approach
3. **Fork the repository** and create a feature branch
4. **Follow the implementation steps** outlined above
5. **Submit a pull request** with tests and documentation

For questions, open an issue on GitHub or check the [Contributing Guide](../../CONTRIBUTING.md).
