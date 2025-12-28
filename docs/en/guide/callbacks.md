# Callbacks for Observability

langchain-maritaca provides callback handlers for monitoring and observability of your LLM interactions. These callbacks help you track costs, measure latency, and monitor streaming performance.

## Available Callbacks

| Callback | Purpose |
|----------|---------|
| `CostTrackingCallback` | Track token usage and estimate API costs |
| `LatencyTrackingCallback` | Measure response times with percentile statistics |
| `TokenStreamingCallback` | Monitor streaming token rates |
| `CombinedCallback` | Combined cost and latency tracking |

## Cost Tracking

Track token usage and estimate costs based on Maritaca AI pricing:

```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback

# Create callback
cost_callback = CostTrackingCallback()

# Use with model
model = ChatMaritaca(callbacks=[cost_callback])

# Make some calls
model.invoke("Hello, how are you?")
model.invoke("Tell me about Brazil")

# Check costs
print(f"Total cost: ${cost_callback.total_cost:.6f}")
print(f"Total tokens: {cost_callback.total_tokens}")
print(f"Input tokens: {cost_callback.total_input_tokens}")
print(f"Output tokens: {cost_callback.total_output_tokens}")
print(f"Number of calls: {cost_callback.call_count}")

# Get detailed summary
summary = cost_callback.get_summary()
print(f"Average cost per call: ${summary['average_cost_per_call']:.6f}")

# Reset for new tracking session
cost_callback.reset()
```

### Pricing Information

The callback uses estimated pricing based on Maritaca AI's public pricing:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| sabia-3.1 | $0.50 | $1.50 |
| sabiazinho-3.1 | $0.10 | $0.30 |

> **Note**: Prices are estimates and may change. Check [Maritaca AI](https://www.maritaca.ai/) for current pricing.

## Latency Tracking

Monitor API response times with statistical analysis:

```python
from langchain_maritaca import ChatMaritaca, LatencyTrackingCallback

# Create callback
latency_callback = LatencyTrackingCallback()

# Use with model
model = ChatMaritaca(callbacks=[latency_callback])

# Make multiple calls
for _ in range(10):
    model.invoke("Quick response please")

# Check latency statistics
print(f"Average latency: {latency_callback.average_latency:.2f}s")
print(f"Min latency: {latency_callback.min_latency:.2f}s")
print(f"Max latency: {latency_callback.max_latency:.2f}s")
print(f"P50 (median): {latency_callback.p50_latency:.2f}s")
print(f"P95 latency: {latency_callback.p95_latency:.2f}s")
print(f"P99 latency: {latency_callback.p99_latency:.2f}s")

# Get all latencies
latencies = latency_callback.latencies
print(f"All latencies: {latencies}")

# Get summary
summary = latency_callback.get_summary()
```

### Error Handling

The latency callback also tracks latency for failed calls:

```python
try:
    model.invoke("This might fail")
except Exception:
    pass

# Latency is still tracked even on error
print(f"Calls tracked: {latency_callback.call_count}")
```

## Token Streaming

Monitor streaming token rates:

```python
from langchain_maritaca import ChatMaritaca, TokenStreamingCallback

# Create callback
streaming_callback = TokenStreamingCallback()

# Use with streaming model
model = ChatMaritaca(streaming=True, callbacks=[streaming_callback])

# Stream a response
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

print()  # New line

# Check streaming stats
print(f"Tokens streamed: {streaming_callback.token_count}")
print(f"Tokens per second: {streaming_callback.tokens_per_second:.1f}")
print(f"Elapsed time: {streaming_callback.elapsed_time:.2f}s")

# Get all tokens
tokens = streaming_callback.tokens
```

## Combined Tracking

Use `CombinedCallback` for both cost and latency tracking:

```python
from langchain_maritaca import ChatMaritaca, CombinedCallback

# Create combined callback
callback = CombinedCallback()

# Use with model
model = ChatMaritaca(callbacks=[callback])

# Make calls
model.invoke("Hello!")
model.invoke("How are you?")

# Access cost tracking
print(f"Total cost: ${callback.cost.total_cost:.6f}")
print(f"Total tokens: {callback.cost.total_tokens}")

# Access latency tracking
print(f"Average latency: {callback.latency.average_latency:.2f}s")
print(f"P95 latency: {callback.latency.p95_latency:.2f}s")

# Get combined summary
summary = callback.get_summary()
print(f"Cost summary: {summary['cost']}")
print(f"Latency summary: {summary['latency']}")

# Reset both
callback.reset()
```

## Using Multiple Callbacks

You can use multiple callbacks simultaneously:

```python
from langchain_maritaca import (
    ChatMaritaca,
    CostTrackingCallback,
    LatencyTrackingCallback,
    TokenStreamingCallback,
)

# Create multiple callbacks
cost_cb = CostTrackingCallback()
latency_cb = LatencyTrackingCallback()
streaming_cb = TokenStreamingCallback()

# Use all callbacks
model = ChatMaritaca(
    streaming=True,
    callbacks=[cost_cb, latency_cb, streaming_cb]
)

# Stream a response
for chunk in model.stream("Tell me about Python"):
    print(chunk.content, end="", flush=True)

print()

# Check all metrics
print(f"Cost: ${cost_cb.total_cost:.6f}")
print(f"Latency: {latency_cb.average_latency:.2f}s")
print(f"Streaming rate: {streaming_cb.tokens_per_second:.1f} tokens/s")
```

## Integration with Monitoring Tools

### Prometheus Example

```python
from prometheus_client import Counter, Histogram, start_http_server
from langchain_maritaca import ChatMaritaca, CombinedCallback

# Create Prometheus metrics
llm_cost_total = Counter('llm_cost_usd_total', 'Total LLM cost in USD')
llm_latency = Histogram('llm_latency_seconds', 'LLM call latency')
llm_tokens_total = Counter('llm_tokens_total', 'Total tokens used', ['type'])

class PrometheusCallback(CombinedCallback):
    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)

        # Update Prometheus metrics
        llm_cost_total.inc(self.cost._costs_by_call[-1]['cost'])
        llm_latency.observe(self.latency.latencies[-1])

        last_call = self.cost._costs_by_call[-1]
        llm_tokens_total.labels(type='input').inc(last_call['input_tokens'])
        llm_tokens_total.labels(type='output').inc(last_call['output_tokens'])

# Start Prometheus server
start_http_server(8000)

# Use custom callback
callback = PrometheusCallback()
model = ChatMaritaca(callbacks=[callback])
```

### Logging Example

```python
import logging
from langchain_maritaca import ChatMaritaca, CombinedCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingCallback(CombinedCallback):
    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)

        last_cost = self.cost._costs_by_call[-1]
        last_latency = self.latency.latencies[-1]

        logger.info(
            f"LLM call completed: "
            f"model={last_cost['model']}, "
            f"tokens={last_cost['total_tokens']}, "
            f"cost=${last_cost['cost']:.6f}, "
            f"latency={last_latency:.2f}s"
        )

callback = LoggingCallback()
model = ChatMaritaca(callbacks=[callback])
```

## Best Practices

1. **Reset callbacks** between independent tracking sessions
2. **Use CombinedCallback** when you need both cost and latency metrics
3. **Create custom callbacks** by extending the base classes for integration with your monitoring stack
4. **Monitor in production** to track costs and detect performance regressions
5. **Set up alerts** for unusual cost spikes or latency increases
