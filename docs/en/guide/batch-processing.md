# Batch Processing

This guide covers how to efficiently process multiple requests using ChatMaritaca's batch optimization features.

## Overview

When you need to process many requests, batch processing can significantly improve throughput by running multiple requests concurrently while respecting API rate limits.

## Async Batch Processing

Use `abatch_with_concurrency()` for efficient parallel processing:

```python
import asyncio
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca()

# Multiple questions to process
inputs = [
    [HumanMessage(content="What is the capital of Brazil?")],
    [HumanMessage(content="What is the capital of Argentina?")],
    [HumanMessage(content="What is the capital of Chile?")],
    [HumanMessage(content="What is the capital of Peru?")],
    [HumanMessage(content="What is the capital of Colombia?")],
]

# Process with max 3 concurrent requests
results = asyncio.run(
    model.abatch_with_concurrency(inputs, max_concurrency=3)
)

for result in results:
    print(result.generations[0].message.content)
```

### Concurrency Control

The `max_concurrency` parameter controls how many requests run simultaneously:

```python
# Conservative: 2 concurrent requests
results = await model.abatch_with_concurrency(inputs, max_concurrency=2)

# Aggressive: 10 concurrent requests (may hit rate limits)
results = await model.abatch_with_concurrency(inputs, max_concurrency=10)
```

Recommended values:
- **2-3**: Safe for most use cases
- **5**: Good balance of speed and reliability
- **10+**: Only if you have high rate limits

### Error Handling

Use `return_exceptions=True` to handle errors gracefully:

```python
results = await model.abatch_with_concurrency(
    inputs,
    max_concurrency=5,
    return_exceptions=True,  # Don't fail on individual errors
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Request {i} failed: {result}")
    else:
        print(f"Request {i}: {result.generations[0].message.content}")
```

## Batch with Progress Tracking

Use `batch_with_progress()` for synchronous processing with progress updates:

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca()

def on_progress(completed, total, result):
    print(f"Progress: {completed}/{total} ({100*completed/total:.0f}%)")

inputs = [
    [HumanMessage(content=f"Question {i}")] for i in range(20)
]

results = model.batch_with_progress(
    inputs,
    max_concurrency=3,
    callback=on_progress,
)
```

### Progress Callback

The callback receives:
- `completed`: Number of completed requests
- `total`: Total number of requests
- `result`: The ChatResult for the just-completed request

```python
def detailed_progress(completed, total, result):
    content = result.generations[0].message.content[:50]
    print(f"[{completed}/{total}] Received: {content}...")
```

## Cost Estimation

Estimate batch costs before execution with `abatch_estimate_cost()`:

```python
inputs = [
    [HumanMessage(content="Short question")],
    [HumanMessage(content="Another question")],
    [HumanMessage(content="Third question")],
]

estimate = await model.abatch_estimate_cost(
    inputs,
    max_output_tokens=500,  # Expected output per request
)

print(f"Total requests: {estimate['total_requests']}")
print(f"Estimated input tokens: {estimate['total_input_tokens']}")
print(f"Estimated output tokens: {estimate['total_output_tokens']}")
print(f"Estimated cost: ${estimate['total_cost']:.6f}")
```

### Cost Planning

```python
# Check cost before processing large batch
if estimate['total_cost'] > 1.00:  # Budget limit
    print("Warning: Batch cost exceeds $1.00")
    proceed = input("Continue? (y/n): ")
    if proceed.lower() != 'y':
        exit()

results = await model.abatch_with_concurrency(inputs)
```

## Best Practices

### 1. Choose Appropriate Concurrency

```python
# For rate-limited APIs
results = await model.abatch_with_concurrency(inputs, max_concurrency=2)

# For high-throughput APIs
results = await model.abatch_with_concurrency(inputs, max_concurrency=10)
```

### 2. Handle Failures Gracefully

```python
async def safe_batch(model, inputs):
    results = await model.abatch_with_concurrency(
        inputs,
        return_exceptions=True,
    )

    successful = []
    failed = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed.append((i, inputs[i], result))
        else:
            successful.append(result)

    # Retry failed requests
    if failed:
        retry_inputs = [item[1] for item in failed]
        retry_results = await model.abatch_with_concurrency(
            retry_inputs,
            max_concurrency=1,  # Slower, more reliable
        )
        successful.extend(retry_results)

    return successful
```

### 3. Combine with Cost Estimation

```python
async def batch_with_budget(model, inputs, max_cost=1.0):
    estimate = await model.abatch_estimate_cost(inputs)

    if estimate['total_cost'] > max_cost:
        raise ValueError(f"Batch cost ${estimate['total_cost']:.4f} exceeds budget ${max_cost}")

    return await model.abatch_with_concurrency(inputs)
```

## API Reference

### `abatch_with_concurrency(inputs, max_concurrency=5, stop=None, return_exceptions=False)`

Process multiple message lists concurrently.

**Parameters:**
- `inputs`: List of message lists to process
- `max_concurrency`: Maximum concurrent requests (default: 5)
- `stop`: Optional stop sequences
- `return_exceptions`: If True, return exceptions instead of raising

**Returns:** List of ChatResult objects

### `batch_with_progress(inputs, max_concurrency=5, stop=None, callback=None)`

Process multiple message lists with progress callback.

**Parameters:**
- `inputs`: List of message lists to process
- `max_concurrency`: Maximum concurrent requests
- `stop`: Optional stop sequences
- `callback`: Progress callback function

**Returns:** List of ChatResult objects

### `abatch_estimate_cost(inputs, max_output_tokens=1000)`

Estimate batch cost before execution.

**Parameters:**
- `inputs`: List of message lists to estimate
- `max_output_tokens`: Expected output tokens per request

**Returns:** Dictionary with cost estimates
