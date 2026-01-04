# Model Selection

This guide helps you choose the best Maritaca model for your use case.

## Available Models

ChatMaritaca provides utilities to discover and select the optimal model:

```python
from langchain_maritaca import ChatMaritaca

# List all available models with specifications
models = ChatMaritaca.list_available_models()

for name, spec in models.items():
    print(f"{name}:")
    print(f"  Context: {spec['context_limit']:,} tokens")
    print(f"  Input cost: ${spec['input_cost_per_1m']}/1M tokens")
    print(f"  Output cost: ${spec['output_cost_per_1m']}/1M tokens")
    print(f"  Speed: {spec['speed']}")
    print(f"  Complexity: {spec['complexity']}")
```

## Model Comparison

| Model | Context | Speed | Complexity | Cost (Input/Output per 1M) |
|-------|---------|-------|------------|---------------------------|
| sabia-3.1 | 32,768 | Medium | High | $0.50 / $1.50 |
| sabiazinho-3.1 | 8,192 | Fast | Medium | $0.10 / $0.30 |

### When to Use sabia-3.1

- Complex reasoning tasks
- Long document analysis
- Multi-step problem solving
- High-quality output is critical
- Long conversations requiring 32K context

### When to Use sabiazinho-3.1

- Simple Q&A
- Quick responses needed
- Cost-sensitive applications
- High-volume, simple tasks
- Conversations under 8K tokens

## Automatic Model Recommendation

Use `recommend_model()` to get intelligent suggestions:

```python
# Simple task, optimize for cost
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    priority="cost",
)
print(f"Recommended: {rec['model']}")
print(f"Reason: {rec['reason']}")

# Use the recommended model
model = ChatMaritaca(model=rec['model'])
```

### Task Complexity

- **simple**: Quick Q&A, simple text, summarization
- **medium**: Standard conversations, moderate reasoning
- **complex**: Complex reasoning, analysis, long-form generation

### Priority

- **cost**: Minimize API costs
- **speed**: Maximize response speed
- **quality**: Maximize output quality

### Input Length Consideration

```python
# For long inputs, recommend models with sufficient context
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    input_length=10000,  # 10K tokens needed
)

# Will recommend sabia-3.1 due to context requirements
print(rec['model'])  # 'sabia-3.1'
```

## Practical Examples

### Cost Optimization

```python
# For a chatbot handling simple queries
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    priority="cost",
)

model = ChatMaritaca(model=rec['model'])  # sabiazinho-3.1
```

### Quality Optimization

```python
# For document analysis requiring deep understanding
rec = ChatMaritaca.recommend_model(
    task_complexity="complex",
    priority="quality",
)

model = ChatMaritaca(model=rec['model'])  # sabia-3.1
```

### Speed Optimization

```python
# For real-time applications needing fast responses
rec = ChatMaritaca.recommend_model(
    task_complexity="medium",
    priority="speed",
)

model = ChatMaritaca(model=rec['model'])  # sabiazinho-3.1
```

### Dynamic Selection

```python
def select_model_for_task(task_description, input_text):
    # Estimate input length
    input_length = len(input_text) // 4  # Rough token estimate

    # Determine complexity
    if any(word in task_description.lower() for word in ['analyze', 'complex', 'detailed']):
        complexity = 'complex'
    elif any(word in task_description.lower() for word in ['simple', 'quick', 'brief']):
        complexity = 'simple'
    else:
        complexity = 'medium'

    rec = ChatMaritaca.recommend_model(
        task_complexity=complexity,
        input_length=input_length,
        priority='quality',
    )

    return ChatMaritaca(model=rec['model'])
```

## Alternatives

The recommendation includes alternatives when available:

```python
rec = ChatMaritaca.recommend_model(
    task_complexity="medium",
    priority="quality",
)

print(f"Primary recommendation: {rec['model']}")
print("Alternatives:")
for alt in rec['alternatives']:
    print(f"  - {alt['model']}: {alt['specs']['description']}")
```

## API Reference

### `list_available_models()`

Returns dictionary of all available models with their specifications.

**Returns:** Dict mapping model names to specifications including:
- `context_limit`: Maximum context window
- `input_cost_per_1m`: Cost per 1M input tokens
- `output_cost_per_1m`: Cost per 1M output tokens
- `complexity`: Model capability level
- `speed`: Response speed
- `capabilities`: List of supported features
- `description`: Human-readable description

### `recommend_model(task_complexity, input_length, priority)`

Get intelligent model recommendation.

**Parameters:**
- `task_complexity`: 'simple', 'medium', or 'complex'
- `input_length`: Estimated input token count (optional)
- `priority`: 'cost', 'speed', or 'quality'

**Returns:** Dictionary with:
- `model`: Recommended model name
- `reason`: Explanation for recommendation
- `specs`: Model specifications
- `alternatives`: List of other viable options
