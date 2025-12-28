# Token Counting & Cost Estimation

langchain-maritaca provides built-in methods for counting tokens and estimating costs before making API calls. This helps you manage API costs and ensure your requests fit within model context limits.

## Token Counting Methods

### get_num_tokens

Count the number of tokens in a text string:

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca()

# Count tokens in text
tokens = model.get_num_tokens("Olá, como você está?")
print(f"Tokens: {tokens}")
```

### get_num_tokens_from_messages

Count tokens in a list of messages, including formatting overhead:

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatMaritaca()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Brazil?"),
    AIMessage(content="The capital of Brazil is Brasília."),
    HumanMessage(content="Tell me more about it."),
]

tokens = model.get_num_tokens_from_messages(messages)
print(f"Total tokens: {tokens}")
```

## Cost Estimation

### estimate_cost

Estimate the cost of a request before making it:

```python
from langchain_core.messages import HumanMessage

model = ChatMaritaca(model="sabia-3.1")

messages = [
    HumanMessage(content="Tell me a long story about Brazil")
]

# Estimate cost with expected output tokens
estimate = model.estimate_cost(messages, max_output_tokens=2000)

print(f"Input tokens: {estimate['input_tokens']}")
print(f"Output tokens: {estimate['output_tokens']}")
print(f"Input cost: ${estimate['input_cost']:.6f}")
print(f"Output cost: ${estimate['output_cost']:.6f}")
print(f"Total cost: ${estimate['total_cost']:.6f}")
```

### Compare Model Costs

Compare costs between different models:

```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Tell me about Brazilian history")]

# Compare sabia vs sabiazinho
sabia = ChatMaritaca(model="sabia-3.1")
sabiazinho = ChatMaritaca(model="sabiazinho-3.1")

sabia_cost = sabia.estimate_cost(messages, max_output_tokens=1000)
sabiazinho_cost = sabiazinho.estimate_cost(messages, max_output_tokens=1000)

print(f"sabia-3.1 cost: ${sabia_cost['total_cost']:.6f}")
print(f"sabiazinho-3.1 cost: ${sabiazinho_cost['total_cost']:.6f}")
print(f"Savings with sabiazinho: {(1 - sabiazinho_cost['total_cost']/sabia_cost['total_cost'])*100:.1f}%")
```

## Pricing Reference

Current estimated pricing for Maritaca AI models:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| sabia-3.1 | $0.50 | $1.50 |
| sabiazinho-3.1 | $0.10 | $0.30 |

> **Note**: Prices are estimates and may change. Check [Maritaca AI](https://www.maritaca.ai/) for current pricing.

## Tokenizer Options

By default, token counting uses tiktoken if available, with a character-based fallback:

### With tiktoken (Recommended)

Install with the tokenizer extra for accurate counting:

```bash
pip install langchain-maritaca[tokenizer]
```

This uses OpenAI's `cl100k_base` encoding, which provides good accuracy for Portuguese text.

### Without tiktoken (Fallback)

If tiktoken is not installed, a character-based estimation is used (~4 characters per token). This is less accurate but requires no additional dependencies.

## Use Cases

### Budget Monitoring

Track costs across multiple calls:

```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback

# Use callback for actual costs after calls
cost_cb = CostTrackingCallback()
model = ChatMaritaca(callbacks=[cost_cb])

# Or estimate before calls
messages = [HumanMessage(content="Hello")]
estimate = model.estimate_cost(messages)
print(f"Estimated: ${estimate['total_cost']:.6f}")

model.invoke(messages)
print(f"Actual: ${cost_cb.total_cost:.6f}")
```

### Context Window Management

Ensure messages fit within context limits:

```python
MAX_CONTEXT = 8192  # Example context window

model = ChatMaritaca()
messages = [...]  # Your conversation history

tokens = model.get_num_tokens_from_messages(messages)

if tokens > MAX_CONTEXT - 1000:  # Leave room for response
    print("Warning: Approaching context limit!")
    # Truncate or summarize older messages
```

### Cost-Aware Routing

Route to different models based on cost:

```python
def get_model_for_task(messages, complexity="simple"):
    """Choose model based on task complexity and cost."""
    sabia = ChatMaritaca(model="sabia-3.1")
    sabiazinho = ChatMaritaca(model="sabiazinho-3.1")

    if complexity == "complex":
        return sabia

    # For simple tasks, use the cheaper model
    estimate = sabiazinho.estimate_cost(messages)
    if estimate["total_cost"] < 0.001:  # Under $0.001
        return sabiazinho

    return sabia
```

## Best Practices

1. **Estimate before long operations**: For batch processing, estimate total cost before starting
2. **Use sabiazinho for simple tasks**: It's 5x cheaper for input and output
3. **Monitor actual vs estimated**: Use `CostTrackingCallback` to verify estimates
4. **Install tiktoken**: For more accurate counting, especially with Portuguese text
5. **Cache repeated queries**: Use LangChain caching to avoid redundant API calls
