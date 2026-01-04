# Context Window Management

This guide covers how to manage context window limits when working with ChatMaritaca.

## Overview

Large language models have a maximum number of tokens they can process in a single request (the "context window"). When your conversation exceeds this limit, the API will return an error. ChatMaritaca provides tools to help you manage this limitation.

## Model Context Limits

Each Maritaca model has different context window sizes:

| Model | Context Limit | Best For |
|-------|---------------|----------|
| sabia-3.1 | 32,768 tokens | Long conversations, document analysis |
| sabiazinho-3.1 | 8,192 tokens | Quick Q&A, simple tasks |

## Checking Context Usage

Use `check_context_usage()` to monitor how much of your context window is being used:

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatMaritaca()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Brazil's history..."),
]

usage = model.check_context_usage(messages)
print(f"Tokens used: {usage['tokens']}")
print(f"Context limit: {usage['limit']}")
print(f"Usage: {usage['usage_percent']:.1%}")
```

### Automatic Warnings

By default, ChatMaritaca will emit warnings when:
- Context usage exceeds 90% of the limit (configurable via `context_warning_threshold`)
- Context limit is exceeded

```python
# Configure warning threshold
model = ChatMaritaca(context_warning_threshold=0.8)  # Warn at 80%
```

## Truncating Messages

When your conversation exceeds the context limit, use `truncate_messages()` to fit within bounds:

```python
model = ChatMaritaca()

# Long conversation that exceeds limit
long_conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="First question"),
    AIMessage(content="First answer"),
    HumanMessage(content="Second question"),
    AIMessage(content="Second answer"),
    # ... many more messages ...
    HumanMessage(content="Most recent question"),
]

# Truncate to fit within context
truncated = model.truncate_messages(long_conversation)

# Now safe to use
response = model.invoke(truncated)
```

### Truncation Options

```python
truncated = model.truncate_messages(
    messages,
    max_tokens=4096,       # Custom token limit
    preserve_system=True,  # Keep system messages (default: True)
    preserve_recent=2,     # Keep last N non-system messages (default: 1)
)
```

The truncation algorithm:
1. Always preserves system messages (if `preserve_system=True`)
2. Always preserves the most recent messages (specified by `preserve_recent`)
3. Removes older messages starting from the oldest

## Custom Context Limits

You can set a custom context limit lower than the model's maximum:

```python
# Limit to 4096 tokens even though model supports more
model = ChatMaritaca(max_context_tokens=4096)
```

This is useful when:
- You want to leave room for model output
- You're building a chatbot with consistent memory limits
- You want to control costs by limiting input size

## Best Practices

### 1. Monitor Before Long Conversations

```python
def chat_with_monitoring(model, messages, user_input):
    messages.append(HumanMessage(content=user_input))

    # Check before sending
    usage = model.check_context_usage(messages, warn=False)
    if usage['usage_percent'] > 0.8:
        messages = model.truncate_messages(messages, preserve_recent=4)

    response = model.invoke(messages)
    messages.append(response)
    return response
```

### 2. Use Appropriate Models

For long conversations, prefer `sabia-3.1` with its 32K context. For quick interactions, `sabiazinho-3.1` is more economical.

### 3. Summarize Old Context

Instead of truncating, consider summarizing older messages:

```python
def summarize_and_continue(model, messages):
    if len(messages) > 10:
        # Keep system message
        system = messages[0] if isinstance(messages[0], SystemMessage) else None

        # Summarize old messages
        old_messages = messages[1:-2]  # All except system and last 2
        summary_prompt = "Summarize this conversation briefly: " + str(old_messages)
        summary = model.invoke([HumanMessage(content=summary_prompt)])

        # Build new context
        new_messages = []
        if system:
            new_messages.append(system)
        new_messages.append(SystemMessage(content=f"Previous context: {summary.content}"))
        new_messages.extend(messages[-2:])  # Keep last 2 messages

        return new_messages
    return messages
```

## API Reference

### `get_context_limit()`

Returns the context window limit for the current model.

```python
limit = model.get_context_limit()
```

### `check_context_usage(messages, warn=True)`

Returns context usage information and optionally emits warnings.

**Returns:**
- `tokens`: Estimated token count
- `limit`: Context window limit
- `usage_percent`: Percentage of context used
- `exceeds_limit`: Whether context is exceeded
- `exceeds_threshold`: Whether warning threshold is exceeded

### `truncate_messages(messages, max_tokens=None, preserve_system=True, preserve_recent=1)`

Truncates messages to fit within context window.

**Parameters:**
- `messages`: List of messages to truncate
- `max_tokens`: Maximum tokens (default: model's context limit)
- `preserve_system`: Whether to keep system messages
- `preserve_recent`: Number of recent messages to always keep

**Returns:** Truncated list of messages
