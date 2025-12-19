# Future Improvements for langchain-maritaca

This document outlines planned improvements and enhancements for the langchain-maritaca package.

## High Priority

### 1. Embeddings Support
Add `MaritacaEmbeddings` class for RAG (Retrieval-Augmented Generation) workflows.

```python
from langchain_maritaca import MaritacaEmbeddings

embeddings = MaritacaEmbeddings()
vectors = embeddings.embed_documents(["Hello", "World"])
```

**Status**: Planned
**Complexity**: Medium
**Impact**: High - Enables complete RAG pipelines with Maritaca

### 2. Structured Output
Implement `with_structured_output()` method to return Pydantic models directly.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class MedicalEntity(BaseModel):
    value: str = Field(..., description="Entity value")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence score")

class MedicalEntities(BaseModel):
    symptoms: List[MedicalEntity] = Field(default_factory=list)
    medications: List[MedicalEntity] = Field(default_factory=list)
    diagnoses: List[MedicalEntity] = Field(default_factory=list)

model = ChatMaritaca()
structured_model = model.with_structured_output(MedicalEntities)
result = structured_model.invoke("Paciente com dor de cabeça, receitado paracetamol")
# Returns MedicalEntities(symptoms=[...], medications=[...], diagnoses=[...])
```

**Real Use Case**: TelepatiaAI Medical Scribe application needs to extract structured medical data (SOAP notes, symptoms, medications) from Portuguese transcriptions. Currently requires manual JSON parsing which is error-prone when the LLM returns null for optional fields.

**Implementation Approach**:
1. Use `bind_tools()` internally with Pydantic model as tool schema
2. Parse tool call response and return validated Pydantic instance
3. Reference implementations:
   - `langchain-anthropic`: Uses Claude's native tool_use feature
   - `langchain-openai`: Uses function calling with JSON schema

**API Signature** (following LangChain conventions):
```python
def with_structured_output(
    self,
    schema: Type[BaseModel],
    *,
    include_raw: bool = False,
    method: Literal["function_calling", "json_mode"] = "function_calling",
) -> Runnable[LanguageModelInput, BaseModel]:
    ...
```

**Status**: Planned - HIGH PRIORITY
**Complexity**: Medium
**Impact**: High - Simplifies data extraction tasks, enables medical AI applications

---

## Medium Priority

### 3. Cache Integration
Add native support for LangChain caching to reduce API costs.

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())
model = ChatMaritaca()  # Now uses cache automatically
```

**Status**: Planned
**Complexity**: Low
**Impact**: Medium - Reduces costs for repeated queries

### 4. Enhanced Callbacks
More granular callbacks for token-by-token streaming, cost tracking, and latency monitoring.

**Status**: Planned
**Complexity**: Low
**Impact**: Medium - Better observability

### 5. Configurable Retry Logic
Expose retry parameters for more control over error handling.

```python
model = ChatMaritaca(
    retry_if_rate_limited=True,
    retry_delay=1.0,
    retry_max_delay=60.0,
    retry_multiplier=2.0,
)
```

**Status**: Planned
**Complexity**: Low
**Impact**: Medium - Better resilience in production

### 6. Coverage Badge
Add Codecov integration to show test coverage in README.

**Status**: Planned
**Complexity**: Low
**Impact**: Low - Documentation improvement

---

## Low Priority

### 7. Token Counter
Implement `get_num_tokens()` method to estimate token count before API calls.

```python
model = ChatMaritaca()
tokens = model.get_num_tokens("Olá, como vai você?")
```

**Status**: Planned
**Complexity**: Medium (requires tokenizer)
**Impact**: Medium - Helps with cost estimation

### 8. Batch Optimization
Use batch API endpoint if available for better throughput.

**Status**: Planned
**Complexity**: Medium
**Impact**: Low - Performance improvement for batch operations

### 9. Multimodal/Vision Support
Add support for image inputs when Maritaca API supports it.

**Status**: Waiting for API support
**Complexity**: High
**Impact**: High - Enables vision tasks

### 10. Bilingual Documentation
Add Portuguese (PT-BR) version of README since the target audience is Brazilian developers.

**Status**: Planned
**Complexity**: Low
**Impact**: Medium - Better accessibility for Portuguese speakers

---

## Contributing

Contributions are welcome! If you'd like to work on any of these improvements:

1. Open an issue to discuss the implementation approach
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

For questions or suggestions, please open an issue on GitHub.
