# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-10

### Changed

- **LangChain 1.x Support**: Updated dependency to support `langchain-core>=0.3.0,<2.0.0`
  - Fully compatible with langchain-core 1.0, 1.1, and 1.2
  - All 157 unit tests pass with langchain-core 1.2.7
  - No breaking changes to the public API
- This is a minor version bump to signal the expanded compatibility range

### Notes

- Projects using langchain 1.x can now use langchain-maritaca
- Backward compatible with langchain-core 0.3.x

## [0.2.4] - 2026-01-04

### Added

- **Context Window Management**: Automatic context limit handling
  - `get_context_limit()`: Get model's context window size
  - `check_context_usage()`: Check token usage percentage with warnings
  - `truncate_messages()`: Automatically truncate to fit context
  - `max_context_tokens`: Custom context limit parameter
  - `auto_truncate`: Enable automatic message truncation
  - `context_warning_threshold`: Configurable warning threshold (default: 90%)
- **Model Selection Helper**: Intelligent model recommendations
  - `list_available_models()`: List all models with specs
  - `recommend_model()`: Get recommendations based on task complexity, input length, and priority
  - Model specs include context limits, pricing, and capabilities
- **Batch Processing Optimization**: Efficient bulk request handling
  - `abatch_with_concurrency()`: Async batch with configurable concurrency
  - `batch_with_progress()`: Sync batch with progress callbacks
  - `abatch_estimate_cost()`: Estimate cost before batch execution

## [0.2.3] - 2025-12-28

### Added

- **Configurable Retry Logic**: Fine-grained control over retry behavior
  - `retry_if_rate_limited`: Enable/disable auto-retry on HTTP 429 (default: `True`)
  - `retry_delay`: Initial delay between retries in seconds (default: `1.0`)
  - `retry_max_delay`: Maximum delay cap in seconds (default: `60.0`)
  - `retry_multiplier`: Exponential backoff multiplier (default: `2.0`)
  - Validation for all retry parameters
  - Works for both sync and async requests
- **Cache Integration**: LangChain's native caching works out of the box
  - Supports `InMemoryCache`, `SQLiteCache`, `RedisCache`
  - Reduces API costs for repeated queries
  - No code changes required, just set global cache
- **Enhanced Callbacks**: Callback handlers for observability
  - `CostTrackingCallback`: Track token usage and estimate API costs
  - `LatencyTrackingCallback`: Measure response times with P50/P95/P99 percentiles
  - `TokenStreamingCallback`: Monitor streaming token rates
  - `CombinedCallback`: Combined cost and latency tracking
- **Token Counting & Cost Estimation**: Pre-request cost estimation
  - `get_num_tokens()`: Count tokens in text using tiktoken (with fallback)
  - `get_num_tokens_from_messages()`: Count tokens in messages with overhead
  - `estimate_cost()`: Estimate request cost before making it
  - Optional `tiktoken` dependency: `pip install langchain-maritaca[tokenizer]`
- **Codecov Badge**: Test coverage badge in README
- **Portuguese README**: Complete translation of README to Brazilian Portuguese (`README.pt-br.md`)
- **Consolidated Roadmap**: Detailed implementation steps in `docs/planning/ROADMAP.md`
- **Documentation**: Comprehensive guides for caching, callbacks, and token counting

### Changed

- Improved retry logic with exponential backoff using configurable parameters
- Rate limit retry can now be disabled with `retry_if_rate_limited=False`
- Test coverage increased to 107 unit tests

## [0.2.2] - 2025-12-22

### Added

- **Structured Output**: `with_structured_output()` method for ChatMaritaca
  - Returns Pydantic models or dicts directly from LLM responses
  - Supports `function_calling` (default) and `json_mode` methods
  - `include_raw` option to return raw response alongside parsed output
- **Embeddings Support**: `DeepInfraEmbeddings` class for RAG workflows
  - Uses `intfloat/multilingual-e5-large` model (recommended by Maritaca AI)
  - 1024-dimensional embeddings, supports 100 languages including Portuguese
  - Sync and async methods: `embed_query`, `embed_documents`, `aembed_query`, `aembed_documents`
  - Automatic batching with configurable `batch_size`
- **Bilingual Documentation**: MkDocs site with PT-BR and English support
  - Complete user guide, API reference, and examples
  - Dark/light mode toggle
  - Available at GitHub Pages

## [0.2.1] - 2025-12-21

### Fixed

- Fix PyPI metadata: author name and email now correctly displayed

## [0.2.0] - 2025-12-18

### Added

- **Tool Calling / Function Calling**: Full support for binding tools to the model
  - `bind_tools()` method for binding Pydantic models, functions, or tool schemas
  - `tool_choice` parameter to control tool selection ("auto", "required", or specific tool)
  - `ToolMessage` support for tool call responses
  - Full conversation loop support with tool execution
- **Message Conversion**: Extended to handle tool-related messages
  - `AIMessage` with `tool_calls` attribute
  - `ToolMessage` for returning tool execution results
- **Planning Documentation**: Added `docs/planning/future-improvements.md` roadmap

### Changed

- Enhanced `_convert_message_to_dict()` to support tool calls in AIMessage
- Enhanced `_convert_dict_to_message()` to parse tool calls from API response
- Updated `_default_params` to include `tools` and `tool_choice` when configured

## [0.1.1] - 2025-12-15

### Changed

- Update default model from `sabia-3` to `sabia-3.1`
- Update model references to use `sabia-3.1` and `sabiazinho-3.1`
- Sabiá 3.0 models have been discontinued by Maritaca AI

## [0.1.0] - 2025-12-15

### Added

- Initial release of `langchain-maritaca`
- `ChatMaritaca` class for interacting with Maritaca AI models
- Support for `sabia-3.1` and `sabiazinho-3.1` models
- Synchronous and asynchronous generation
- Streaming support (sync and async)
- Automatic retry logic with exponential backoff
- Rate limiting handling
- LangSmith tracing integration
- Usage metadata tracking
- Full type hints and documentation
- Comprehensive test suite

### Features

- **Chat Completions**: Full support for chat-based interactions
- **Streaming**: Real-time token streaming for better UX
- **Async Support**: Native async/await support
- **Retry Logic**: Automatic retries with configurable backoff
- **Rate Limiting**: Graceful handling of API rate limits
- **Tracing**: Built-in LangSmith integration for observability

[Unreleased]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anderson-ufrj/langchain-maritaca/releases/tag/v0.1.0
