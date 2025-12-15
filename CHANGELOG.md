# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-15

### Added

- Initial release of `langchain-maritaca`
- `ChatMaritaca` class for interacting with Maritaca AI models
- Support for `sabia-3` and `sabiazinho-3` models
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

[Unreleased]: https://github.com/anderson-ufrj/langchain-maritaca/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anderson-ufrj/langchain-maritaca/releases/tag/v0.1.0
