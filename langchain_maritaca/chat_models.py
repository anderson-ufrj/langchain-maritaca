"""Maritaca AI Chat wrapper for LangChain.

Maritaca AI provides Brazilian Portuguese-optimized language models,
including the Sabiá family of models.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_maritaca.version import __version__

# HTTP status code for rate limiting
HTTP_TOO_MANY_REQUESTS = 429

# Logger for context window warnings
logger = logging.getLogger(__name__)

# Context window limits for Maritaca models (in tokens)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "sabia-3.1": 128000,
    "sabiazinho-3.1": 32000,
    "sabiazinho-4": 128000,
}
DEFAULT_CONTEXT_LIMIT = 128000

# Warning threshold (percentage of context used)
CONTEXT_WARNING_THRESHOLD = 0.9  # Warn at 90% usage

# Model specifications for selection helper
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "sabia-3.1": {
        "context_limit": 128000,
        "input_cost_per_1m": 5.00,
        "output_cost_per_1m": 10.00,
        "complexity": "high",
        "speed": "medium",
        "capabilities": ["complex_reasoning", "long_context", "tool_calling", "vision"],
        "description": "Most capable model, best for complex tasks",
    },
    "sabiazinho-3.1": {
        "context_limit": 32000,
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 3.00,
        "complexity": "medium",
        "speed": "fast",
        "capabilities": ["simple_tasks", "quick_responses", "tool_calling", "vision"],
        "description": "Fast and economical, great for simple tasks",
    },
    "sabiazinho-4": {
        "context_limit": 128000,
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 4.00,
        "complexity": "medium",
        "speed": "fast",
        "capabilities": ["simple_tasks", "quick_responses", "tool_calling", "vision"],
        "description": "Latest fast model with vision support and 128k context",
    },
}


class ChatMaritaca(BaseChatModel):
    r"""Maritaca AI Chat large language models API.

    Maritaca AI provides Brazilian Portuguese-optimized language models,
    offering excellent performance for Portuguese text generation, analysis,
    and understanding tasks.

    To use, you should have the environment variable `MARITACA_API_KEY`
    set with your API key, or pass it as a named parameter to the constructor.

    Setup:
        Install `langchain-maritaca` and set environment variable
        `MARITACA_API_KEY`.

        ```bash
        pip install -U langchain-maritaca
        export MARITACA_API_KEY="your-api-key"
        ```

    Key init args - completion params:
        model:
            Name of Maritaca model to use. Available models:
            - `sabia-3.1` (default): Most capable model
            - `sabiazinho-3.1`: Faster and more economical
        temperature:
            Sampling temperature. Ranges from 0.0 to 2.0.
        max_tokens:
            Max number of tokens to generate.

    Key init args - client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Maritaca API key. If not passed in will be read from
            env var `MARITACA_API_KEY`.

    Instantiate:
        ```python
        from langchain_maritaca import ChatMaritaca

        model = ChatMaritaca(
            model="sabia-3.1",
            temperature=0.7,
            max_retries=2,
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "Você é um assistente prestativo."),
            ("human", "Qual é a capital do Brasil?"),
        ]
        model.invoke(messages)
        ```
        ```python
        AIMessage(
            content="A capital do Brasil é Brasília.",
            response_metadata={"model": "sabia-3.1", "finish_reason": "stop"},
        )
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

    Async:
        ```python
        await model.ainvoke(messages)
        ```
    """

    client: Any = Field(default=None, exclude=True)
    """Sync HTTP client."""

    async_client: Any = Field(default=None, exclude=True)
    """Async HTTP client."""

    model_name: str = Field(default="sabia-3.1", alias="model")
    """Model name to use.

    Available models:
    - sabia-3.1: Most capable model, best for complex tasks
    - sabiazinho-3.1: Fast and economical, great for simple tasks
    """

    temperature: float = 0.7
    """Sampling temperature (0.0 to 2.0)."""

    max_tokens: int | None = Field(default=None)
    """Maximum number of tokens to generate."""

    top_p: float = 0.9
    """Top-p sampling parameter."""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    frequency_penalty: float = 0.0
    """Frequency penalty (-2.0 to 2.0)."""

    presence_penalty: float = 0.0
    """Presence penalty (-2.0 to 2.0)."""

    maritaca_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("MARITACA_API_KEY", default=None),
    )
    """Maritaca API key. Automatically inferred from env var `MARITACA_API_KEY`."""

    maritaca_api_base: str = Field(
        alias="base_url",
        default_factory=from_env(
            "MARITACA_API_BASE", default="https://chat.maritaca.ai/api"
        ),
    )
    """Base URL for Maritaca API."""

    request_timeout: float | None = Field(default=60.0, alias="timeout")
    """Timeout for requests in seconds."""

    max_retries: int = 2
    """Maximum number of retries."""

    retry_if_rate_limited: bool = True
    """Whether to automatically retry when rate limited (HTTP 429)."""

    retry_delay: float = 1.0
    """Initial delay in seconds between retries for non-rate-limit errors."""

    retry_max_delay: float = 60.0
    """Maximum delay in seconds between retries."""

    retry_multiplier: float = 2.0
    """Multiplier for exponential backoff between retries."""

    streaming: bool = False
    """Whether to stream results."""

    n: int = 1
    """Number of completions to generate."""

    max_context_tokens: int | None = Field(default=None)
    """Maximum context window tokens to use.

    If set, messages will be automatically truncated to fit within this limit.
    If not set, the model's default context limit is used.
    """

    auto_truncate: bool = False
    """Whether to automatically truncate messages when exceeding context limit.

    If True, older messages (except system messages) will be removed to fit
    within the context window. If False, a warning is emitted instead.
    """

    context_warning_threshold: float = 0.9
    """Emit warning when context usage exceeds this percentage (0.0 to 1.0)."""

    tools: list[dict[str, Any]] | None = Field(default=None, exclude=True)
    """List of tools (functions) available for the model to call."""

    tool_choice: str | dict[str, Any] | None = Field(default=None, exclude=True)
    """Control which tool is called. Options: 'auto', 'required', or specific tool."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that API key exists and initialize HTTP clients."""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        # Validate retry parameters
        if self.retry_delay < 0:
            msg = "retry_delay must be non-negative."
            raise ValueError(msg)
        if self.retry_max_delay < self.retry_delay:
            msg = "retry_max_delay must be greater than or equal to retry_delay."
            raise ValueError(msg)
        if self.retry_multiplier < 1.0:
            msg = "retry_multiplier must be at least 1.0."
            raise ValueError(msg)

        # Ensure temperature is not exactly 0 (causes issues with some APIs)
        if self.temperature == 0:
            self.temperature = 1e-8

        # Initialize HTTP clients
        api_key = (
            self.maritaca_api_key.get_secret_value() if self.maritaca_api_key else ""
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"langchain-maritaca/{__version__}",
        }

        if not self.client:
            self.client = httpx.Client(
                base_url=self.maritaca_api_base,
                headers=headers,
                timeout=httpx.Timeout(self.request_timeout),
            )

        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.maritaca_api_base,
                headers=headers,
                timeout=httpx.Timeout(self.request_timeout),
            )

        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {"maritaca_api_key": "MARITACA_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "maritaca-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="maritaca",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Maritaca API."""
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        if self.tools is not None:
            params["tools"] = self.tools
        if self.tool_choice is not None:
            params["tool_choice"] = self.tool_choice
        return params

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        """Bind tools to this chat model.

        Args:
            tools: A list of tools to bind. Can be:
                - Dict with OpenAI tool schema
                - Pydantic BaseModel class
                - Python function with type hints
                - LangChain BaseTool instance
            tool_choice: Control which tool is called:
                - "auto": Model decides (default)
                - "required": Model must call a tool
                - {"type": "function", "function": {"name": "..."}}:
                  Force specific tool
            **kwargs: Additional arguments passed to the model.

        Returns:
            A Runnable that will pass the tools to the model.

        Example:
            .. code-block:: python

                from pydantic import BaseModel, Field


                class GetWeather(BaseModel):
                    '''Get the weather for a location.'''

                    location: str = Field(description="City name")


                model = ChatMaritaca()
                model_with_tools = model.bind_tools([GetWeather])
                response = model_with_tools.invoke("What's the weather in São Paulo?")
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel] | None = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, dict[str, Any] | BaseModel]:
        """Create a runnable that returns structured output matching a schema.

        Uses the model's tool-calling or JSON mode capabilities to guarantee
        output conforms to the specified schema.

        Args:
            schema: The output schema. Can be:
                - A Pydantic BaseModel class
                - A dictionary with JSON Schema
            method: The method to use for structured output:
                - "function_calling": Uses tool calling (default, recommended)
                - "json_mode": Uses JSON response format
            include_raw: If True, returns a dict with keys:
                - "raw": The raw model response (BaseMessage)
                - "parsed": The parsed structured output
                - "parsing_error": Any parsing error that occurred
            **kwargs: Additional arguments passed to the model.

        Returns:
            A Runnable that outputs structured data matching the schema.

        Example:
            .. code-block:: python

                from pydantic import BaseModel, Field


                class Person(BaseModel):
                    '''Information about a person.'''

                    name: str = Field(description="Person's name")
                    age: int = Field(description="Person's age")


                model = ChatMaritaca()
                structured_model = model.with_structured_output(Person)
                result = structured_model.invoke("João tem 25 anos")
                # Returns: Person(name="João", age=25)

        Note:
            The "function_calling" method is more reliable as it uses the
            model's native tool calling capabilities.
        """
        if schema is None:
            msg = "schema must be specified for with_structured_output"
            raise ValueError(msg)

        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)

        if method == "function_calling":
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                **kwargs,
            )
            if is_pydantic_schema:
                # Type narrowing: we know schema is type[BaseModel] here
                pydantic_schema: type[BaseModel] = schema  # type: ignore[assignment]
                output_parser: Runnable[Any, Any] = PydanticToolsParser(
                    tools=[pydantic_schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )

        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                **kwargs,
            )
            if is_pydantic_schema:
                # Type narrowing: we know schema is type[BaseModel] here
                pydantic_schema = schema  # type: ignore[assignment]
                output_parser = PydanticOutputParser(pydantic_object=pydantic_schema)
            else:
                output_parser = JsonOutputParser()

        else:
            msg = (
                f"Unrecognized method argument. Expected 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback

        return llm | output_parser

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion."""
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        response = self._make_request(message_dicts, params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a chat completion."""
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        response = await self._amake_request(message_dicts, params)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat completion."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        with self.client.stream(
            "POST",
            "/chat/completions",
            json={"messages": message_dicts, **params},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if not chunk.get("choices"):
                            continue
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        message_chunk = AIMessageChunk(content=content)
                        generation_info = {}

                        if finish_reason := choice.get("finish_reason"):
                            generation_info["finish_reason"] = finish_reason
                            generation_info["model"] = self.model_name

                        if generation_info:
                            message_chunk = message_chunk.model_copy(
                                update={"response_metadata": generation_info}
                            )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )

                        yield generation_chunk

                    except json.JSONDecodeError:
                        continue

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream a chat completion."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        async with self.async_client.stream(
            "POST",
            "/chat/completions",
            json={"messages": message_dicts, **params},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if not chunk.get("choices"):
                            continue
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        message_chunk = AIMessageChunk(content=content)
                        generation_info = {}

                        if finish_reason := choice.get("finish_reason"):
                            generation_info["finish_reason"] = finish_reason
                            generation_info["model"] = self.model_name

                        if generation_info:
                            message_chunk = message_chunk.model_copy(
                                update={"response_metadata": generation_info}
                            )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            await run_manager.on_llm_new_token(
                                token=generation_chunk.text, chunk=generation_chunk
                            )

                        yield generation_chunk

                    except json.JSONDecodeError:
                        continue

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in seconds, capped at retry_max_delay.
        """
        delay = self.retry_delay * (self.retry_multiplier**attempt)
        return min(delay, self.retry_max_delay)

    def _make_request(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a sync request to Maritaca API."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json={"messages": messages, **params},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                is_rate_limited = e.response.status_code == HTTP_TOO_MANY_REQUESTS
                can_retry = attempt < self.max_retries
                if is_rate_limited and self.retry_if_rate_limited and can_retry:
                    retry_after = int(
                        e.response.headers.get("Retry-After", self.retry_max_delay)
                    )
                    time.sleep(retry_after)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                raise
        msg = f"Failed after {self.max_retries + 1} attempts"
        raise RuntimeError(msg)

    async def _amake_request(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make an async request to Maritaca API."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.async_client.post(
                    "/chat/completions",
                    json={"messages": messages, **params},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                is_rate_limited = e.response.status_code == HTTP_TOO_MANY_REQUESTS
                can_retry = attempt < self.max_retries
                if is_rate_limited and self.retry_if_rate_limited and can_retry:
                    retry_after = int(
                        e.response.headers.get("Retry-After", self.retry_max_delay)
                    )
                    await asyncio.sleep(retry_after)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
        msg = f"Failed after {self.max_retries + 1} attempts"
        raise RuntimeError(msg)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert LangChain messages to Maritaca format."""
        params = self._default_params.copy()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: dict[str, Any]) -> ChatResult:
        """Create a ChatResult from Maritaca API response."""
        generations = []
        token_usage = response.get("usage", {})

        for choice in response.get("choices", []):
            message = _convert_dict_to_message(choice.get("message", {}))

            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)

            generation_info = {"finish_reason": choice.get("finish_reason")}
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model": response.get("model", self.model_name),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def get_num_tokens(self, text: str) -> int:
        """Get the estimated number of tokens in a text string.

        Uses tiktoken if available, otherwise falls back to character-based
        estimation (~4 characters per token).

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated number of tokens.

        Example:
            .. code-block:: python

                model = ChatMaritaca()
                tokens = model.get_num_tokens("Olá, como você está?")
                print(f"Estimated tokens: {tokens}")

        Note:
            This is an estimate. Actual token count may vary depending on
            the model's tokenizer. For precise counts, use the API response.
        """
        try:
            import tiktoken

            # Use cl100k_base as approximation (GPT-4 tokenizer)
            # Works reasonably well for Portuguese text
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: estimate ~4 characters per token
            # This is a rough approximation
            return max(1, len(text) // 4)

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Get the estimated number of tokens in a list of messages.

        Accounts for message formatting overhead (role prefixes, separators).

        Args:
            messages: List of LangChain messages.
            tools: Optional list of tools (not used, for signature compatibility).

        Returns:
            Estimated total number of tokens.

        Example:
            .. code-block:: python

                from langchain_core.messages import HumanMessage, SystemMessage

                model = ChatMaritaca()
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="What is the capital of Brazil?"),
                ]
                tokens = model.get_num_tokens_from_messages(messages)
                print(f"Estimated tokens: {tokens}")

        Note:
            This is an estimate. Actual token count may vary.
        """
        total_tokens = 0

        # Tokens per message overhead (role, separators, etc.)
        # Based on OpenAI's token counting documentation
        tokens_per_message = 4  # <|start|>role\ncontent<|end|>

        for message in messages:
            total_tokens += tokens_per_message

            # Count content tokens
            content = message.content
            if isinstance(content, str):
                total_tokens += self.get_num_tokens(content)
            elif isinstance(content, list):
                # Handle multi-part content
                for part in content:
                    if isinstance(part, str):
                        total_tokens += self.get_num_tokens(part)
                    elif isinstance(part, dict) and "text" in part:
                        total_tokens += self.get_num_tokens(part["text"])

            # Add role tokens
            if isinstance(message, HumanMessage):
                total_tokens += 1  # "user"
            elif isinstance(message, AIMessage):
                total_tokens += 1  # "assistant"
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # Add tokens for tool call structure
                        total_tokens += self.get_num_tokens(tool_call.get("name", ""))
                        args_str = json.dumps(tool_call.get("args", {}))
                        total_tokens += self.get_num_tokens(args_str)
                        total_tokens += 10  # Overhead for tool call structure
            elif isinstance(message, SystemMessage):
                total_tokens += 1  # "system"
            elif isinstance(message, ToolMessage):
                total_tokens += 1  # "tool"
                total_tokens += 2  # tool_call_id overhead

        # Add tokens for message boundaries
        total_tokens += 3  # <|start|>assistant

        return total_tokens

    def estimate_cost(
        self,
        messages: list[BaseMessage],
        max_output_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Estimate the cost of a request before making it.

        Uses token counting and Maritaca AI pricing to estimate costs.

        Args:
            messages: List of input messages.
            max_output_tokens: Expected maximum output tokens (default: 1000).

        Returns:
            Dictionary with cost estimates:
                - input_tokens: Estimated input token count
                - output_tokens: Expected output tokens
                - input_cost: Estimated input cost in USD
                - output_cost: Estimated output cost in USD
                - total_cost: Total estimated cost in USD

        Example:
            .. code-block:: python

                model = ChatMaritaca(model="sabia-3.1")
                messages = [HumanMessage(content="Tell me a long story")]

                estimate = model.estimate_cost(messages, max_output_tokens=2000)
                print(f"Estimated cost: ${estimate['total_cost']:.6f}")

        Note:
            Prices are estimates based on public pricing and may change.
            Check https://www.maritaca.ai/ for current pricing.
        """
        # Maritaca AI pricing (USD per 1M tokens)
        pricing = {
            "sabia-3.1": {"input": 0.50, "output": 1.50},
            "sabiazinho-3.1": {"input": 0.10, "output": 0.30},
            "default": {"input": 0.50, "output": 1.50},
        }

        model_pricing = pricing.get(self.model_name, pricing["default"])

        input_tokens = self.get_num_tokens_from_messages(messages)
        output_tokens = max_output_tokens

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": self.model_name,
        }

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model.

        Returns the custom limit if set via max_context_tokens, otherwise
        returns the model's default context limit.

        Returns:
            Maximum number of tokens allowed in the context window.

        Example:
            .. code-block:: python

                model = ChatMaritaca(model="sabia-3.1")
                print(f"Context limit: {model.get_context_limit()}")  # 32768

                model = ChatMaritaca(model="sabiazinho-3.1")
                print(f"Context limit: {model.get_context_limit()}")  # 8192
        """
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        return MODEL_CONTEXT_LIMITS.get(self.model_name, DEFAULT_CONTEXT_LIMIT)

    def check_context_usage(
        self,
        messages: list[BaseMessage],
        warn: bool = True,
    ) -> dict[str, Any]:
        """Check context window usage for a list of messages.

        Args:
            messages: List of messages to check.
            warn: Whether to emit a warning if usage exceeds threshold.

        Returns:
            Dictionary with context usage information:
                - tokens: Estimated token count
                - limit: Context window limit
                - usage_percent: Percentage of context used
                - exceeds_limit: Whether context is exceeded
                - exceeds_threshold: Whether warning threshold is exceeded

        Example:
            .. code-block:: python

                model = ChatMaritaca()
                messages = [HumanMessage(content="...long text...")]
                usage = model.check_context_usage(messages)
                print(f"Using {usage['usage_percent']:.1%} of context")
        """
        tokens = self.get_num_tokens_from_messages(messages)
        limit = self.get_context_limit()
        usage_percent = tokens / limit if limit > 0 else 0

        exceeds_limit = tokens > limit
        exceeds_threshold = usage_percent >= self.context_warning_threshold

        if warn and exceeds_threshold and not exceeds_limit:
            warnings.warn(
                f"Context usage at {usage_percent:.1%} ({tokens}/{limit} tokens). "
                "Consider truncating messages to avoid errors.",
                stacklevel=2,
            )

        if warn and exceeds_limit:
            warnings.warn(
                f"Context limit exceeded: {tokens} tokens > {limit} limit. "
                "Request may fail. Use truncate_messages() to fit within limits.",
                stacklevel=2,
            )

        return {
            "tokens": tokens,
            "limit": limit,
            "usage_percent": usage_percent,
            "exceeds_limit": exceeds_limit,
            "exceeds_threshold": exceeds_threshold,
        }

    def truncate_messages(
        self,
        messages: list[BaseMessage],
        max_tokens: int | None = None,
        preserve_system: bool = True,
        preserve_recent: int = 1,
    ) -> list[BaseMessage]:
        """Truncate messages to fit within context window.

        Removes older messages (keeping system messages and recent messages)
        until the total token count is within the limit.

        Args:
            messages: List of messages to truncate.
            max_tokens: Maximum tokens allowed. Defaults to model's context limit.
            preserve_system: Whether to preserve system messages.
            preserve_recent: Number of recent non-system messages to always keep.

        Returns:
            Truncated list of messages that fits within the token limit.

        Example:
            .. code-block:: python

                model = ChatMaritaca()
                long_conversation = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="First message"),
                    AIMessage(content="First response"),
                    HumanMessage(content="Second message"),
                    AIMessage(content="Second response"),
                    HumanMessage(content="Third message"),  # Most recent
                ]

                # Truncate to 1000 tokens, keeping system + last message
                truncated = model.truncate_messages(
                    long_conversation,
                    max_tokens=1000,
                    preserve_recent=2,  # Keep last 2 non-system messages
                )
        """
        if not messages:
            return []

        limit = max_tokens or self.get_context_limit()

        # Check if already within limit
        current_tokens = self.get_num_tokens_from_messages(messages)
        if current_tokens <= limit:
            return messages.copy()

        # Separate system messages and other messages
        system_messages: list[BaseMessage] = []
        other_messages: list[BaseMessage] = []

        for msg in messages:
            if preserve_system and isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Calculate tokens used by system messages
        system_tokens = (
            self.get_num_tokens_from_messages(system_messages) if system_messages else 0
        )
        available_tokens = limit - system_tokens

        if available_tokens <= 0:
            # System messages alone exceed limit, return just system messages
            logger.warning(
                "System messages alone exceed context limit. "
                "Consider shortening system prompt."
            )
            return system_messages.copy()

        # Keep recent messages, remove older ones
        preserved_recent = other_messages[-preserve_recent:] if preserve_recent else []
        candidates = other_messages[:-preserve_recent] if preserve_recent else []

        # Start with preserved recent messages
        result_other: list[BaseMessage] = []
        used_tokens = self.get_num_tokens_from_messages(preserved_recent)

        # Add older messages from most recent to oldest until we can't fit more
        for msg in reversed(candidates):
            msg_tokens = self.get_num_tokens_from_messages([msg])
            if used_tokens + msg_tokens <= available_tokens:
                result_other.insert(0, msg)
                used_tokens += msg_tokens

        # Add preserved recent messages at the end
        result_other.extend(preserved_recent)

        # Combine system messages and other messages
        final_result = system_messages + result_other

        if len(final_result) < len(messages):
            removed = len(messages) - len(final_result)
            logger.info(
                f"Truncated {removed} messages to fit within {limit} token limit."
            )

        return final_result

    @classmethod
    def list_available_models(cls) -> dict[str, dict[str, Any]]:
        """List all available Maritaca models with their specifications.

        Returns:
            Dictionary mapping model names to their specifications including
            context limits, pricing, and capabilities.

        Example:
            .. code-block:: python

                models = ChatMaritaca.list_available_models()
                for name, spec in models.items():
                    print(f"{name}: {spec['description']}")
                    print(f"  Context: {spec['context_limit']} tokens")
                    print(f"  Cost: ${spec['input_cost_per_1m']}/1M input tokens")
        """
        return MODEL_SPECS.copy()

    @classmethod
    def recommend_model(
        cls,
        task_complexity: Literal["simple", "medium", "complex"] = "medium",
        input_length: int | None = None,
        priority: Literal["cost", "speed", "quality"] = "quality",
    ) -> dict[str, Any]:
        """Recommend the best model for a given task.

        Analyzes task requirements and returns the most suitable model
        along with reasoning for the recommendation.

        Args:
            task_complexity: Complexity of the task:
                - "simple": Quick Q&A, simple text, summarization
                - "medium": Standard conversations, moderate reasoning
                - "complex": Complex reasoning, analysis, long-form generation
            input_length: Estimated input token count. If provided, ensures
                the recommended model's context window can accommodate it.
            priority: Optimization priority:
                - "cost": Minimize API costs
                - "speed": Maximize response speed
                - "quality": Maximize output quality

        Returns:
            Dictionary with recommendation:
                - model: Recommended model name
                - reason: Explanation for the recommendation
                - specs: Model specifications
                - alternatives: Other viable options

        Example:
            .. code-block:: python

                # Get recommendation for a simple task
                rec = ChatMaritaca.recommend_model(
                    task_complexity="simple",
                    priority="cost",
                )
                print(f"Recommended: {rec['model']}")
                print(f"Reason: {rec['reason']}")

                # Use the recommended model
                model = ChatMaritaca(model=rec["model"])

                # For long input that needs large context
                rec = ChatMaritaca.recommend_model(
                    task_complexity="complex",
                    input_length=10000,
                    priority="quality",
                )
        """
        models = MODEL_SPECS.copy()
        candidates: list[tuple[str, dict[str, Any], int]] = []

        for model_name, specs in models.items():
            score = 0
            context_limit = specs["context_limit"]

            # Filter by context limit if input_length is specified
            if input_length is not None and input_length > context_limit * 0.8:
                # Skip models that can't comfortably fit the input
                continue

            # Score based on task complexity
            model_complexity = specs["complexity"]
            if task_complexity == "simple":
                if model_complexity == "medium":
                    score += 10  # Prefer simpler model
                else:
                    score += 5
            elif task_complexity == "medium":
                score += 7
            else:  # complex
                if model_complexity == "high":
                    score += 10  # Prefer more capable model
                else:
                    score += 3

            # Score based on priority
            if priority == "cost":
                # Lower cost = higher score
                cost = specs["input_cost_per_1m"] + specs["output_cost_per_1m"]
                score += int(20 / (cost + 0.1))  # Normalize to reasonable range
            elif priority == "speed":
                if specs["speed"] == "fast":
                    score += 15
                else:
                    score += 5
            else:  # quality
                if model_complexity == "high":
                    score += 15
                else:
                    score += 5

            candidates.append((model_name, specs, score))

        if not candidates:
            # No suitable model found, return the most capable one
            default_model = "sabia-3.1"
            return {
                "model": default_model,
                "reason": (
                    "Input length requires maximum context window. "
                    f"Consider using {default_model} with message truncation."
                ),
                "specs": models[default_model],
                "alternatives": [],
            }

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        best_model, best_specs, _best_score = candidates[0]

        # Generate reason
        reasons: list[str] = []
        if task_complexity == "simple":
            reasons.append("optimized for simple tasks")
        elif task_complexity == "complex":
            reasons.append("provides superior reasoning capabilities")
        else:
            reasons.append("balanced for general use")

        if priority == "cost":
            reasons.append("minimizes API costs")
        elif priority == "speed":
            reasons.append("maximizes response speed")
        else:
            reasons.append("prioritizes output quality")

        if input_length is not None:
            reasons.append(
                f"accommodates {input_length} token input "
                f"(limit: {best_specs['context_limit']})"
            )

        reason = (
            f"{best_specs['description']}. Selected because it {', '.join(reasons)}."
        )

        # Get alternatives
        alternatives = [
            {"model": name, "specs": specs} for name, specs, score in candidates[1:]
        ]

        return {
            "model": best_model,
            "reason": reason,
            "specs": best_specs,
            "alternatives": alternatives,
        }

    async def abatch_with_concurrency(
        self,
        inputs: list[list[BaseMessage]],
        max_concurrency: int = 5,
        *,
        stop: list[str] | None = None,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[ChatResult | BaseException]:
        """Process multiple message lists concurrently with rate limiting.

        This method provides optimized batch processing by running multiple
        requests in parallel, respecting a configurable concurrency limit.

        Args:
            inputs: List of message lists to process.
            max_concurrency: Maximum number of concurrent requests.
            stop: Optional stop sequences for all requests.
            return_exceptions: If True, return exceptions instead of raising.
            **kwargs: Additional arguments passed to _agenerate.

        Returns:
            List of ChatResult objects (or exceptions if return_exceptions=True).

        Example:
            .. code-block:: python

                import asyncio
                from langchain_maritaca import ChatMaritaca
                from langchain_core.messages import HumanMessage

                model = ChatMaritaca()

                # Multiple inputs to process
                inputs = [
                    [HumanMessage(content="What is the capital of Brazil?")],
                    [HumanMessage(content="What is the capital of Argentina?")],
                    [HumanMessage(content="What is the capital of Chile?")],
                ]

                # Process with max 3 concurrent requests
                results = asyncio.run(model.abatch_with_concurrency(inputs, max_concurrency=3))
                for result in results:
                    print(result.generations[0].text)
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_one(messages: list[BaseMessage]) -> ChatResult:
            async with semaphore:
                return await self._agenerate(messages, stop=stop, **kwargs)

        tasks = [process_one(messages) for messages in inputs]

        if return_exceptions:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = await asyncio.gather(*tasks)

        return results  # type: ignore[return-value]

    def batch_with_progress(
        self,
        inputs: list[list[BaseMessage]],
        max_concurrency: int = 5,
        *,
        stop: list[str] | None = None,
        callback: Callable[[int, int, ChatResult | None], None] | None = None,
        **kwargs: Any,
    ) -> list[ChatResult]:
        """Process multiple message lists with progress callback.

        Synchronous wrapper around abatch_with_concurrency that provides
        progress updates via callback.

        Args:
            inputs: List of message lists to process.
            max_concurrency: Maximum number of concurrent requests.
            stop: Optional stop sequences for all requests.
            callback: Optional callback function called after each completion.
                Signature: callback(completed_count, total_count, result)
            **kwargs: Additional arguments passed to _agenerate.

        Returns:
            List of ChatResult objects.

        Example:
            .. code-block:: python

                from langchain_maritaca import ChatMaritaca
                from langchain_core.messages import HumanMessage

                model = ChatMaritaca()


                def on_progress(completed, total, result):
                    print(f"Progress: {completed}/{total}")


                inputs = [
                    [HumanMessage(content="Question 1")],
                    [HumanMessage(content="Question 2")],
                    [HumanMessage(content="Question 3")],
                ]

                results = model.batch_with_progress(
                    inputs,
                    max_concurrency=2,
                    callback=on_progress,
                )
        """

        async def run_batch() -> list[ChatResult]:
            semaphore = asyncio.Semaphore(max_concurrency)
            total = len(inputs)
            results: list[tuple[int, ChatResult]] = []

            async def process_one(
                idx: int, messages: list[BaseMessage]
            ) -> tuple[int, ChatResult]:
                async with semaphore:
                    result = await self._agenerate(messages, stop=stop, **kwargs)
                    return idx, result

            tasks = [process_one(i, messages) for i, messages in enumerate(inputs)]

            # Process as they complete
            for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
                idx, result = await coro
                if callback:
                    callback(completed, total, result)
                results.append((idx, result))

            # Sort by original order
            results.sort(key=lambda x: x[0])
            return [r for _, r in results]

        return asyncio.run(run_batch())

    async def abatch_estimate_cost(
        self,
        inputs: list[list[BaseMessage]],
        max_output_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Estimate total cost for a batch of requests before execution.

        Args:
            inputs: List of message lists to estimate.
            max_output_tokens: Expected max output tokens per request.

        Returns:
            Dictionary with batch cost estimates:
                - total_requests: Number of requests
                - total_input_tokens: Estimated total input tokens
                - total_output_tokens: Expected total output tokens
                - total_input_cost: Estimated total input cost
                - total_output_cost: Estimated total output cost
                - total_cost: Total estimated cost

        Example:
            .. code-block:: python

                inputs = [
                    [HumanMessage(content="Question 1")],
                    [HumanMessage(content="Question 2")],
                ]
                estimate = await model.abatch_estimate_cost(inputs)
                print(f"Batch cost: ${estimate['total_cost']:.6f}")
        """
        total_input_tokens = 0
        for messages in inputs:
            total_input_tokens += self.get_num_tokens_from_messages(messages)

        total_output_tokens = max_output_tokens * len(inputs)

        # Get pricing
        pricing = {
            "sabia-3.1": {"input": 0.50, "output": 1.50},
            "sabiazinho-3.1": {"input": 0.10, "output": 0.30},
            "default": {"input": 0.50, "output": 1.50},
        }
        model_pricing = pricing.get(self.model_name, pricing["default"])

        input_cost = (total_input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (total_output_tokens / 1_000_000) * model_pricing["output"]

        return {
            "total_requests": len(inputs),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_input_cost": input_cost,
            "total_output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": self.model_name,
        }


def _format_image_content(content: str | list[Any]) -> str | list[dict[str, Any]]:
    """Format message content, handling multimodal inputs with images.

    Converts LangChain standard image blocks and OpenAI image_url format
    to Maritaca API format (Anthropic-style).

    Args:
        content: String content or list of content blocks.

    Returns:
        Formatted content for Maritaca API.
    """
    if isinstance(content, str):
        return content

    # Multimodal content (list of blocks)
    formatted: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            formatted.append({"type": "text", "text": block})
        elif isinstance(block, dict):
            block_type = block.get("type", "text")

            if block_type == "text":
                formatted.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "image":
                # LangChain standard format -> Anthropic format
                image_block: dict[str, Any] = {"type": "image", "source": {}}

                if "base64" in block:
                    image_block["source"] = {
                        "type": "base64",
                        "media_type": block.get("mime_type", "image/png"),
                        "data": block["base64"],
                    }
                elif "url" in block:
                    image_block["source"] = {
                        "type": "url",
                        "url": block["url"],
                    }
                formatted.append(image_block)

            elif block_type == "image_url":
                # OpenAI format compatibility -> Anthropic format
                image_url = block.get("image_url", {})
                url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict)
                    else str(image_url)
                )

                if url.startswith("data:"):
                    # Parse data URI: data:image/png;base64,xxxxx
                    parts = url.split(",", 1)
                    header = parts[0]  # data:image/png;base64
                    data = parts[1] if len(parts) > 1 else ""
                    media_type = header.replace("data:", "").replace(";base64", "")

                    formatted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                else:
                    formatted.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
            else:
                # Pass through other types
                formatted.append(block)

    return formatted


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to Maritaca format.

    Args:
        message: The LangChain message.

    Returns:
        Dictionary in Maritaca API format.
    """
    if isinstance(message, ChatMessage):
        return {
            "role": message.role,
            "content": _format_image_content(message.content),
        }
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": _format_image_content(message.content)}
    if isinstance(message, AIMessage):
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        # Handle tool calls in AIMessage
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in message.tool_calls
            ]
        return result
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    msg = f"Got unknown message type: {type(message)}"
    raise TypeError(msg)


def _convert_dict_to_message(message_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a Maritaca message dict to LangChain message.

    Args:
        message_dict: Dictionary from Maritaca API response.

    Returns:
        LangChain BaseMessage.
    """
    role = message_dict.get("role", "")
    content = message_dict.get("content", "") or ""

    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        # Parse tool_calls if present (handle None from API)
        tool_calls_data = message_dict.get("tool_calls") or []
        tool_calls = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    name=func.get("name", ""),
                    args=args,
                    id=tc.get("id", ""),
                )
            )
        return AIMessage(content=content, tool_calls=tool_calls if tool_calls else [])
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=message_dict.get("tool_call_id", ""),
        )
    return ChatMessage(content=content, role=role)


def _create_usage_metadata(token_usage: dict[str, Any]) -> UsageMetadata:
    """Create usage metadata from Maritaca token usage response.

    Args:
        token_usage: Token usage dict from Maritaca API response.

    Returns:
        UsageMetadata with token counts.
    """
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
