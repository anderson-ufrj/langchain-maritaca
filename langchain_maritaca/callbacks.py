"""Callback handlers for ChatMaritaca.

Provides callback handlers for cost tracking, latency monitoring,
and other observability features.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
"""

from __future__ import annotations

import time
from typing import Any, ClassVar
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler for tracking API costs.

    Tracks token usage and calculates costs based on Maritaca AI pricing.
    Costs are estimates based on public pricing and may vary.

    Example:
        .. code-block:: python

            from langchain_maritaca import ChatMaritaca
            from langchain_maritaca.callbacks import CostTrackingCallback

            cost_callback = CostTrackingCallback()
            model = ChatMaritaca(callbacks=[cost_callback])

            model.invoke("Hello, how are you?")

            print(f"Total cost: ${cost_callback.total_cost:.6f}")
            print(f"Total tokens: {cost_callback.total_tokens}")
    """

    # Maritaca AI pricing (USD per 1M tokens) - estimates, check official pricing
    # https://www.maritaca.ai/
    PRICING: ClassVar[dict[str, dict[str, float]]] = {
        "sabia-3.1": {"input": 0.50, "output": 1.50},
        "sabiazinho-3.1": {"input": 0.10, "output": 0.30},
        # Default pricing for unknown models
        "default": {"input": 0.50, "output": 1.50},
    }

    def __init__(self) -> None:
        """Initialize the cost tracking callback."""
        super().__init__()
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0
        self._costs_by_call: list[dict[str, Any]] = []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Calculate cost when LLM call ends."""
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        model = llm_output.get("model", "default")

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        total = token_usage.get("total_tokens", input_tokens + output_tokens)

        # Get pricing for model
        pricing = self.PRICING.get(model, self.PRICING["default"])

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        call_cost = input_cost + output_cost

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total
        self.total_cost += call_cost
        self.call_count += 1

        # Store call details
        self._costs_by_call.append(
            {
                "run_id": str(run_id),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
                "cost": call_cost,
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked costs.

        Returns:
            Dictionary with cost summary including totals and per-call breakdown.
        """
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "average_cost_per_call": (
                self.total_cost / self.call_count if self.call_count > 0 else 0
            ),
            "calls": self._costs_by_call,
        }

    def reset(self) -> None:
        """Reset all tracked costs."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self._costs_by_call = []


class LatencyTrackingCallback(BaseCallbackHandler):
    """Callback handler for tracking API latency.

    Measures time taken for each LLM call and provides statistics.

    Example:
        .. code-block:: python

            from langchain_maritaca import ChatMaritaca
            from langchain_maritaca.callbacks import LatencyTrackingCallback

            latency_callback = LatencyTrackingCallback()
            model = ChatMaritaca(callbacks=[latency_callback])

            model.invoke("Hello!")
            model.invoke("How are you?")

            print(f"Average latency: {latency_callback.average_latency:.2f}s")
            print(f"P95 latency: {latency_callback.p95_latency:.2f}s")
    """

    def __init__(self) -> None:
        """Initialize the latency tracking callback."""
        super().__init__()
        self._start_times: dict[UUID, float] = {}
        self._latencies: list[float] = []
        self.call_count: int = 0

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record start time when LLM call begins."""
        self._start_times[run_id] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Calculate latency when LLM call ends."""
        if run_id in self._start_times:
            latency = time.perf_counter() - self._start_times[run_id]
            self._latencies.append(latency)
            self.call_count += 1
            del self._start_times[run_id]

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Clean up on error."""
        if run_id in self._start_times:
            latency = time.perf_counter() - self._start_times[run_id]
            self._latencies.append(latency)
            self.call_count += 1
            del self._start_times[run_id]

    @property
    def latencies(self) -> list[float]:
        """Get all recorded latencies."""
        return self._latencies.copy()

    @property
    def total_latency(self) -> float:
        """Get total latency across all calls."""
        return sum(self._latencies)

    @property
    def average_latency(self) -> float:
        """Get average latency across all calls."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def min_latency(self) -> float:
        """Get minimum latency."""
        if not self._latencies:
            return 0.0
        return min(self._latencies)

    @property
    def max_latency(self) -> float:
        """Get maximum latency."""
        if not self._latencies:
            return 0.0
        return max(self._latencies)

    @property
    def p50_latency(self) -> float:
        """Get 50th percentile (median) latency."""
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        """Get 95th percentile latency."""
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        """Get 99th percentile latency."""
        return self._percentile(99)

    def _percentile(self, p: float) -> float:
        """Calculate percentile of latencies."""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        index = int((p / 100) * len(sorted_latencies))
        index = min(index, len(sorted_latencies) - 1)
        return sorted_latencies[index]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all latency statistics.

        Returns:
            Dictionary with latency statistics.
        """
        return {
            "call_count": self.call_count,
            "total_latency": self.total_latency,
            "average_latency": self.average_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
        }

    def reset(self) -> None:
        """Reset all tracked latencies."""
        self._start_times = {}
        self._latencies = []
        self.call_count = 0


class TokenStreamingCallback(BaseCallbackHandler):
    """Callback handler for tracking streaming tokens.

    Counts tokens as they stream in and provides timing information.

    Example:
        .. code-block:: python

            from langchain_maritaca import ChatMaritaca
            from langchain_maritaca.callbacks import TokenStreamingCallback

            streaming_callback = TokenStreamingCallback()
            model = ChatMaritaca(streaming=True, callbacks=[streaming_callback])

            for chunk in model.stream("Tell me a story"):
                print(chunk.content, end="", flush=True)

            print(f"\\nTokens streamed: {streaming_callback.token_count}")
            print(f"Tokens per second: {streaming_callback.tokens_per_second:.1f}")
    """

    def __init__(self) -> None:
        """Initialize the token streaming callback."""
        super().__init__()
        self._start_time: float | None = None
        self._end_time: float | None = None
        self.token_count: int = 0
        self._tokens: list[str] = []

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record start time."""
        self._start_time = time.perf_counter()
        self._end_time = None
        self.token_count = 0
        self._tokens = []

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Count each new token."""
        self.token_count += 1
        self._tokens.append(token)
        self._end_time = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record end time."""
        self._end_time = time.perf_counter()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @property
    def tokens_per_second(self) -> float:
        """Get tokens per second rate."""
        if self.elapsed_time == 0:
            return 0.0
        return self.token_count / self.elapsed_time

    @property
    def tokens(self) -> list[str]:
        """Get all streamed tokens."""
        return self._tokens.copy()

    def reset(self) -> None:
        """Reset the callback state."""
        self._start_time = None
        self._end_time = None
        self.token_count = 0
        self._tokens = []


class CombinedCallback(BaseCallbackHandler):
    """Callback that combines cost and latency tracking.

    Convenience class that provides both cost and latency tracking
    in a single callback handler.

    Example:
        .. code-block:: python

            from langchain_maritaca import ChatMaritaca
            from langchain_maritaca.callbacks import CombinedCallback

            callback = CombinedCallback()
            model = ChatMaritaca(callbacks=[callback])

            model.invoke("Hello!")

            summary = callback.get_summary()
            print(f"Cost: ${summary['cost']['total_cost']:.6f}")
            print(f"Latency: {summary['latency']['average_latency']:.2f}s")
    """

    def __init__(self) -> None:
        """Initialize combined callback."""
        super().__init__()
        self._cost_callback = CostTrackingCallback()
        self._latency_callback = LatencyTrackingCallback()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Delegate to latency callback."""
        self._latency_callback.on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Delegate to both callbacks."""
        self._cost_callback.on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )
        self._latency_callback.on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Delegate to latency callback."""
        self._latency_callback.on_llm_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    @property
    def cost(self) -> CostTrackingCallback:
        """Access cost tracking callback."""
        return self._cost_callback

    @property
    def latency(self) -> LatencyTrackingCallback:
        """Access latency tracking callback."""
        return self._latency_callback

    def get_summary(self) -> dict[str, Any]:
        """Get combined summary of cost and latency.

        Returns:
            Dictionary with both cost and latency summaries.
        """
        return {
            "cost": self._cost_callback.get_summary(),
            "latency": self._latency_callback.get_summary(),
        }

    def reset(self) -> None:
        """Reset both callbacks."""
        self._cost_callback.reset()
        self._latency_callback.reset()
