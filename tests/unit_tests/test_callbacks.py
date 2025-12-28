"""Unit tests for callback handlers.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_maritaca.callbacks import (
    CombinedCallback,
    CostTrackingCallback,
    LatencyTrackingCallback,
    TokenStreamingCallback,
)


class TestCostTrackingCallback:
    """Test suite for CostTrackingCallback."""

    @pytest.fixture
    def callback(self) -> CostTrackingCallback:
        """Create a cost tracking callback."""
        return CostTrackingCallback()

    @pytest.fixture
    def mock_llm_result(self) -> LLMResult:
        """Create a mock LLM result with token usage."""
        return LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Hello!"))]],
            llm_output={
                "model": "sabia-3.1",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

    def test_initial_state(self, callback: CostTrackingCallback) -> None:
        """Test initial callback state."""
        assert callback.total_cost == 0.0
        assert callback.total_tokens == 0
        assert callback.total_input_tokens == 0
        assert callback.total_output_tokens == 0
        assert callback.call_count == 0

    def test_on_llm_end_tracks_cost(
        self, callback: CostTrackingCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test that on_llm_end tracks cost correctly."""
        run_id = uuid4()
        callback.on_llm_end(mock_llm_result, run_id=run_id)

        assert callback.total_input_tokens == 100
        assert callback.total_output_tokens == 50
        assert callback.total_tokens == 150
        assert callback.call_count == 1

        # Calculate expected cost for sabia-3.1
        # Input: 100 tokens * $0.50/1M = $0.00005
        # Output: 50 tokens * $1.50/1M = $0.000075
        expected_cost = (100 / 1_000_000) * 0.50 + (50 / 1_000_000) * 1.50
        assert callback.total_cost == pytest.approx(expected_cost)

    def test_on_llm_end_accumulates(
        self, callback: CostTrackingCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test that multiple calls accumulate costs."""
        callback.on_llm_end(mock_llm_result, run_id=uuid4())
        callback.on_llm_end(mock_llm_result, run_id=uuid4())

        assert callback.total_input_tokens == 200
        assert callback.total_output_tokens == 100
        assert callback.total_tokens == 300
        assert callback.call_count == 2

    def test_get_summary(
        self, callback: CostTrackingCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test get_summary method."""
        callback.on_llm_end(mock_llm_result, run_id=uuid4())

        summary = callback.get_summary()

        assert "total_cost" in summary
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "call_count" in summary
        assert "average_cost_per_call" in summary
        assert "calls" in summary
        assert len(summary["calls"]) == 1

    def test_reset(
        self, callback: CostTrackingCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test reset method."""
        callback.on_llm_end(mock_llm_result, run_id=uuid4())
        callback.reset()

        assert callback.total_cost == 0.0
        assert callback.total_tokens == 0
        assert callback.call_count == 0

    def test_default_pricing_for_unknown_model(
        self, callback: CostTrackingCallback
    ) -> None:
        """Test that unknown models use default pricing."""
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Hello!"))]],
            llm_output={
                "model": "unknown-model",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )
        callback.on_llm_end(result, run_id=uuid4())

        # Should use default pricing
        expected_cost = (100 / 1_000_000) * 0.50 + (50 / 1_000_000) * 1.50
        assert callback.total_cost == pytest.approx(expected_cost)

    def test_sabiazinho_pricing(self, callback: CostTrackingCallback) -> None:
        """Test pricing for sabiazinho-3.1 model."""
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Hello!"))]],
            llm_output={
                "model": "sabiazinho-3.1",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )
        callback.on_llm_end(result, run_id=uuid4())

        # sabiazinho pricing: input $0.10/1M, output $0.30/1M
        expected_cost = (100 / 1_000_000) * 0.10 + (50 / 1_000_000) * 0.30
        assert callback.total_cost == pytest.approx(expected_cost)


class TestLatencyTrackingCallback:
    """Test suite for LatencyTrackingCallback."""

    @pytest.fixture
    def callback(self) -> LatencyTrackingCallback:
        """Create a latency tracking callback."""
        return LatencyTrackingCallback()

    @pytest.fixture
    def mock_llm_result(self) -> LLMResult:
        """Create a mock LLM result."""
        return LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Hello!"))]]
        )

    def test_initial_state(self, callback: LatencyTrackingCallback) -> None:
        """Test initial callback state."""
        assert callback.call_count == 0
        assert callback.average_latency == 0.0
        assert callback.latencies == []

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_tracks_latency(
        self,
        mock_time: MagicMock,
        callback: LatencyTrackingCallback,
        mock_llm_result: LLMResult,
    ) -> None:
        """Test that latency is tracked correctly."""
        mock_time.side_effect = [0.0, 1.5]  # Start at 0, end at 1.5s

        run_id = uuid4()
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_end(mock_llm_result, run_id=run_id)

        assert callback.call_count == 1
        assert callback.latencies == [1.5]
        assert callback.average_latency == 1.5

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_percentiles(
        self,
        mock_time: MagicMock,
        callback: LatencyTrackingCallback,
        mock_llm_result: LLMResult,
    ) -> None:
        """Test percentile calculations."""
        # Simulate 10 calls with different latencies
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        time_values = []
        for lat in latencies:
            time_values.extend([0.0, lat])
        mock_time.side_effect = time_values

        for _ in range(10):
            run_id = uuid4()
            callback.on_llm_start({}, ["prompt"], run_id=run_id)
            callback.on_llm_end(mock_llm_result, run_id=run_id)

        assert callback.call_count == 10
        assert callback.min_latency == pytest.approx(0.1)
        assert callback.max_latency == pytest.approx(1.0)
        # P50 at index 5 (0-indexed) = 0.6 for our implementation
        assert callback.p50_latency == pytest.approx(0.6)
        assert callback.p95_latency == pytest.approx(1.0)

    def test_get_summary(self, callback: LatencyTrackingCallback) -> None:
        """Test get_summary method."""
        summary = callback.get_summary()

        assert "call_count" in summary
        assert "total_latency" in summary
        assert "average_latency" in summary
        assert "min_latency" in summary
        assert "max_latency" in summary
        assert "p50_latency" in summary
        assert "p95_latency" in summary
        assert "p99_latency" in summary

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_on_llm_error_tracks_latency(
        self, mock_time: MagicMock, callback: LatencyTrackingCallback
    ) -> None:
        """Test that latency is tracked even on error."""
        mock_time.side_effect = [0.0, 0.5]

        run_id = uuid4()
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_error(Exception("Test error"), run_id=run_id)

        assert callback.call_count == 1
        assert callback.latencies == [0.5]

    def test_reset(self, callback: LatencyTrackingCallback) -> None:
        """Test reset method."""
        callback._latencies = [1.0, 2.0]
        callback.call_count = 2
        callback.reset()

        assert callback.call_count == 0
        assert callback.latencies == []


class TestTokenStreamingCallback:
    """Test suite for TokenStreamingCallback."""

    @pytest.fixture
    def callback(self) -> TokenStreamingCallback:
        """Create a token streaming callback."""
        return TokenStreamingCallback()

    def test_initial_state(self, callback: TokenStreamingCallback) -> None:
        """Test initial callback state."""
        assert callback.token_count == 0
        assert callback.tokens == []

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_counts_tokens(
        self, mock_time: MagicMock, callback: TokenStreamingCallback
    ) -> None:
        """Test that tokens are counted correctly."""
        mock_time.return_value = 0.0

        run_id = uuid4()
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_new_token("Hello", run_id=run_id)
        callback.on_llm_new_token(" ", run_id=run_id)
        callback.on_llm_new_token("World", run_id=run_id)

        assert callback.token_count == 3
        assert callback.tokens == ["Hello", " ", "World"]

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_tokens_per_second(
        self, mock_time: MagicMock, callback: TokenStreamingCallback
    ) -> None:
        """Test tokens per second calculation."""
        mock_time.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0]

        run_id = uuid4()
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_new_token("a", run_id=run_id)
        callback.on_llm_new_token("b", run_id=run_id)
        callback.on_llm_new_token("c", run_id=run_id)
        callback.on_llm_new_token("d", run_id=run_id)

        # 4 tokens in 2 seconds = 2 tokens/second
        assert callback.tokens_per_second == pytest.approx(2.0)

    def test_reset(self, callback: TokenStreamingCallback) -> None:
        """Test reset method."""
        callback.token_count = 5
        callback._tokens = ["a", "b", "c", "d", "e"]
        callback.reset()

        assert callback.token_count == 0
        assert callback.tokens == []


class TestCombinedCallback:
    """Test suite for CombinedCallback."""

    @pytest.fixture
    def callback(self) -> CombinedCallback:
        """Create a combined callback."""
        return CombinedCallback()

    @pytest.fixture
    def mock_llm_result(self) -> LLMResult:
        """Create a mock LLM result with token usage."""
        return LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Hello!"))]],
            llm_output={
                "model": "sabia-3.1",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

    @patch("langchain_maritaca.callbacks.time.perf_counter")
    def test_tracks_both_cost_and_latency(
        self,
        mock_time: MagicMock,
        callback: CombinedCallback,
        mock_llm_result: LLMResult,
    ) -> None:
        """Test that both cost and latency are tracked."""
        mock_time.side_effect = [0.0, 1.0]

        run_id = uuid4()
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_end(mock_llm_result, run_id=run_id)

        assert callback.cost.call_count == 1
        assert callback.cost.total_tokens == 150
        assert callback.latency.call_count == 1
        assert callback.latency.latencies == [1.0]

    def test_get_summary(
        self, callback: CombinedCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test get_summary method."""
        callback.on_llm_end(mock_llm_result, run_id=uuid4())

        summary = callback.get_summary()

        assert "cost" in summary
        assert "latency" in summary
        assert "total_cost" in summary["cost"]
        assert "average_latency" in summary["latency"]

    def test_reset(
        self, callback: CombinedCallback, mock_llm_result: LLMResult
    ) -> None:
        """Test reset method."""
        callback.on_llm_end(mock_llm_result, run_id=uuid4())
        callback.reset()

        assert callback.cost.call_count == 0
        assert callback.latency.call_count == 0


class TestCallbackImports:
    """Test that callbacks can be imported from main package."""

    def test_import_from_package(self) -> None:
        """Test importing callbacks from langchain_maritaca."""
        from langchain_maritaca import (
            CombinedCallback,
            CostTrackingCallback,
            LatencyTrackingCallback,
            TokenStreamingCallback,
        )

        assert CostTrackingCallback is not None
        assert LatencyTrackingCallback is not None
        assert TokenStreamingCallback is not None
        assert CombinedCallback is not None
