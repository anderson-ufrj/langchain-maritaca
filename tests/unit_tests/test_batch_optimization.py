"""Tests for batch optimization features."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_maritaca import ChatMaritaca


@pytest.fixture
def mock_chat_result() -> ChatResult:
    """Create a mock ChatResult for testing."""
    from langchain_core.messages import AIMessage

    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content="Test response"),
                generation_info={"finish_reason": "stop"},
            )
        ],
        llm_output={"model": "sabia-3.1"},
    )


class TestAbatchWithConcurrency:
    """Tests for abatch_with_concurrency method."""

    @pytest.mark.asyncio
    async def test_abatch_with_concurrency_basic(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test basic batch processing."""
        model = ChatMaritaca(api_key="test")

        with patch.object(model, "_agenerate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_chat_result

            inputs = [
                [HumanMessage(content="Question 1")],
                [HumanMessage(content="Question 2")],
            ]

            results = await model.abatch_with_concurrency(inputs, max_concurrency=2)

            assert len(results) == 2
            assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_abatch_with_concurrency_respects_limit(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test that concurrency limit is respected."""
        model = ChatMaritaca(api_key="test")
        concurrent_calls = []
        max_concurrent = 0

        async def track_concurrency(*args, **kwargs):
            concurrent_calls.append(1)
            current = len(concurrent_calls)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.01)  # Small delay to allow overlap
            concurrent_calls.pop()
            return mock_chat_result

        with patch.object(model, "_agenerate", side_effect=track_concurrency):
            inputs = [[HumanMessage(content=f"Q{i}")] for i in range(10)]
            await model.abatch_with_concurrency(inputs, max_concurrency=3)

            # Max concurrent should not exceed limit
            assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_abatch_with_concurrency_return_exceptions(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test return_exceptions parameter."""
        model = ChatMaritaca(api_key="test")

        call_count = 0

        async def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Test error")
            return mock_chat_result

        with patch.object(model, "_agenerate", side_effect=sometimes_fail):
            inputs = [
                [HumanMessage(content="Q1")],
                [HumanMessage(content="Q2")],
                [HumanMessage(content="Q3")],
            ]

            results = await model.abatch_with_concurrency(
                inputs, max_concurrency=1, return_exceptions=True
            )

            assert len(results) == 3
            assert isinstance(results[0], ChatResult)
            assert isinstance(results[1], ValueError)
            assert isinstance(results[2], ChatResult)

    @pytest.mark.asyncio
    async def test_abatch_with_concurrency_empty_input(self) -> None:
        """Test batch with empty input list."""
        model = ChatMaritaca(api_key="test")
        results = await model.abatch_with_concurrency([], max_concurrency=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_abatch_with_concurrency_passes_stop(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test that stop sequences are passed correctly."""
        model = ChatMaritaca(api_key="test")

        with patch.object(model, "_agenerate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_chat_result

            inputs = [[HumanMessage(content="Question")]]
            await model.abatch_with_concurrency(inputs, max_concurrency=1, stop=["END"])

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["stop"] == ["END"]


class TestBatchWithProgress:
    """Tests for batch_with_progress method."""

    def test_batch_with_progress_basic(self, mock_chat_result: ChatResult) -> None:
        """Test basic batch with progress callback."""
        model = ChatMaritaca(api_key="test")
        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int, result: ChatResult | None) -> None:
            progress_calls.append((completed, total))

        with patch.object(model, "_agenerate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_chat_result

            inputs = [
                [HumanMessage(content="Q1")],
                [HumanMessage(content="Q2")],
                [HumanMessage(content="Q3")],
            ]

            results = model.batch_with_progress(
                inputs, max_concurrency=2, callback=on_progress
            )

            assert len(results) == 3
            assert len(progress_calls) == 3
            # Progress should be 1, 2, 3 (in some order due to async)
            completed_counts = [p[0] for p in progress_calls]
            assert set(completed_counts) == {1, 2, 3}
            # Total should always be 3
            assert all(p[1] == 3 for p in progress_calls)

    def test_batch_with_progress_no_callback(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test batch without progress callback."""
        model = ChatMaritaca(api_key="test")

        with patch.object(model, "_agenerate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_chat_result

            inputs = [
                [HumanMessage(content="Q1")],
                [HumanMessage(content="Q2")],
            ]

            results = model.batch_with_progress(inputs, max_concurrency=2)

            assert len(results) == 2

    def test_batch_with_progress_preserves_order(
        self, mock_chat_result: ChatResult
    ) -> None:
        """Test that results are returned in original order."""
        model = ChatMaritaca(api_key="test")

        async def ordered_response(messages, *args, **kwargs):
            # Extract question number from input to ensure correct mapping
            content = str(messages[0].content)
            q_num = content.replace("Q", "")
            # Delay inversely to mix up completion order
            await asyncio.sleep(0.01 * (4 - int(q_num)))
            from langchain_core.messages import AIMessage

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"Response {q_num}"),
                        generation_info={"finish_reason": "stop"},
                    )
                ],
                llm_output={"model": "sabia-3.1"},
            )

        with patch.object(model, "_agenerate", side_effect=ordered_response):
            inputs = [
                [HumanMessage(content="Q1")],
                [HumanMessage(content="Q2")],
                [HumanMessage(content="Q3")],
            ]

            results = model.batch_with_progress(inputs, max_concurrency=3)

            # Results should be in original order
            assert len(results) == 3
            contents = [r.generations[0].message.content for r in results]
            assert contents == ["Response 1", "Response 2", "Response 3"]


class TestAbatchEstimateCost:
    """Tests for abatch_estimate_cost method."""

    @pytest.mark.asyncio
    async def test_abatch_estimate_cost_basic(self) -> None:
        """Test basic batch cost estimation."""
        model = ChatMaritaca(api_key="test", model="sabia-3.1")

        inputs = [
            [HumanMessage(content="Short question")],
            [HumanMessage(content="Another question")],
        ]

        estimate = await model.abatch_estimate_cost(inputs, max_output_tokens=100)

        assert "total_requests" in estimate
        assert "total_input_tokens" in estimate
        assert "total_output_tokens" in estimate
        assert "total_input_cost" in estimate
        assert "total_output_cost" in estimate
        assert "total_cost" in estimate
        assert "model" in estimate

        assert estimate["total_requests"] == 2
        assert estimate["total_output_tokens"] == 200  # 100 * 2
        assert estimate["model"] == "sabia-3.1"

    @pytest.mark.asyncio
    async def test_abatch_estimate_cost_different_models(self) -> None:
        """Test cost estimation for different models."""
        model_sabia = ChatMaritaca(api_key="test", model="sabia-3.1")
        model_sabiazinho = ChatMaritaca(api_key="test", model="sabiazinho-3.1")

        inputs = [[HumanMessage(content="Test question")]]

        estimate_sabia = await model_sabia.abatch_estimate_cost(inputs)
        estimate_sabiazinho = await model_sabiazinho.abatch_estimate_cost(inputs)

        # sabiazinho should be cheaper
        assert estimate_sabiazinho["total_cost"] < estimate_sabia["total_cost"]

    @pytest.mark.asyncio
    async def test_abatch_estimate_cost_empty_input(self) -> None:
        """Test cost estimation with empty input."""
        model = ChatMaritaca(api_key="test")

        estimate = await model.abatch_estimate_cost([])

        assert estimate["total_requests"] == 0
        assert estimate["total_input_tokens"] == 0
        assert estimate["total_output_tokens"] == 0
        assert estimate["total_cost"] == 0

    @pytest.mark.asyncio
    async def test_abatch_estimate_cost_scales_with_tokens(self) -> None:
        """Test that cost scales with output tokens."""
        model = ChatMaritaca(api_key="test")

        inputs = [[HumanMessage(content="Test")]]

        estimate_100 = await model.abatch_estimate_cost(inputs, max_output_tokens=100)
        estimate_1000 = await model.abatch_estimate_cost(inputs, max_output_tokens=1000)

        # Cost should scale with output tokens
        assert estimate_1000["total_output_cost"] > estimate_100["total_output_cost"]
        assert estimate_1000["total_cost"] > estimate_100["total_cost"]
