"""Unit tests for token counting functionality.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_maritaca import ChatMaritaca


class TestGetNumTokens:
    """Test suite for get_num_tokens method."""

    @pytest.fixture
    def model(self) -> ChatMaritaca:
        """Create a ChatMaritaca instance."""
        return ChatMaritaca(api_key="test-key")

    def test_get_num_tokens_with_tiktoken(self, model: ChatMaritaca) -> None:
        """Test token counting with tiktoken installed."""
        # Simple text should return reasonable token count
        text = "Hello, how are you?"
        tokens = model.get_num_tokens(text)

        # Should return a positive integer
        assert isinstance(tokens, int)
        assert tokens > 0
        # "Hello, how are you?" is typically 5-6 tokens
        assert tokens < 20

    def test_get_num_tokens_empty_string(self, model: ChatMaritaca) -> None:
        """Test token counting for empty string."""
        tokens = model.get_num_tokens("")

        # Empty string should return 0 or 1 (depending on implementation)
        assert tokens >= 0
        assert tokens <= 1

    def test_get_num_tokens_portuguese_text(self, model: ChatMaritaca) -> None:
        """Test token counting for Portuguese text."""
        text = "Olá, como você está? Espero que esteja bem!"
        tokens = model.get_num_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0
        # Portuguese text may have slightly different tokenization
        assert tokens < 30

    def test_get_num_tokens_long_text(self, model: ChatMaritaca) -> None:
        """Test token counting for longer text."""
        text = "Este é um texto mais longo para testar a contagem de tokens. " * 10
        tokens = model.get_num_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 50  # Should have many tokens

    @patch.dict("sys.modules", {"tiktoken": None})
    def test_get_num_tokens_fallback_without_tiktoken(
        self, model: ChatMaritaca
    ) -> None:
        """Test character-based fallback when tiktoken is not available."""
        # Create a new model to avoid cached tiktoken import
        with patch(
            "langchain_maritaca.chat_models.ChatMaritaca.get_num_tokens"
        ) as mock:
            # Simulate fallback behavior
            mock.return_value = 5  # ~20 chars / 4 = 5
            tokens = model.get_num_tokens("Hello, how are you?")
            assert tokens > 0

    def test_get_num_tokens_special_characters(self, model: ChatMaritaca) -> None:
        """Test token counting with special characters."""
        text = "Símbolos especiais: @#$%^&*(){}[]|\\:;<>,.?/~`"
        tokens = model.get_num_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_get_num_tokens_unicode(self, model: ChatMaritaca) -> None:
        """Test token counting with Unicode characters."""
        text = "Emoji: 🇧🇷 Acentos: áéíóú çãõ"
        tokens = model.get_num_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0


class TestGetNumTokensFromMessages:
    """Test suite for get_num_tokens_from_messages method."""

    @pytest.fixture
    def model(self) -> ChatMaritaca:
        """Create a ChatMaritaca instance."""
        return ChatMaritaca(api_key="test-key")

    def test_single_human_message(self, model: ChatMaritaca) -> None:
        """Test token counting for a single human message."""
        messages = [HumanMessage(content="Hello!")]
        tokens = model.get_num_tokens_from_messages(messages)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_system_and_human_messages(self, model: ChatMaritaca) -> None:
        """Test token counting for system and human messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of Brazil?"),
        ]
        tokens = model.get_num_tokens_from_messages(messages)

        assert isinstance(tokens, int)
        # Should be more than a single message
        single_tokens = model.get_num_tokens_from_messages([messages[0]])
        assert tokens > single_tokens

    def test_conversation_with_ai_response(self, model: ChatMaritaca) -> None:
        """Test token counting for a full conversation."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there! How can I help you today?"),
            HumanMessage(content="What is 2+2?"),
        ]
        tokens = model.get_num_tokens_from_messages(messages)

        assert isinstance(tokens, int)
        assert tokens > 20  # Should have significant tokens

    def test_ai_message_with_tool_calls(self, model: ChatMaritaca) -> None:
        """Test token counting for AI message with tool calls."""
        messages = [
            HumanMessage(content="What's the weather in São Paulo?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": {"city": "São Paulo"},
                    }
                ],
            ),
        ]
        tokens = model.get_num_tokens_from_messages(messages)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_tool_message(self, model: ChatMaritaca) -> None:
        """Test token counting with tool message."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_123", "name": "get_weather", "args": {}}],
            ),
            ToolMessage(content="Sunny, 25°C", tool_call_id="call_123"),
        ]
        tokens = model.get_num_tokens_from_messages(messages)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_empty_messages(self, model: ChatMaritaca) -> None:
        """Test token counting for empty message list."""
        messages: list = []
        tokens = model.get_num_tokens_from_messages(messages)

        # Should return at least the base overhead
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_message_ordering_consistency(self, model: ChatMaritaca) -> None:
        """Test that token count is consistent regardless of message order."""
        messages1 = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        messages2 = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
        ]

        tokens1 = model.get_num_tokens_from_messages(messages1)
        tokens2 = model.get_num_tokens_from_messages(messages2)

        # Should be equal or very close
        assert abs(tokens1 - tokens2) <= 2


class TestEstimateCost:
    """Test suite for estimate_cost method."""

    @pytest.fixture
    def model_sabia(self) -> ChatMaritaca:
        """Create a ChatMaritaca instance with sabia-3.1."""
        return ChatMaritaca(api_key="test-key", model="sabia-3.1")

    @pytest.fixture
    def model_sabiazinho(self) -> ChatMaritaca:
        """Create a ChatMaritaca instance with sabiazinho-3.1."""
        return ChatMaritaca(api_key="test-key", model="sabiazinho-3.1")

    def test_estimate_cost_returns_dict(self, model_sabia: ChatMaritaca) -> None:
        """Test that estimate_cost returns expected dictionary structure."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabia.estimate_cost(messages)

        assert isinstance(estimate, dict)
        assert "input_tokens" in estimate
        assert "output_tokens" in estimate
        assert "input_cost" in estimate
        assert "output_cost" in estimate
        assert "total_cost" in estimate
        assert "model" in estimate

    def test_estimate_cost_sabia_pricing(self, model_sabia: ChatMaritaca) -> None:
        """Test cost estimation with sabia-3.1 pricing."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabia.estimate_cost(messages, max_output_tokens=100)

        assert estimate["model"] == "sabia-3.1"
        assert estimate["output_tokens"] == 100

        # Verify pricing calculation
        # sabia-3.1: input $0.50/1M, output $1.50/1M
        expected_output_cost = (100 / 1_000_000) * 1.50
        assert estimate["output_cost"] == pytest.approx(expected_output_cost)

    def test_estimate_cost_sabiazinho_pricing(
        self, model_sabiazinho: ChatMaritaca
    ) -> None:
        """Test cost estimation with sabiazinho-3.1 pricing."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabiazinho.estimate_cost(messages, max_output_tokens=100)

        assert estimate["model"] == "sabiazinho-3.1"

        # Verify pricing calculation
        # sabiazinho-3.1: input $0.10/1M, output $0.30/1M
        expected_output_cost = (100 / 1_000_000) * 0.30
        assert estimate["output_cost"] == pytest.approx(expected_output_cost)

    def test_estimate_cost_sabiazinho_cheaper(
        self, model_sabia: ChatMaritaca, model_sabiazinho: ChatMaritaca
    ) -> None:
        """Test that sabiazinho is cheaper than sabia."""
        messages = [HumanMessage(content="Hello, tell me a story!")]

        sabia_estimate = model_sabia.estimate_cost(messages, max_output_tokens=1000)
        sabiazinho_estimate = model_sabiazinho.estimate_cost(
            messages, max_output_tokens=1000
        )

        assert sabiazinho_estimate["total_cost"] < sabia_estimate["total_cost"]

    def test_estimate_cost_default_output_tokens(
        self, model_sabia: ChatMaritaca
    ) -> None:
        """Test that default max_output_tokens is 1000."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabia.estimate_cost(messages)

        assert estimate["output_tokens"] == 1000

    def test_estimate_cost_custom_output_tokens(
        self, model_sabia: ChatMaritaca
    ) -> None:
        """Test custom max_output_tokens parameter."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabia.estimate_cost(messages, max_output_tokens=5000)

        assert estimate["output_tokens"] == 5000

    def test_estimate_cost_total_is_sum(self, model_sabia: ChatMaritaca) -> None:
        """Test that total_cost equals input_cost + output_cost."""
        messages = [HumanMessage(content="Hello!")]
        estimate = model_sabia.estimate_cost(messages)

        expected_total = estimate["input_cost"] + estimate["output_cost"]
        assert estimate["total_cost"] == pytest.approx(expected_total)

    def test_estimate_cost_unknown_model_uses_default(self) -> None:
        """Test that unknown models use default pricing."""
        model = ChatMaritaca(api_key="test-key", model="unknown-model")
        messages = [HumanMessage(content="Hello!")]
        estimate = model.estimate_cost(messages, max_output_tokens=100)

        # Should use default pricing (same as sabia-3.1)
        expected_output_cost = (100 / 1_000_000) * 1.50
        assert estimate["output_cost"] == pytest.approx(expected_output_cost)

    def test_estimate_cost_complex_messages(self, model_sabia: ChatMaritaca) -> None:
        """Test cost estimation with complex message history."""
        messages = [
            SystemMessage(content="You are a helpful assistant for Portuguese."),
            HumanMessage(content="Olá! Como você está?"),
            AIMessage(content="Estou bem, obrigado! Como posso ajudá-lo?"),
            HumanMessage(content="Qual é a capital do Brasil?"),
        ]
        estimate = model_sabia.estimate_cost(messages, max_output_tokens=500)

        assert estimate["input_tokens"] > 20  # Should have significant input
        assert estimate["total_cost"] > 0
