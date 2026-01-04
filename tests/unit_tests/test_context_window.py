"""Tests for context window management features."""

import warnings

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_maritaca import ChatMaritaca
from langchain_maritaca.chat_models import (
    DEFAULT_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
)


class TestContextWindowConstants:
    """Tests for context window constants."""

    def test_model_context_limits_defined(self) -> None:
        """Test that model context limits are properly defined."""
        assert "sabia-3.1" in MODEL_CONTEXT_LIMITS
        assert "sabiazinho-3.1" in MODEL_CONTEXT_LIMITS
        assert MODEL_CONTEXT_LIMITS["sabia-3.1"] == 32768
        assert MODEL_CONTEXT_LIMITS["sabiazinho-3.1"] == 8192

    def test_default_context_limit(self) -> None:
        """Test default context limit value."""
        assert DEFAULT_CONTEXT_LIMIT == 32768


class TestGetContextLimit:
    """Tests for get_context_limit method."""

    def test_get_context_limit_sabia(self) -> None:
        """Test context limit for sabia-3.1."""
        model = ChatMaritaca(model="sabia-3.1", api_key="test")
        assert model.get_context_limit() == 32768

    def test_get_context_limit_sabiazinho(self) -> None:
        """Test context limit for sabiazinho-3.1."""
        model = ChatMaritaca(model="sabiazinho-3.1", api_key="test")
        assert model.get_context_limit() == 8192

    def test_get_context_limit_custom(self) -> None:
        """Test custom context limit override."""
        model = ChatMaritaca(
            model="sabia-3.1",
            api_key="test",
            max_context_tokens=4096,
        )
        assert model.get_context_limit() == 4096

    def test_get_context_limit_unknown_model(self) -> None:
        """Test context limit for unknown model uses default."""
        model = ChatMaritaca(model="unknown-model", api_key="test")
        assert model.get_context_limit() == DEFAULT_CONTEXT_LIMIT


class TestCheckContextUsage:
    """Tests for check_context_usage method."""

    def test_check_context_usage_basic(self) -> None:
        """Test basic context usage check."""
        model = ChatMaritaca(api_key="test", max_context_tokens=1000)
        messages = [HumanMessage(content="Hello")]

        usage = model.check_context_usage(messages, warn=False)

        assert "tokens" in usage
        assert "limit" in usage
        assert "usage_percent" in usage
        assert "exceeds_limit" in usage
        assert "exceeds_threshold" in usage
        assert usage["limit"] == 1000
        assert usage["exceeds_limit"] is False

    def test_check_context_usage_exceeds_limit(self) -> None:
        """Test context usage when limit is exceeded."""
        model = ChatMaritaca(api_key="test", max_context_tokens=10)
        messages = [HumanMessage(content="This is a longer message that exceeds limit")]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            usage = model.check_context_usage(messages, warn=True)

            assert usage["exceeds_limit"] is True
            assert len(w) >= 1
            assert "exceeded" in str(w[-1].message).lower()

    def test_check_context_usage_threshold_warning(self) -> None:
        """Test warning when approaching context limit."""
        model = ChatMaritaca(
            api_key="test",
            max_context_tokens=100,
            context_warning_threshold=0.5,
        )
        # Create message that uses more than 50% of context
        messages = [HumanMessage(content="x" * 200)]  # ~50 tokens

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            usage = model.check_context_usage(messages, warn=True)

            # Should trigger threshold warning
            assert usage["exceeds_threshold"] is True

    def test_check_context_usage_no_warn(self) -> None:
        """Test that warn=False suppresses warnings."""
        model = ChatMaritaca(api_key="test", max_context_tokens=10)
        messages = [HumanMessage(content="Long message that exceeds limit for sure")]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.check_context_usage(messages, warn=False)

            # No warnings should be emitted
            context_warnings = [x for x in w if "context" in str(x.message).lower()]
            assert len(context_warnings) == 0


class TestTruncateMessages:
    """Tests for truncate_messages method."""

    def test_truncate_empty_messages(self) -> None:
        """Test truncating empty message list."""
        model = ChatMaritaca(api_key="test")
        result = model.truncate_messages([])
        assert result == []

    def test_truncate_within_limit(self) -> None:
        """Test that messages within limit are not modified."""
        model = ChatMaritaca(api_key="test", max_context_tokens=10000)
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]

        result = model.truncate_messages(messages)

        assert len(result) == 2
        assert result[0].content == "You are helpful."
        assert result[1].content == "Hello"

    def test_truncate_preserves_system_message(self) -> None:
        """Test that system messages are preserved during truncation."""
        model = ChatMaritaca(api_key="test", max_context_tokens=100)
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Most recent"),
        ]

        result = model.truncate_messages(messages, preserve_recent=1)

        # System message should be first
        assert isinstance(result[0], SystemMessage)
        # Most recent should be last
        assert result[-1].content == "Most recent"

    def test_truncate_preserves_recent_messages(self) -> None:
        """Test that recent messages are preserved."""
        model = ChatMaritaca(api_key="test", max_context_tokens=200)
        messages = [
            HumanMessage(content="Old message 1"),
            AIMessage(content="Old response 1"),
            HumanMessage(content="Recent message"),
            AIMessage(content="Recent response"),
        ]

        result = model.truncate_messages(messages, preserve_recent=2)

        # Last 2 non-system messages should be preserved
        assert len(result) >= 2
        assert result[-1].content == "Recent response"
        assert result[-2].content == "Recent message"

    def test_truncate_without_preserve_system(self) -> None:
        """Test truncation without preserving system messages."""
        model = ChatMaritaca(api_key="test", max_context_tokens=50)
        messages = [
            SystemMessage(content="Long system prompt that takes tokens"),
            HumanMessage(content="Recent"),
        ]

        result = model.truncate_messages(messages, preserve_system=False)

        # System message may be truncated
        assert len(result) >= 1
        assert result[-1].content == "Recent"

    def test_truncate_returns_copy(self) -> None:
        """Test that truncate returns a copy, not original list."""
        model = ChatMaritaca(api_key="test", max_context_tokens=10000)
        messages = [HumanMessage(content="Hello")]

        result = model.truncate_messages(messages)

        assert result is not messages
        assert result == messages


class TestContextWindowParameters:
    """Tests for context window parameters."""

    def test_max_context_tokens_parameter(self) -> None:
        """Test max_context_tokens parameter is set correctly."""
        model = ChatMaritaca(api_key="test", max_context_tokens=4096)
        assert model.max_context_tokens == 4096

    def test_auto_truncate_parameter(self) -> None:
        """Test auto_truncate parameter is set correctly."""
        model = ChatMaritaca(api_key="test", auto_truncate=True)
        assert model.auto_truncate is True

    def test_context_warning_threshold_parameter(self) -> None:
        """Test context_warning_threshold parameter is set correctly."""
        model = ChatMaritaca(api_key="test", context_warning_threshold=0.8)
        assert model.context_warning_threshold == 0.8

    def test_default_context_warning_threshold(self) -> None:
        """Test default context warning threshold."""
        model = ChatMaritaca(api_key="test")
        assert model.context_warning_threshold == 0.9
