"""Unit tests for ChatMaritaca.with_smart_fallbacks."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_maritaca import ChatMaritaca
from langchain_maritaca.chat_models import (
    _default_fallback_chain,
    _TransientHTTPStatusError,
)

if "MARITACA_API_KEY" not in os.environ:
    os.environ["MARITACA_API_KEY"] = "fake-key"


class TestDefaultFallbackChain:
    """Tests for the _default_fallback_chain helper."""

    def test_primary_sabia_31_falls_back_to_zinho_family(self) -> None:
        assert _default_fallback_chain("sabia-3.1") == [
            "sabiazinho-4",
            "sabiazinho-3.1",
        ]

    def test_primary_sabiazinho_4_falls_back_to_other_models(self) -> None:
        assert _default_fallback_chain("sabiazinho-4") == [
            "sabia-3.1",
            "sabiazinho-3.1",
        ]

    def test_primary_sabiazinho_31_falls_back_to_other_models(self) -> None:
        assert _default_fallback_chain("sabiazinho-3.1") == [
            "sabia-3.1",
            "sabiazinho-4",
        ]

    def test_unknown_primary_returns_empty_list(self) -> None:
        assert _default_fallback_chain("unknown-model") == []


class TestWithSmartFallbacksInstance:
    """Tests for the ChatMaritaca.with_smart_fallbacks() instance method."""

    def test_returns_runnable_with_default_fallbacks(self) -> None:
        model = ChatMaritaca(api_key="test-key", model="sabia-3.1")  # type: ignore[arg-type]
        chain = model.with_smart_fallbacks()

        # RunnableWithFallbacks exposes .fallbacks
        assert hasattr(chain, "fallbacks")
        fallback_models = [f.model_name for f in chain.fallbacks]
        assert fallback_models == ["sabiazinho-4", "sabiazinho-3.1"]

    def test_explicit_fallback_list_overrides_defaults(self) -> None:
        model = ChatMaritaca(api_key="test-key", model="sabia-3.1")  # type: ignore[arg-type]
        chain = model.with_smart_fallbacks(fallbacks=["sabiazinho-3.1"])
        fallback_models = [f.model_name for f in chain.fallbacks]
        assert fallback_models == ["sabiazinho-3.1"]

    def test_shared_kwargs_propagate_to_fallbacks(self) -> None:
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            temperature=0.42,
            max_tokens=123,
        )
        chain = model.with_smart_fallbacks()
        for f in chain.fallbacks:
            assert f.temperature == 0.42
            assert f.max_tokens == 123

    def test_unknown_fallback_model_raises_at_construction(self) -> None:
        model = ChatMaritaca(api_key="test-key", model="sabia-3.1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown model"):
            model.with_smart_fallbacks(fallbacks=["not-a-real-model"])

    def test_unknown_primary_with_no_explicit_fallbacks_raises(self) -> None:
        # model_name "sabia-4" is not in MODEL_SPECS; default chain is empty
        model = ChatMaritaca(api_key="test-key", model="sabia-4")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no default fallback chain"):
            model.with_smart_fallbacks()


class TestWithSmartFallbacksClassmethod:
    """Tests for ChatMaritaca.with_smart_fallbacks_from_primary(...)."""

    def test_classmethod_builds_primary_and_fallbacks(self) -> None:
        chain = ChatMaritaca.with_smart_fallbacks_from_primary(
            primary="sabia-3.1",
            api_key="test-key",
            temperature=0.3,
        )
        # RunnableWithFallbacks exposes the primary as .runnable
        assert chain.runnable.model_name == "sabia-3.1"
        assert chain.runnable.temperature == 0.3
        fallback_models = [f.model_name for f in chain.fallbacks]
        assert fallback_models == ["sabiazinho-4", "sabiazinho-3.1"]

    def test_classmethod_accepts_explicit_fallbacks(self) -> None:
        chain = ChatMaritaca.with_smart_fallbacks_from_primary(
            primary="sabia-3.1",
            fallbacks=["sabiazinho-3.1"],
            api_key="test-key",
        )
        assert [f.model_name for f in chain.fallbacks] == ["sabiazinho-3.1"]

    def test_classmethod_matches_instance_method_for_same_config(self) -> None:
        from_classmethod = ChatMaritaca.with_smart_fallbacks_from_primary(
            primary="sabia-3.1", api_key="test-key", temperature=0.7
        )
        from_instance = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            temperature=0.7,
        ).with_smart_fallbacks()
        assert [f.model_name for f in from_classmethod.fallbacks] == [
            f.model_name for f in from_instance.fallbacks
        ]


class TestFallbackExceptionFiltering:
    """End-to-end tests that fallback kicks in on transient errors only."""

    def _make_http_status_error(self, status_code: int) -> httpx.HTTPStatusError:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {}
        return httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_response,
        )

    def _success_payload(self, model: str, text: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    def _make_transient_error(self, status_code: int) -> _TransientHTTPStatusError:
        """Build the subclass that _make_request emits after rewrap."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {}
        return _TransientHTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_response,
        )

    def test_503_triggers_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        primary = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
        )
        chain = primary.with_smart_fallbacks(fallbacks=["sabiazinho-4"])
        error = self._make_transient_error(503)
        payload = self._success_payload("sabiazinho-4", "from fallback")
        call_count = {"n": 0}

        # Primary and fallback share the same ChatMaritaca class, so we use a
        # counter to simulate per-instance behavior: call 1 (primary) raises,
        # call 2 (fallback) succeeds. Two calls means fallback was attempted.
        def mock_make_request(self: Any, messages: Any, params: Any) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise error
            return payload

        monkeypatch.setattr(ChatMaritaca, "_make_request", mock_make_request)

        result = chain.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "from fallback"
        assert call_count["n"] == 2

    def test_429_triggers_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        primary = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
            retry_if_rate_limited=False,
        )
        chain = primary.with_smart_fallbacks(fallbacks=["sabiazinho-4"])
        error = self._make_transient_error(429)
        payload = self._success_payload("sabiazinho-4", "from fallback")
        call_count = {"n": 0}

        def mock_make_request(self: Any, messages: Any, params: Any) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise error
            return payload

        monkeypatch.setattr(ChatMaritaca, "_make_request", mock_make_request)

        result = chain.invoke([HumanMessage(content="hi")])
        assert result.content == "from fallback"
        assert call_count["n"] == 2

    def test_401_does_not_trigger_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        primary = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
        )
        chain = primary.with_smart_fallbacks(fallbacks=["sabiazinho-4"])
        error = self._make_http_status_error(401)
        call_count = {"n": 0}

        # Both primary and fallback share the same class, so we use a counter:
        # call 1 = primary invocation (raises 401), call 2+ = fallback invocation.
        # If fallback filtering works correctly, call 2 is never reached.
        def always_raises(self: Any, messages: Any, params: Any) -> dict[str, Any]:
            call_count["n"] += 1
            raise error

        monkeypatch.setattr(ChatMaritaca, "_make_request", always_raises)

        with pytest.raises(httpx.HTTPStatusError):
            chain.invoke([HumanMessage(content="hi")])
        assert call_count["n"] == 1, (
            f"Expected only 1 call (primary); got {call_count['n']} — "
            "fallback was incorrectly triggered for a 401 error"
        )

    def test_timeout_triggers_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        primary = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
        )
        chain = primary.with_smart_fallbacks(fallbacks=["sabiazinho-4"])
        payload = self._success_payload("sabiazinho-4", "from fallback")
        call_count = {"n": 0}

        def mock_make_request(self: Any, messages: Any, params: Any) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.TimeoutException("slow")
            return payload

        monkeypatch.setattr(ChatMaritaca, "_make_request", mock_make_request)

        result = chain.invoke([HumanMessage(content="hi")])
        assert result.content == "from fallback"
        assert call_count["n"] == 2


class TestTransientStatusRewrap:
    """Verify _make_request rewraps transient status codes to the subclass.

    Composed with TestFallbackExceptionFiltering, these tests prove the
    full path: HTTP 503 from the server → _make_request rewraps →
    _TransientHTTPStatusError → fallback filter fires.
    """

    def _http_response_mock(self, status_code: int) -> MagicMock:
        """Build a mock httpx.Response whose raise_for_status emits the canonical
        httpx.HTTPStatusError for the given status code."""
        response = MagicMock()
        response.status_code = status_code
        response.headers = {}
        response.request = MagicMock()

        def _raise() -> None:
            raise httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=response.request,
                response=response,
            )

        response.raise_for_status = _raise
        return response

    def test_503_is_rewrapped_to_transient_subclass(self) -> None:
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
        )
        model.client = MagicMock()
        model.client.post.return_value = self._http_response_mock(503)

        with pytest.raises(_TransientHTTPStatusError):
            model._make_request([], {})

    def test_502_and_504_also_rewrapped(self) -> None:
        for code in (502, 504):
            model = ChatMaritaca(
                api_key="test-key",  # type: ignore[arg-type]
                model="sabia-3.1",
                max_retries=0,
            )
            model.client = MagicMock()
            model.client.post.return_value = self._http_response_mock(code)
            with pytest.raises(_TransientHTTPStatusError):
                model._make_request([], {})

    def test_429_is_rewrapped_when_retry_disabled(self) -> None:
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
            retry_if_rate_limited=False,
        )
        model.client = MagicMock()
        model.client.post.return_value = self._http_response_mock(429)

        with pytest.raises(_TransientHTTPStatusError):
            model._make_request([], {})

    def test_401_is_not_rewrapped(self) -> None:
        """A non-transient status must propagate as the canonical class."""
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3.1",
            max_retries=0,
        )
        model.client = MagicMock()
        model.client.post.return_value = self._http_response_mock(401)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            model._make_request([], {})
        # Must be the base class, NOT the subclass
        assert not isinstance(exc_info.value, _TransientHTTPStatusError)
        assert exc_info.value.response.status_code == 401
