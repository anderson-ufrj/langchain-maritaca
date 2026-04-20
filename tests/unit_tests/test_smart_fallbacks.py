"""Unit tests for ChatMaritaca.with_smart_fallbacks."""

from __future__ import annotations

import os

import pytest

from langchain_maritaca import ChatMaritaca
from langchain_maritaca.chat_models import _default_fallback_chain

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
