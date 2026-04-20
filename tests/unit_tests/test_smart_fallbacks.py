"""Unit tests for ChatMaritaca.with_smart_fallbacks."""

from __future__ import annotations

import os

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
