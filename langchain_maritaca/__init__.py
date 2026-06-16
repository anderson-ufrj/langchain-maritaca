"""Maritaca AI integration for LangChain.

Maritaca AI provides Brazilian Portuguese-optimized language models.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ntlabs
"""

from langchain_maritaca.cache import MaritacaSemanticCache
from langchain_maritaca.callbacks import (
    CombinedCallback,
    CostTrackingCallback,
    LatencyTrackingCallback,
    TokenStreamingCallback,
)
from langchain_maritaca.chat_models import ChatMaritaca
from langchain_maritaca.embeddings import DeepInfraEmbeddings
from langchain_maritaca.version import __version__

__all__ = [
    "ChatMaritaca",
    "CombinedCallback",
    "CostTrackingCallback",
    "DeepInfraEmbeddings",
    "LatencyTrackingCallback",
    "MaritacaSemanticCache",
    "TokenStreamingCallback",
    "__version__",
]
