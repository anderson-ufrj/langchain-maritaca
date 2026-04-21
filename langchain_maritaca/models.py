"""Single source of truth for Maritaca AI model metadata.

All model names, context windows, pricing, capabilities, and canonical
fallback ordering live here. Everything else in the package derives from
this module, so updating the Maritaca lineup is a one-file edit.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MaritacaModelSpec:
    """Immutable descriptor for a Maritaca AI model.

    Attributes:
        name: API model identifier (what you pass as ``model=``).
        context_limit: Maximum input context in tokens.
        input_cost_per_1m_brl: Price in BRL per 1M input tokens.
        output_cost_per_1m_brl: Price in BRL per 1M output tokens.
        max_output_tokens: Maximum tokens the API will emit per response.
        capabilities: Feature tags (``tool_calling``, ``vision``, etc.).
        description: Short human-readable blurb.
        fallback_tier: Ordering hint for the canonical fallback chain.
            ``1`` = preferred primary; larger numbers are fallback candidates
            in ascending order. Tiers need not be contiguous.
        active: ``False`` means the model is no longer offered by the API and
            should be excluded from every derived structure (default chain,
            ``list_available_models``, context limits).
    """

    name: str
    context_limit: int
    input_cost_per_1m_brl: float
    output_cost_per_1m_brl: float
    max_output_tokens: int = 12000
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    fallback_tier: int = 99
    active: bool = True


# The single source of truth. Edit this dict to add, deprecate, or reorder
# Maritaca models; every derived structure below picks the change up.
_MODELS: dict[str, MaritacaModelSpec] = {
    "sabia-4": MaritacaModelSpec(
        name="sabia-4",
        context_limit=128000,
        input_cost_per_1m_brl=5.00,
        output_cost_per_1m_brl=20.00,
        capabilities=(
            "complex_reasoning",
            "long_context",
            "tool_calling",
            "vision",
        ),
        description=(
            "Flagship Sabiá 4 model: best quality, 128k context, vision. "
            "Use for complex reasoning and production-critical tasks."
        ),
        fallback_tier=1,
    ),
    "sabiazinho-4": MaritacaModelSpec(
        name="sabiazinho-4",
        context_limit=128000,
        input_cost_per_1m_brl=1.00,
        output_cost_per_1m_brl=4.00,
        capabilities=(
            "simple_tasks",
            "quick_responses",
            "tool_calling",
            "vision",
        ),
        description=(
            "Fast and economical Sabiá 4 variant: 128k context, vision, "
            "5x cheaper output than sabia-4. Ideal for high-throughput tasks."
        ),
        fallback_tier=2,
    ),
}


def get_model(name: str) -> MaritacaModelSpec:
    """Return the spec for ``name``. Raises ``KeyError`` if unknown or inactive.

    Inactive models are treated as unknown to callers — they should not appear
    in fallback chains or recommendation output.
    """
    spec = _MODELS.get(name)
    if spec is None or not spec.active:
        raise KeyError(name)
    return spec


def active_models() -> dict[str, MaritacaModelSpec]:
    """Return only currently-offered models, keyed by name."""
    return {name: spec for name, spec in _MODELS.items() if spec.active}


def default_primary() -> str:
    """Return the name of the active model with the lowest ``fallback_tier``.

    This is the value used as the default for ``ChatMaritaca.model_name``.
    """
    actives = active_models()
    if not actives:
        msg = "No active Maritaca models configured in SoT."
        raise RuntimeError(msg)
    return min(actives, key=lambda k: actives[k].fallback_tier)


def canonical_fallback_order() -> tuple[str, ...]:
    """Active model names sorted by ``fallback_tier`` ascending.

    ``with_smart_fallbacks`` uses this as the default fallback candidate list
    (excluding the current primary).
    """
    actives = active_models()
    return tuple(sorted(actives, key=lambda k: actives[k].fallback_tier))


def as_legacy_specs() -> dict[str, dict[str, object]]:
    """Return the legacy ``MODEL_SPECS`` shape used by older helpers.

    Kept as a derivation so existing APIs (``list_available_models``,
    ``recommend_model``) continue to work while code migrates to the typed
    ``MaritacaModelSpec`` interface.
    """
    specs: dict[str, dict[str, object]] = {}
    for name, spec in active_models().items():
        if spec.output_cost_per_1m_brl >= 10.0:
            complexity = "high"
            speed = "medium"
        else:
            complexity = "medium"
            speed = "fast"
        specs[name] = {
            "context_limit": spec.context_limit,
            "input_cost_per_1m": spec.input_cost_per_1m_brl,
            "output_cost_per_1m": spec.output_cost_per_1m_brl,
            "complexity": complexity,
            "speed": speed,
            "capabilities": list(spec.capabilities),
            "description": spec.description,
        }
    return specs


def context_limits() -> dict[str, int]:
    """Return the legacy ``MODEL_CONTEXT_LIMITS`` shape."""
    return {name: spec.context_limit for name, spec in active_models().items()}
