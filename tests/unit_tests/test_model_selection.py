"""Tests for model selection helper features."""

from langchain_maritaca import ChatMaritaca
from langchain_maritaca.chat_models import MODEL_SPECS


class TestModelSpecs:
    """Tests for MODEL_SPECS constant."""

    def test_model_specs_defined(self) -> None:
        """Test that model specs are properly defined."""
        assert "sabia-3.1" in MODEL_SPECS
        assert "sabiazinho-3.1" in MODEL_SPECS

    def test_model_specs_has_required_fields(self) -> None:
        """Test that each model spec has all required fields."""
        required_fields = [
            "context_limit",
            "input_cost_per_1m",
            "output_cost_per_1m",
            "complexity",
            "speed",
            "capabilities",
            "description",
        ]

        for model_name, specs in MODEL_SPECS.items():
            for field in required_fields:
                assert field in specs, f"{model_name} missing {field}"

    def test_sabia_specs(self) -> None:
        """Test sabia-3.1 specifications."""
        specs = MODEL_SPECS["sabia-3.1"]
        assert specs["context_limit"] == 32768
        assert specs["input_cost_per_1m"] == 0.50
        assert specs["output_cost_per_1m"] == 1.50
        assert specs["complexity"] == "high"
        assert specs["speed"] == "medium"
        assert "complex_reasoning" in specs["capabilities"]

    def test_sabiazinho_specs(self) -> None:
        """Test sabiazinho-3.1 specifications."""
        specs = MODEL_SPECS["sabiazinho-3.1"]
        assert specs["context_limit"] == 8192
        assert specs["input_cost_per_1m"] == 0.10
        assert specs["output_cost_per_1m"] == 0.30
        assert specs["complexity"] == "medium"
        assert specs["speed"] == "fast"
        assert "simple_tasks" in specs["capabilities"]


class TestListAvailableModels:
    """Tests for list_available_models method."""

    def test_list_available_models_returns_dict(self) -> None:
        """Test that list_available_models returns a dictionary."""
        models = ChatMaritaca.list_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_list_available_models_contains_all_models(self) -> None:
        """Test that all defined models are returned."""
        models = ChatMaritaca.list_available_models()
        assert "sabia-3.1" in models
        assert "sabiazinho-3.1" in models

    def test_list_available_models_returns_copy(self) -> None:
        """Test that list_available_models returns a copy."""
        models1 = ChatMaritaca.list_available_models()
        models2 = ChatMaritaca.list_available_models()
        assert models1 is not models2


class TestRecommendModel:
    """Tests for recommend_model method."""

    def test_recommend_model_returns_dict(self) -> None:
        """Test that recommend_model returns a dictionary."""
        rec = ChatMaritaca.recommend_model()
        assert isinstance(rec, dict)
        assert "model" in rec
        assert "reason" in rec
        assert "specs" in rec
        assert "alternatives" in rec

    def test_recommend_model_simple_task_cost_priority(self) -> None:
        """Test recommendation for simple task with cost priority."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="simple",
            priority="cost",
        )
        # Should recommend sabiazinho for cost optimization
        assert rec["model"] == "sabiazinho-3.1"
        assert "cost" in rec["reason"].lower()

    def test_recommend_model_complex_task_quality_priority(self) -> None:
        """Test recommendation for complex task with quality priority."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="complex",
            priority="quality",
        )
        # Should recommend sabia for quality
        assert rec["model"] == "sabia-3.1"
        assert "quality" in rec["reason"].lower()

    def test_recommend_model_speed_priority(self) -> None:
        """Test recommendation with speed priority."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="medium",
            priority="speed",
        )
        # Should recommend sabiazinho for speed
        assert rec["model"] == "sabiazinho-3.1"
        assert "speed" in rec["reason"].lower()

    def test_recommend_model_with_input_length(self) -> None:
        """Test recommendation with input length constraint."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="simple",
            input_length=5000,
        )
        # Should accommodate the input length
        assert rec["specs"]["context_limit"] >= 5000

    def test_recommend_model_large_input_length(self) -> None:
        """Test recommendation with large input length requiring larger context."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="simple",
            input_length=10000,  # Exceeds sabiazinho's comfortable limit
        )
        # Should recommend sabia due to context requirements
        assert rec["model"] == "sabia-3.1"

    def test_recommend_model_extreme_input_length(self) -> None:
        """Test recommendation when input exceeds all model limits."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="simple",
            input_length=100000,  # Exceeds all models
        )
        # Should return sabia with truncation warning
        assert rec["model"] == "sabia-3.1"
        assert "truncation" in rec["reason"].lower()

    def test_recommend_model_includes_alternatives(self) -> None:
        """Test that alternatives are included in recommendation."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="medium",
            priority="quality",
        )
        # Should have at least one alternative
        assert isinstance(rec["alternatives"], list)

    def test_recommend_model_alternatives_have_specs(self) -> None:
        """Test that alternatives include specs."""
        rec = ChatMaritaca.recommend_model()
        for alt in rec["alternatives"]:
            assert "model" in alt
            assert "specs" in alt

    def test_recommend_model_reason_describes_selection(self) -> None:
        """Test that reason explains why model was selected."""
        rec = ChatMaritaca.recommend_model(
            task_complexity="simple",
            priority="cost",
        )
        # Reason should mention the optimization criteria
        assert len(rec["reason"]) > 20
        assert rec["specs"]["description"] in rec["reason"]

    def test_recommend_model_default_parameters(self) -> None:
        """Test recommendation with default parameters."""
        rec = ChatMaritaca.recommend_model()
        # Should work with defaults (medium complexity, quality priority)
        assert rec["model"] in MODEL_SPECS
        assert "general use" in rec["reason"].lower()
