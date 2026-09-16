"""Tests for Keras inference logic and BeamSampler configuration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.keras.inference as inf
from gemma_4_sql.backends.keras.inference import (
    compute_keras_confidence,
    configure_beam_sampler,
    generate_sql,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_compute_keras_confidence_all_branches() -> None:
    """Test compute_keras_confidence across various score formats."""
    # Zero tokens
    assert compute_keras_confidence([0.1], 0) == 0.0

    # Negative log probabilities
    log_probs = [-0.1, -0.2]
    res_log = compute_keras_confidence(log_probs, 2)
    assert 0.0 < res_log <= 1.0

    # Linear probabilities (positive)
    lin_probs = [0.8, 0.9]
    res_lin = compute_keras_confidence(lin_probs, 2)
    assert res_lin == pytest.approx(0.85)

    # Empty nested list
    assert compute_keras_confidence([[]], 2) == 0.5

    # Non-number in container
    assert compute_keras_confidence(["invalid_item"], 2) == 0.5

    # Negative scalar
    res_scalar_neg = compute_keras_confidence(-0.4, 2)
    assert 0.0 < res_scalar_neg <= 1.0

    # Positive scalar
    res_scalar_pos = compute_keras_confidence(0.92, 2)
    assert res_scalar_pos == pytest.approx(0.92)

    # None scores fallback
    res_fallback = compute_keras_confidence(None, 5)
    assert 0.1 <= res_fallback <= 0.95

    # Object with numpy / tolist methods
    class MockArray:
        def numpy(self) -> MockArray:
            return self

        def tolist(self) -> list[float]:
            return [-0.05, -0.05]

    assert compute_keras_confidence(MockArray(), 2) > 0.9


def test_configure_beam_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test configuring BeamSampler on model."""
    mock_sampler_cls = MagicMock()
    mock_keras_nlp = type("MockNLP", (), {"samplers": type("Samplers", (), {"BeamSampler": mock_sampler_cls})})
    monkeypatch.setitem(sys.modules, "keras_nlp", mock_keras_nlp)

    # Model with compile
    model_compile = MagicMock()
    sampler = configure_beam_sampler(model_compile, beam_width=4)
    assert sampler is not None
    model_compile.compile.assert_called_once_with(sampler=sampler)

    # Model without compile but with sampler attribute
    class ModelWithSampler:
        sampler: object = None

    m_sampler = ModelWithSampler()
    sampler2 = configure_beam_sampler(m_sampler, beam_width=2)
    assert m_sampler.sampler == sampler2

    # Model with neither compile nor sampler
    class ModelBare:
        pass

    sampler3 = configure_beam_sampler(ModelBare(), beam_width=2)
    assert sampler3 is not None

    # keras_nlp without samplers attribute
    mock_keras_nlp_no_samplers = type("MockNLPNoSamplers", (), {})
    monkeypatch.setitem(sys.modules, "keras_nlp", mock_keras_nlp_no_samplers)
    assert configure_beam_sampler(ModelBare(), beam_width=2) is None

    # Import failure
    monkeypatch.setitem(sys.modules, "keras_nlp", None)
    assert configure_beam_sampler(ModelWithSampler(), beam_width=2) is None


def test_generate_sql_success_dict_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql when model returns dictionary output with scores."""

    class MockModel:
        def generate(self, prompt: str, max_length: int = 50) -> dict[str, object]:
            return {"text": f"{prompt} SELECT name FROM employees", "scores": [-0.1, -0.2]}

    class MockGemma:
        @staticmethod
        def from_preset(*_a: object, **_k: object) -> MockModel:
            return MockModel()

    mock_models = type("MockModels", (), {"GemmaCausalLM": MockGemma})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_models)
    monkeypatch.setattr(inf, "keras", type("K", (), {}))
    monkeypatch.setattr(inf, "tf", type("T", (), {}))

    res = generate_sql("gemma_sql", "List employees", beam_width=3)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT name FROM employees"
    assert 0.0 < res["confidence_score"] <= 1.0


def test_generate_sql_success_tuple_and_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql when model returns tuple or string output."""

    class MockModelTuple:
        def generate(self, prompt: str, max_length: int = 50) -> tuple[str, list[float]]:
            return (f"{prompt} SELECT 1", [0.95])

    class MockGemmaTuple:
        @staticmethod
        def from_preset(*_a: object, **_k: object) -> MockModelTuple:
            return MockModelTuple()

    mock_models_tuple = type("MockModels", (), {"GemmaCausalLM": MockGemmaTuple})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_models_tuple)
    monkeypatch.setattr(inf, "keras", type("K", (), {}))
    monkeypatch.setattr(inf, "tf", type("T", (), {}))

    res = generate_sql("gemma_sql", "prompt", beam_width=2)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT 1"
    assert res["confidence_score"] == pytest.approx(0.95)

    # Test string output
    class MockModelStr:
        def generate(self, prompt: str, max_length: int = 50) -> str:
            return f"{prompt} SELECT 42"

    class MockGemmaStr:
        @staticmethod
        def from_preset(*_a: object, **_k: object) -> MockModelStr:
            return MockModelStr()

    mock_models_str = type("MockModelsStr", (), {"GemmaCausalLM": MockGemmaStr})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_models_str)

    res_str = generate_sql("gemma_sql", "prompt")
    assert res_str["status"] == "success"
    assert res_str["sql"] == "SELECT 42"

    # Test non-string output fallback (e.g. integer or other object)
    class MockModelNonStr:
        def generate(self, prompt: str, max_length: int = 50) -> int:
            return 12345

    class MockGemmaNonStr:
        @staticmethod
        def from_preset(*_a: object, **_k: object) -> MockModelNonStr:
            return MockModelNonStr()

    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("M", (), {"GemmaCausalLM": MockGemmaNonStr}))
    res_non_str = generate_sql("gemma_sql", "prompt")
    assert res_non_str["status"] == "success"
    assert res_non_str["sql"] == "12345"


def test_generate_sql_empty_output_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test InferenceError when model generates empty output."""

    class MockModelEmpty:
        def generate(self, prompt: str, max_length: int = 50) -> str:
            return prompt

    class MockGemmaEmpty:
        @staticmethod
        def from_preset(*_a: object, **_k: object) -> MockModelEmpty:
            return MockModelEmpty()

    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("M", (), {"GemmaCausalLM": MockGemmaEmpty}))
    monkeypatch.setattr(inf, "keras", type("K", (), {}))
    monkeypatch.setattr(inf, "tf", type("T", (), {}))

    res = generate_sql("gemma_sql", "prompt")
    assert "failed" in res["status"]
    assert "empty SQL sequence" in res["status"]
    assert res["sql"] == ""
    assert res["confidence_score"] == 0.0


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DependencyMissingError when keras or tf is missing."""
    monkeypatch.setattr(inf, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        generate_sql("mock-model", "test prompt")


def test_inference_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reload when keras or tf is missing."""
    importlib = __import__("importlib", fromlist=[""])
    sys_mod = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.inference", fromlist=[""])
    monkeypatch.setitem(sys_mod.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys_mod.modules, "tensorflow", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
