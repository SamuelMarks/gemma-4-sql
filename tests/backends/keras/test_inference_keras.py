"""Module docstring."""

import math
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.inference import (
    _extract_flat_scores,
    compute_keras_confidence,
    configure_beam_sampler,
    generate_sql,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_extract_flat_scores():
    """Docstring for test_extract_flat_scores."""
    assert _extract_flat_scores([1.0, 2.0]) == [1.0, 2.0]
    assert _extract_flat_scores([[1.0], [2.0, 3.0]]) == [1.0, 2.0, 3.0]
    assert _extract_flat_scores(1.0) == [1.0]
    assert _extract_flat_scores(((1.0,), 2.0)) == [2.0, 1.0]
    assert _extract_flat_scores(["ignored", 1.0]) == [1.0]  # Cover else branch


def test_compute_keras_confidence_zero_tokens():
    """Docstring for test_compute_keras_confidence_zero_tokens."""
    assert compute_keras_confidence(None, 0) == 0.0
    assert compute_keras_confidence(None, -1) == 0.0


def test_compute_keras_confidence_with_numpy_and_tolist():
    """Docstring for test_compute_keras_confidence_with_numpy_and_tolist."""

    class DummyScores:
        """Docstring for DummyScores."""

        def numpy(self):
            """Docstring for numpy."""
            return self

        def tolist(self):
            """Docstring for tolist."""
            return [1.0, 1.0]

    scores = DummyScores()
    assert compute_keras_confidence(scores, 2) == 1.0


def test_compute_keras_confidence_list():
    """Docstring for test_compute_keras_confidence_list."""
    assert compute_keras_confidence([1.0, 1.0], 2) == 1.0
    assert compute_keras_confidence([0.5, 0.5], 2) == 0.5
    assert compute_keras_confidence([-1.0, -1.0], 2) == math.exp(-1.0)
    assert compute_keras_confidence([], 2) == 0.5


def test_compute_keras_confidence_scalar():
    """Docstring for test_compute_keras_confidence_scalar."""
    assert compute_keras_confidence(0.5, 1) == 0.5
    assert compute_keras_confidence(-2.0, 2) == math.exp(-2.0 / 2)


def test_compute_keras_confidence_fallback():
    """Docstring for test_compute_keras_confidence_fallback."""
    assert compute_keras_confidence(None, 10) == max(0.1, min(0.95, 1.0 / (1.0 + math.exp(-0.1 * 10))))


@patch("gemma_4_sql.backends.keras.inference.logger.warning")
def test_configure_beam_sampler(mock_warning):
    """Docstring for test_configure_beam_sampler."""

    # Setup mock sampler
    class MockBeamSampler:
        """Docstring for MockBeamSampler."""

        def __init__(self, num_beams):
            """Docstring for __init__."""
            self.num_beams = num_beams

    class MockKerasNLP:
        """Docstring for MockKerasNLP."""

        class samplers:
            """Docstring for samplers."""

            BeamSampler = MockBeamSampler

    # Test with compile
    class MockModelCompile:
        """Docstring for MockModelCompile."""

        def compile(self, sampler):
            """Docstring for compile."""
            self.sampler = sampler

    model_compile = MockModelCompile()
    with patch.dict("sys.modules", {"keras_nlp": MockKerasNLP()}):
        sampler = configure_beam_sampler(model_compile, 3)
        assert isinstance(sampler, MockBeamSampler)
        assert sampler.num_beams == 3
        assert model_compile.sampler == sampler

    # Test with sampler attribute
    class MockModelSampler:
        """Docstring for MockModelSampler."""

        sampler = None

    model_sampler = MockModelSampler()
    with patch.dict("sys.modules", {"keras_nlp": MockKerasNLP()}):
        sampler = configure_beam_sampler(model_sampler, 3)
        assert model_sampler.sampler == sampler

    # Test with model having neither compile nor sampler
    class MockModelNothing:
        """Docstring for MockModelNothing."""

    model_nothing = MockModelNothing()
    with patch.dict("sys.modules", {"keras_nlp": MockKerasNLP()}):
        sampler = configure_beam_sampler(model_nothing, 3)
        assert not hasattr(model_nothing, "compile")
        assert not hasattr(model_nothing, "sampler")
        assert sampler is not None

    # Test with BeamSampler being None
    class MockKerasNLPNoSampler:
        """Docstring for MockKerasNLPNoSampler."""

        class samplers:
            """Docstring for samplers."""

            BeamSampler = None

    with patch.dict("sys.modules", {"keras_nlp": MockKerasNLPNoSampler()}):
        sampler = configure_beam_sampler(model_nothing, 3)
        assert sampler is None

    # Test import error
    with patch.dict("sys.modules", {"keras_nlp": None}):
        sampler = configure_beam_sampler(model_sampler, 3)
        assert sampler is None
        mock_warning.assert_called()


def test_generate_sql_missing_deps():
    """Docstring for test_generate_sql_missing_deps."""
    with patch("gemma_4_sql.backends.keras.inference.keras", None), pytest.raises(DependencyMissingError):
        generate_sql("model", "prompt")

    with patch("gemma_4_sql.backends.keras.inference.tf", None), pytest.raises(DependencyMissingError):
        generate_sql("model", "prompt")


def test_generate_sql_success_dict_output():
    """Docstring for test_generate_sql_success_dict_output."""
    mock_model = MagicMock()
    mock_model.generate.return_value = {"text": "prompt SELECT * FROM t;", "scores": [0.9]}
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "prompt ")
        assert result["sql"] == "SELECT * FROM t;"
        assert result["status"] == "success"
        assert result["confidence_score"] == 0.9


def test_generate_sql_success_tuple_output():
    """Docstring for test_generate_sql_success_tuple_output."""
    mock_model = MagicMock()
    mock_model.generate.return_value = ("prompt SELECT 1;", [0.8])
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "prompt ")
        assert result["sql"] == "SELECT 1;"
        assert result["status"] == "success"


def test_generate_sql_success_str_output():
    """Docstring for test_generate_sql_success_str_output."""

    class StrOutput(str):
        """Docstring for StrOutput."""

    out = StrOutput("prompt SELECT 2;")
    out.scores = [0.7]

    mock_model = MagicMock()
    mock_model.generate.return_value = out
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "prompt ")
        assert result["sql"] == "SELECT 2;"
        assert result["status"] == "success"


def test_generate_sql_success_fallback_output():
    """Docstring for test_generate_sql_success_fallback_output."""
    mock_model = MagicMock()
    mock_model.generate.return_value = 123  # Cast to string
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "12")
        assert result["sql"] == "3"
        assert result["status"] == "success"


def test_generate_sql_empty_sql():
    """Docstring for test_generate_sql_empty_sql."""
    mock_model = MagicMock()
    mock_model.generate.return_value = "prompt "
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "prompt ")
        assert "failed" in result["status"]
        assert result["sql"] == ""


def test_generate_sql_exception():
    """Docstring for test_generate_sql_exception."""
    mock_model = MagicMock()
    mock_model.generate.side_effect = ValueError("Some error")
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model

    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.inference.keras", MagicMock()), patch("gemma_4_sql.backends.keras.inference.tf", MagicMock()):
        result = generate_sql("model", "prompt ")
        assert "failed: Some error" in result["status"]
