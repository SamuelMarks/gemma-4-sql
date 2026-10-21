"""Unit and behavioral tests for Keras inference, beam sampling, and confidence scoring."""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.keras.inference as inf
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_extract_flat_scores() -> None:
    """Verify _extract_flat_scores flattens nested score lists and ignores non-numeric entries."""
    res = inf._extract_flat_scores([[1.0, 2], 3.0])
    assert res == [3.0, 1.0, 2.0]

    res2 = inf._extract_flat_scores([[1.0, 2], "skip_me", None, 3.0])
    assert res2 == [3.0, 1.0, 2.0]


def test_compute_keras_confidence() -> None:
    """Verify compute_keras_confidence handles tensor-like objects, negative log-probs, scalars, and empty lists."""
    assert inf.compute_keras_confidence(None, 0) == 0.0

    mock_np = MagicMock()
    mock_np.numpy.return_value = [0.1, 0.2]
    mock_np.tolist.return_value = [0.1, 0.2]
    assert inf.compute_keras_confidence(mock_np, 2) == pytest.approx(0.15)

    assert inf.compute_keras_confidence([-1.0, -2.0], 2) == pytest.approx(math.exp(-1.5))
    assert inf.compute_keras_confidence(0.5, 2) == 0.5
    assert inf.compute_keras_confidence(-1.0, 2) == pytest.approx(math.exp(-0.5))
    assert inf.compute_keras_confidence([], 2) == 0.5
    assert inf.compute_keras_confidence(None, 10) > 0.0


def test_configure_beam_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configure_beam_sampler attaches BeamSampler via model.compile or model.sampler."""
    mock_nlp = MagicMock()
    mock_nlp.samplers.BeamSampler = MagicMock(return_value="sampler_obj")
    monkeypatch.setitem(sys.modules, "keras_nlp", mock_nlp)

    mock_model = MagicMock()
    inf.configure_beam_sampler(mock_model, 2)
    mock_model.compile.assert_called_with(sampler="sampler_obj")

    mock_model2 = MagicMock(spec=["sampler"])
    inf.configure_beam_sampler(mock_model2, 2)
    assert mock_model2.sampler == "sampler_obj"

    mock_model_none = MagicMock(spec=[])
    inf.configure_beam_sampler(mock_model_none, 2)

    mock_nlp.samplers.BeamSampler = None
    assert inf.configure_beam_sampler(mock_model_none, 2) is None

    monkeypatch.setitem(sys.modules, "keras_nlp", None)
    assert inf.configure_beam_sampler(MagicMock(), 2) is None


def test_generate_sql_and_output_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql handles string, dict, tuple, and integer generation outputs and missing dependencies."""
    monkeypatch.setattr(inf, "keras", None)
    monkeypatch.setattr(inf, "tf", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        inf.generate_sql("m", "p")

    monkeypatch.setattr(inf, "keras", MagicMock())
    monkeypatch.setattr(inf, "tf", MagicMock())
    monkeypatch.setattr(inf, "configure_beam_sampler", MagicMock())

    def setup_mock(ret_val: object) -> None:
        """Install a GemmaCausalLM mock returning `ret_val` on `.generate()`."""
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.generate.return_value = ret_val
        mock_cls.from_preset.return_value = mock_model
        monkeypatch.setattr(inf, "GemmaCausalLM", mock_cls)

    setup_mock("prompt SELECT 1")
    res = inf.generate_sql("m", "prompt ")
    assert res["status"] == "success"
    assert res["sql"] == "SELECT 1"

    setup_mock({"text": "prompt SEL", "scores": 0.5})
    assert inf.generate_sql("m", "prompt ")["status"] == "success"

    setup_mock(("prompt SEL", 0.5))
    assert inf.generate_sql("m", "prompt ")["status"] == "success"

    setup_mock("prompt ")
    assert "failed" in inf.generate_sql("m", "prompt ")["status"]

    setup_mock(12345)
    res_num = inf.generate_sql("m", "")
    assert res_num["status"] == "success"
    assert "12345" in res_num["sql"]

    monkeypatch.setattr(inf, "GemmaCausalLM", None)
    monkeypatch.setattr(inf, "optional_attr", lambda *_a, **_k: None)
    res_missing_cls = inf.generate_sql("m", "SELECT")
    assert res_missing_cls["status"].startswith("failed:")
    assert "keras_nlp.models.GemmaCausalLM is not available" in res_missing_cls["status"]
