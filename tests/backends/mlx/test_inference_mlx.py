"""Unit and behavioral tests for MLX inference and beam search."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.mlx.inference as inf_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_real_mlx_beam_search_and_confidence() -> None:
    """Verify mlx_beam_search and compute_confidence_score execute with real MLX arrays."""
    import mlx.core as mx

    assert inf_module.compute_confidence_score([], 0) == 0.0
    assert inf_module.compute_confidence_score([0.1, 0.2], 2) == 1.0
    assert 0.0 < inf_module.compute_confidence_score([-1.0, -2.0], 2) < 1.0
    assert 0.0 < inf_module.compute_confidence_score(-3.0, 2) < 1.0

    def real_mlx_model(seq_tensor: Any) -> Any:
        """Return 3D MLX logits favoring token 5 on step 0 and EOS token 2 on step 1."""
        seq_len = seq_tensor.shape[1]
        logits = mx.zeros((1, seq_len, 16), dtype=mx.float32)
        target_tok = 5 if seq_len <= 3 else 2
        hot = mx.array([10.0 if i == target_tok else 0.0 for i in range(16)], dtype=mx.float32)
        return logits + hot

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.eos_token_id = 2
    mock_tok.decode.return_value = "SELECT 1;"

    model_fn: Any = real_mlx_model
    sql, conf = inf_module.mlx_beam_search(
        model=model_fn,
        tokenizer=mock_tok,
        prompt="prompt",
        beam_width=2,
        max_length=4,
        eos_token_id=None,
    )
    assert sql == "SELECT 1;"
    assert 0.0 < conf <= 1.0


def test_mlx_beam_search_no_model() -> None:
    """Verify mlx_beam_search raises InferenceError when model is None."""
    with pytest.raises(InferenceError, match="Valid model instance is required"):
        inf_module.mlx_beam_search(None, None, "prompt")


def test_mlx_beam_search_shape_variants() -> None:
    """Verify mlx_beam_search handles 2D, list, tuple, and 1D model outputs."""
    import mlx.core as mx

    hot = mx.array([10.0 if i == 5 else 0.0 for i in range(8)], dtype=mx.float32)
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2]
    mock_tok.eos_token_id = 2
    mock_tok.decode.return_value = "SELECT 1;"

    m_2d: Any = lambda _x: mx.stack([hot, hot])
    sql_2d, _ = inf_module.mlx_beam_search(m_2d, mock_tok, "prompt", max_length=1)
    assert sql_2d == "SELECT 1;"

    m_list: Any = lambda _x: [hot]
    sql_list, _ = inf_module.mlx_beam_search(m_list, mock_tok, "prompt", max_length=1)
    assert sql_list == "SELECT 1;"

    m_tuple: Any = lambda _x: (hot,)
    sql_tuple, _ = inf_module.mlx_beam_search(m_tuple, mock_tok, "prompt", max_length=1)
    assert sql_tuple == "SELECT 1;"

    m_1d: Any = lambda _x: hot
    sql_1d, _ = inf_module.mlx_beam_search(m_1d, mock_tok, "prompt", max_length=1)
    assert sql_1d == "SELECT 1;"


def test_mlx_beam_search_no_mx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlx_beam_search raises InferenceError when mx is None and logits lack shape."""
    monkeypatch.setattr(inf_module, "mx", None)
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.eos_token_id = 2
    m_raw: Any = lambda _x: [1, 2]
    with pytest.raises(InferenceError, match=r"yielded an empty sequence|yielded only an EOS"):
        inf_module.mlx_beam_search(m_raw, mock_tok, "prompt", max_length=1)


def test_mlx_beam_search_fallbacks_and_empty_results() -> None:
    """Verify mlx_beam_search fallback to SQLTokenizer and error handling on EOS-only or empty decoded strings."""
    import mlx.core as mx

    hot_tok5 = mx.array([10.0 if i == 5 else 0.0 for i in range(8)], dtype=mx.float32)
    mock_tok_empty = MagicMock(spec=[])
    m_tok5: Any = lambda _x: hot_tok5

    with patch("gemma_4_sql.tokenization.SQLTokenizer") as mock_sql_tok:
        mock_sql_tok.return_value.encode.return_value = [5, 6]
        mock_sql_tok.return_value.decode.return_value = "sql"
        sql, _conf = inf_module.mlx_beam_search(m_tok5, mock_tok_empty, "prompt", max_length=1)
        assert sql == "sql"

    hot_eos1 = mx.array([10.0 if i == 1 else 0.0 for i in range(8)], dtype=mx.float32)
    m_eos1: Any = lambda _x: hot_eos1
    mock_tok = MagicMock()
    mock_tok.encode.return_value = []
    mock_tok.eos_token_id = 1
    with pytest.raises(InferenceError, match="yielded only an EOS"):
        inf_module.mlx_beam_search(m_eos1, mock_tok, "prompt", max_length=1)

    mock_tok.decode.return_value = ""
    with pytest.raises(InferenceError, match="empty SQL query string"):
        inf_module.mlx_beam_search(m_tok5, mock_tok, "prompt", max_length=1)

    with pytest.raises(InferenceError, match="yielded an empty sequence"):
        inf_module.mlx_beam_search(m_tok5, mock_tok, "prompt", max_length=0, eos_token_id=5)


def test_generate_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql orchestrates model loading, beam search, error handling, and missing dependencies."""
    monkeypatch.setattr(inf_module, "load", MagicMock(return_value=(MagicMock(), MagicMock())))

    with patch.object(inf_module, "mlx_beam_search") as mock_beam:
        mock_beam.return_value = ("SELECT 1", 0.9)
        res = inf_module.generate_sql("model", "prompt", eos_token_id=2)
        assert res["sql"] == "SELECT 1"
        assert res["status"] == "success"

        mock_beam.side_effect = RuntimeError("error")
        res2 = inf_module.generate_sql("model", "prompt")
        assert res2["status"] == "failed: error"
        assert res2["sql"] == ""

    monkeypatch.setattr(inf_module, "load", None)
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt")
