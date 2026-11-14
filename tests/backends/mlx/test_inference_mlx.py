"""Unit and behavioral tests for MLX inference and beam search without MagicMock."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest

import gemma_4_sql.backends.mlx.inference as inf_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import MlxArrayProtocol
from tests.conftest import requires_mlx
from tests.utils.tensor_helpers import TypedMlxLinearProtocol, get_typed_mlx_nn, get_typed_mx

pytestmark = requires_mlx


class _StubMlxTokenizer:
    """Typed MLX tokenizer double supporting `encode`, `decode`, and `eos_token_id`."""

    def __init__(
        self,
        encoded: list[int] | None = None,
        decoded: str = "SELECT 1;",
        eos_token_id: int = 2,
    ) -> None:
        """Initialize tokenizer stub."""
        self._encoded = encoded if encoded is not None else [1, 2, 3]
        self.decoded = decoded
        self.eos_token_id = eos_token_id

    def encode(self, _text: str) -> Sequence[int]:
        """Return `self._encoded`."""
        return self._encoded

    def decode(self, _ids: Sequence[int]) -> str:
        """Return `self.decoded`."""
        return self.decoded


def test_real_mlx_beam_search_and_confidence() -> None:
    """Verify mlx_beam_search and compute_confidence_score execute with real MLX arrays."""
    mx = get_typed_mx()

    assert inf_module.compute_confidence_score([], 0) == 0.0
    assert inf_module.compute_confidence_score([0.1, 0.2], 2) == 1.0
    assert 0.0 < inf_module.compute_confidence_score([-1.0, -2.0], 2) < 1.0
    assert 0.0 < inf_module.compute_confidence_score(-3.0, 2) < 1.0

    def real_mlx_model(seq_tensor: MlxArrayProtocol) -> MlxArrayProtocol:
        """Return 3D MLX logits favoring token 5 on step 0 and EOS token 2 on step 1."""
        seq_len = int(seq_tensor.shape[1])
        logits = mx.zeros((1, seq_len, 16), dtype=mx.float32)
        target_tok = 5 if seq_len <= 3 else 2
        hot = mx.array([10.0 if i == target_tok else 0.0 for i in range(16)], dtype=mx.float32)
        return logits + hot

    tok = _StubMlxTokenizer(encoded=[1, 2, 3], decoded="SELECT 1;", eos_token_id=2)
    sql, conf = inf_module.mlx_beam_search(
        model=real_mlx_model,
        tokenizer=tok,
        prompt="prompt",
        beam_width=2,
        max_length=4,
        eos_token_id=None,
    )
    assert sql == "SELECT 1;"
    assert 0.0 < conf <= 1.0

    # Also test beam carry-over when one beam finishes on step 0 while sibling beam continues on step 1
    def split_beam_model(seq_tensor: MlxArrayProtocol) -> MlxArrayProtocol:
        """Favor both EOS (2) and token (5) on step 0, then EOS (2) on step 1.

        Returns:
            Logit array for the current decoding step.
        """
        seq_len = int(seq_tensor.shape[1])
        if seq_len <= 3:
            return mx.array([10.0 if i == 2 else (9.5 if i == 5 else 0.0) for i in range(16)], dtype=mx.float32)
        return mx.array([10.0 if i == 6 else 0.0 for i in range(16)], dtype=mx.float32)

    sql_carry, _ = inf_module.mlx_beam_search(
        model=split_beam_model,
        tokenizer=tok,
        prompt="prompt",
        beam_width=2,
        max_length=3,
        eos_token_id=None,
    )
    assert sql_carry == "SELECT 1;"


def test_mlx_beam_search_no_model() -> None:
    """Verify mlx_beam_search raises InferenceError when model is None."""
    with pytest.raises(InferenceError, match="Valid model instance is required"):
        inf_module.mlx_beam_search(None, None, "prompt")


def test_mlx_beam_search_shape_variants() -> None:
    """Verify mlx_beam_search handles 2D, list, tuple, scalar, and 1D model outputs."""
    mx = get_typed_mx()

    hot = mx.array([10.0 if i == 5 else 0.0 for i in range(8)], dtype=mx.float32)
    tok = _StubMlxTokenizer(encoded=[1, 2], decoded="SELECT 1;", eos_token_id=2)

    def _model_2d(_x: MlxArrayProtocol) -> MlxArrayProtocol:
        """Return 2D stacked logits."""
        return mx.stack([hot, hot])

    def _model_list(_x: MlxArrayProtocol) -> list[MlxArrayProtocol]:
        """Return list of logits."""
        return [hot]

    def _model_tuple(_x: MlxArrayProtocol) -> tuple[MlxArrayProtocol]:
        """Return 1-tuple of logits."""
        return (hot,)

    def _model_1d(_x: MlxArrayProtocol) -> MlxArrayProtocol:
        """Return 1D logits."""
        return hot

    def _model_scalar(_x: MlxArrayProtocol) -> float:
        """Return scalar float (producing EOS fallback candidate)."""
        return 0.0

    assert inf_module.mlx_beam_search(_model_2d, tok, "prompt", max_length=1)[0] == "SELECT 1;"
    assert inf_module.mlx_beam_search(_model_list, tok, "prompt", max_length=1)[0] == "SELECT 1;"
    assert inf_module.mlx_beam_search(_model_tuple, tok, "prompt", max_length=1)[0] == "SELECT 1;"
    assert inf_module.mlx_beam_search(_model_1d, tok, "prompt", max_length=1)[0] == "SELECT 1;"
    with pytest.raises(InferenceError, match="yielded only an EOS"):
        inf_module.mlx_beam_search(_model_scalar, tok, "prompt", max_length=1)


def test_mlx_beam_search_no_mx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlx_beam_search raises InferenceError when mx is None and logits lack shape."""
    monkeypatch.setattr(inf_module, "mx", None)
    tok = _StubMlxTokenizer(encoded=[1, 2, 3], eos_token_id=2)

    def _plain_list_model(_x: list[int]) -> list[int]:
        """Return plain Python list without .shape."""
        return [1, 2]

    with pytest.raises(InferenceError, match=r"yielded an empty sequence|yielded only an EOS"):
        inf_module.mlx_beam_search(_plain_list_model, tok, "prompt", max_length=1)


def test_mlx_beam_search_fallbacks_and_empty_results() -> None:
    """Verify mlx_beam_search fallback to SQLTokenizer and error handling on EOS-only or empty decoded strings."""
    mx = get_typed_mx()

    hot_tok65 = mx.array([10.0 if i == 68 else 0.0 for i in range(128)], dtype=mx.float32)

    def _tok68_model(_x: MlxArrayProtocol) -> MlxArrayProtocol:
        """Return hot_tok65 favoring printable ASCII token ID 68."""
        return hot_tok65

    sql, _conf = inf_module.mlx_beam_search(_tok68_model, None, "prompt", max_length=1)
    assert len(sql) > 0

    hot_eos1 = mx.array([10.0 if i == 1 else 0.0 for i in range(8)], dtype=mx.float32)

    def _eos1_model(_x: MlxArrayProtocol) -> MlxArrayProtocol:
        """Return hot_eos1."""
        return hot_eos1

    tok = _StubMlxTokenizer(encoded=[], eos_token_id=1)
    with pytest.raises(InferenceError, match="yielded only an EOS"):
        inf_module.mlx_beam_search(_eos1_model, tok, "prompt", max_length=1)

    tok.decoded = ""
    with pytest.raises(InferenceError, match="empty SQL query string"):
        inf_module.mlx_beam_search(_tok68_model, tok, "prompt", max_length=1)

    with pytest.raises(InferenceError, match="yielded an empty sequence"):
        inf_module.mlx_beam_search(_tok68_model, tok, "prompt", max_length=0, eos_token_id=5)


def test_generate_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql orchestrates loading, caching, beam search, errors, and missing deps."""
    mlx_nn = get_typed_mlx_nn()

    inf_module._MODEL_CACHE.clear()
    lin = mlx_nn.Linear(4, 4)
    tok = _StubMlxTokenizer()

    def _load_pair(_name: str) -> tuple[TypedMlxLinearProtocol, _StubMlxTokenizer]:
        """Return `(lin, tok)`."""
        return (lin, tok)

    monkeypatch.setattr(inf_module, "load", _load_pair)

    with patch.object(inf_module, "mlx_beam_search", return_value=("SELECT 1", 0.9)):
        res = inf_module.generate_sql("model", "prompt", eos_token_id=2)
        assert res["sql"] == "SELECT 1"
        assert res["status"] == "success"

        # Second call hits _MODEL_CACHE
        res_cached = inf_module.generate_sql("model", "prompt")
        assert res_cached["sql"] == "SELECT 1"

        res_mm = inf_module.generate_sql("model", "prompt", image_path="img.png", audio_path="aud.wav")
        assert res_mm["sql"] == "SELECT 1"

    with patch.object(inf_module, "mlx_beam_search", side_effect=RuntimeError("error")):
        res2 = inf_module.generate_sql("model", "prompt")
        assert res2["status"] == "failed: error"
        assert res2["sql"] == ""

    monkeypatch.setattr(inf_module, "load", None)
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt")
