"""Unit tests for JAX inference, beam search, multimodal forwarding, and checkpoint restoration."""

from __future__ import annotations

import pathlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.jax.inference as inf
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from tests.conftest import requires_jax

pytestmark = requires_jax


@pytest.fixture(autouse=True)
def _bypass_jax_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass XLA jit compilation during unit tests for fast, deterministic execution."""
    if inf.jax is not None:
        monkeypatch.setattr(inf.jax, "jit", lambda fn, *_args, **_kwargs: fn)


def _make_tiny_jax_model() -> Any:
    """Instantiate a real tiny JAX Gemma4ForCausalLM model for behavioral testing.

    Returns:
        A tiny `Gemma4ForCausalLM` instance initialized with Flax NNX RNGs.

    """
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=320,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
    )
    return Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))


def test_compute_step_probs_real_jax_tensors() -> None:
    """Verify _compute_step_probs computes top-k indices and log probabilities on 3D, 2D, and 1D JAX arrays."""
    import jax.numpy as jnp

    logits_3d = jnp.array([[[0.1, 0.2, 5.0, -1.0]]], dtype=jnp.float32)
    top_idx_3d, top_probs_3d = inf._compute_step_probs(logits_3d, beam_width=2)
    assert int(top_idx_3d[0]) == 2
    assert top_probs_3d.shape == (2,)
    assert float(top_probs_3d[0]) > float(top_probs_3d[1])

    logits_2d = jnp.array([[0.1, 4.0, 0.2, -1.0]], dtype=jnp.float32)
    top_idx_2d, _ = inf._compute_step_probs(logits_2d, beam_width=2)
    assert int(top_idx_2d[0]) == 1

    logits_1d = jnp.array([3.0, 0.1, 0.2, -1.0], dtype=jnp.float32)
    top_idx_1d, _ = inf._compute_step_probs(logits_1d, beam_width=2)
    assert int(top_idx_1d[0]) == 0


def test_beam_search_step_and_jax_beam_search_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _beam_search_step and jax_beam_search execute with a real tiny Gemma4ForCausalLM."""
    import jax.numpy as jnp

    model = _make_tiny_jax_model()
    input_ids = jnp.array([[10, 11]], dtype=jnp.int32)

    beams = inf._beam_search_step(input_ids, 0.0, model, beam_width=2)
    assert len(beams) == 2
    assert beams[0][0].shape == (1, 3)
    assert isinstance(beams[0][1], float)

    best_seq, best_score = inf.jax_beam_search(model, input_ids, beam_width=2, max_length=2, eos_token_id=319)
    assert best_seq.shape == (1, 4)
    assert isinstance(best_score, float)

    # Verify early stopping when EOS token is already at the end of the sequence
    eos_seq = jnp.array([[10, 319]], dtype=jnp.int32)
    stopped_seq, stopped_score = inf.jax_beam_search(model, eos_seq, beam_width=2, max_length=3, eos_token_id=319)
    assert stopped_seq.shape == (1, 2)
    assert stopped_score == 0.0

    # Verify fallback when jax.jit is not present on the jax namespace
    fake_jax = MagicMock()
    del fake_jax.jit
    fake_jax.nn = inf.jax.nn  # type: ignore[union-attr]
    monkeypatch.setattr(inf, "jax", fake_jax)
    beams_no_jit = inf._beam_search_step(input_ids, 0.0, model, beam_width=2)
    assert len(beams_no_jit) == 2


def test_generate_sql_real_tiny_model_and_caching(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify generate_sql initializes a tiny model, caches it, and decodes SQL output."""
    tiny_model = _make_tiny_jax_model()
    monkeypatch.setattr(inf.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_model.config))
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", lambda _cfg, rngs=None: tiny_model)

    model_key = str(tmp_path / "nonexistent_checkpoint")
    inf._MODEL_CACHE.pop(model_key, None)

    res = inf.generate_sql(model_key, "SELECT 1", beam_width=2, max_length=2)
    assert res["backend"] == "jax"
    assert res["model"] == model_key
    assert res["status"] == "success"
    assert res["beam_width"] == 2
    assert isinstance(res["sql"], str)
    assert isinstance(res["confidence_score"], float)
    assert model_key in inf._MODEL_CACHE

    # Second call hits _MODEL_CACHE
    res_cached = inf.generate_sql(model_key, "SELECT 1", beam_width=1, max_length=1)
    assert res_cached["status"] == "success"


def test_generate_sql_checkpoint_restore_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify generate_sql handles Orbax checkpoint restoration success, None return, no nnx.update, and errors."""
    tiny_model = _make_tiny_jax_model()
    monkeypatch.setattr(inf.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_model.config))
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", lambda _cfg, rngs=None: tiny_model)
    monkeypatch.setattr(inf, "jax_beam_search", lambda _fn, ids, _bw, _ml, _eos: (ids, -0.5))

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    # 1. Restore succeeds and calls nnx.update
    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.return_value = "restored_state"
    monkeypatch.setitem(sys.modules, "orbax", MagicMock(checkpoint=mock_ocp))
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", mock_ocp)

    mock_nnx = MagicMock()
    monkeypatch.setattr(inf, "nnx", mock_nnx)
    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res1 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res1["status"] == "success"
    mock_nnx.update.assert_called_once()

    # 2. Restore returns None (skips nnx.update)
    mock_ocp.PyTreeCheckpointer.return_value.restore.return_value = None
    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res2 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res2["status"] == "success"

    # 3. Restore raises RuntimeError (gracefully ignored unless strict_checkpoint=True)
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = RuntimeError("corrupt ckpt")
    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res3 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res3["status"] == "success"

    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    with pytest.raises(InferenceError, match="corrupt ckpt"):
        inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1, strict_checkpoint=True)

    # 4. Verify _BoundedModelCache LRU eviction and update
    small_cache = inf._BoundedModelCache(maxsize=2)
    small_cache.put_model("a", 1)
    small_cache.put_model("a", 10)
    small_cache.put_model("b", 2)
    assert small_cache.get_model("a") == 10
    small_cache.put_model("c", 3)
    assert small_cache.get_model("b") is None
    assert small_cache.get_model("c") == 3


def test_generate_sql_multimodal_and_typeerror_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql processes image/audio paths and falls back when model rejects extra kwargs."""
    import jax.numpy as jnp

    monkeypatch.setattr(
        "gemma_4_sql.backends.common_multimodal.process_image",
        lambda _p: {"pixel_values": [[1.0, 2.0]]},
    )
    monkeypatch.setattr(
        "gemma_4_sql.backends.common_multimodal.process_audio",
        lambda _p: {"audio_values": [[0.5, -0.5]]},
    )

    def strict_PosOnlyModel(seq: Any, _pos: Any) -> Any:
        """Model callable that rejects keyword arguments, triggering TypeError fallback."""
        return jnp.ones((1, seq.shape[1], 320), dtype=jnp.float32)

    monkeypatch.setitem(inf._MODEL_CACHE, "strict_pos_model", strict_PosOnlyModel)

    res = inf.generate_sql(
        "strict_pos_model",
        "SELECT * FROM media",
        beam_width=1,
        max_length=1,
        image_path="sample.png",
        audio_path="sample.wav",
    )
    assert res["status"] == "success"
    assert "<image>" in str(res["prompt"])
    assert "<audio>" in str(res["prompt"])

    # Also test when pixel_values and audio_values are pre-supplied and output_ids[0] has no __len__
    class RowWithoutLen:
        """Stub sequence row without __len__ to exercise output_ids.shape[1] fallback."""

        def tolist(self) -> list[int]:
            """Return token list."""
            return [10, 11]

    class OutputArrayStub:
        """Stub 2D output array."""

        shape = (1, 2)

        def __getitem__(self, _idx: int) -> RowWithoutLen:
            """Return RowWithoutLen."""
            return RowWithoutLen()

    def _mock_beam_search(fn: Any, ids: Any, _bw: int, _ml: int, _eos: int) -> tuple[OutputArrayStub, float]:
        """Invoke forward pass and return OutputArrayStub."""
        _ = fn(ids, ids)
        return (OutputArrayStub(), -0.2)

    monkeypatch.setattr(inf, "jax_beam_search", _mock_beam_search)
    res2 = inf.generate_sql(
        "strict_pos_model",
        "SELECT 1",
        beam_width=1,
        max_length=1,
        image_path="sample.png",
        pixel_values=jnp.ones((1, 2), dtype=jnp.float32),
        audio_path="sample.wav",
        audio_values=jnp.ones((1, 2), dtype=jnp.float32),
    )
    assert res2["status"] == "success"


def test_generate_sql_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql raises DependencyMissingError when JAX is unavailable."""
    monkeypatch.setattr(inf, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX inference dependencies are missing"):
        inf.generate_sql("m", "SELECT 1")
