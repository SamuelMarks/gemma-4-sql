"""Unit and behavioral tests for MaxText inference and beam search."""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import MagicMock, patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.maxtext import inference
from gemma_4_sql.backends.maxtext.inference import _beam_search_step, _execute_generate, generate_sql, maxtext_beam_search
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer


@pytest.fixture(autouse=True)
def clear_maxtext_cache() -> None:
    """Clear the MaxText model cache before each test."""
    inference._MODEL_CACHE.clear()


def test_beam_search_step_real_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _beam_search_step expands sequences using real jax.nn.log_softmax and jnp.argsort."""
    seq = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    def model_apply_3d(_s: jax.Array) -> jax.Array:
        """Return 3D logits with peak at token index 4."""
        return jnp.array([[[0.0] * 5, [0.0] * 5, [0.1, 0.2, 0.3, 0.4, 5.0]]], dtype=jnp.float32)

    beams = _beam_search_step(seq, 0.0, model_apply_3d, beam_width=2)
    assert len(beams) == 2
    new_seq_1, score_1 = beams[0]
    assert bool(jnp.array_equal(jnp.asarray(new_seq_1), jnp.array([[1, 2, 3, 4]])))
    expected_log_probs = jax.nn.log_softmax(jnp.array([0.1, 0.2, 0.3, 0.4, 5.0]))
    assert score_1 == pytest.approx(float(expected_log_probs[4]), rel=1e-5)

    def model_apply_2d(_s: jax.Array) -> jax.Array:
        """Return 2D logits with peak at token index 4."""
        return jnp.array([[0.0] * 5, [0.0] * 5, [0.1, 0.2, 0.3, 0.4, 5.0]], dtype=jnp.float32)

    beams_2d = _beam_search_step(seq, 0.0, model_apply_2d, beam_width=1)
    assert bool(jnp.array_equal(jnp.asarray(beams_2d[0][0]), jnp.array([[1, 2, 3, 4]])))

    def model_apply_1d(_s: jax.Array) -> jax.Array:
        """Return 1D logits with peak at token index 4."""
        return jnp.array([0.1, 0.2, 0.3, 0.4, 5.0], dtype=jnp.float32)

    beams_1d = _beam_search_step(seq, 0.0, model_apply_1d, beam_width=1)
    assert bool(jnp.array_equal(jnp.asarray(beams_1d[0][0]), jnp.array([[1, 2, 3, 4]])))

    # Also test when jax.jit is absent
    monkeypatch.delattr(inference.jax, "jit", raising=False)
    beams_no_jit = _beam_search_step(seq, 0.0, model_apply_1d, beam_width=1)
    assert bool(jnp.array_equal(jnp.asarray(beams_no_jit[0][0]), jnp.array([[1, 2, 3, 4]])))


def test_maxtext_beam_search_and_eos_handling() -> None:
    """Verify maxtext_beam_search decodes tokens and terminates early on EOS."""

    def always_one(_seq: jax.Array) -> jax.Array:
        """Return logits favoring token 1."""
        return jnp.array([0.0, 10.0], dtype=jnp.float32)

    out_seq, score = maxtext_beam_search(
        model_apply_fn=always_one,
        input_ids=jnp.array([[0]], dtype=jnp.int32),
        beam_width=1,
        max_length=3,
        eos_token_id=5,
    )
    assert bool(jnp.array_equal(jnp.asarray(out_seq), jnp.array([[0, 1, 1, 1]])))
    assert score < 0.0

    call_count = 0

    def eos_on_second(_seq: jax.Array) -> jax.Array:
        """Predict token 1 on step 1 and token 0 (EOS) on step 2."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return jnp.array([0.0, 10.0], dtype=jnp.float32)
        return jnp.array([10.0, 0.0], dtype=jnp.float32)

    out_eos, _ = maxtext_beam_search(
        model_apply_fn=eos_on_second,
        input_ids=jnp.array([[2]], dtype=jnp.int32),
        beam_width=1,
        max_length=5,
        eos_token_id=0,
    )
    assert bool(jnp.array_equal(jnp.asarray(out_eos), jnp.array([[2, 1, 0]])))


def test_execute_generate_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_generate runs with real JAX arrays, caches models, and handles callable models."""

    class DummyMaxTextModel:
        """Stateless MaxText model stub with init and apply."""

        def __init__(self, name: str) -> None:
            """Initialize DummyMaxTextModel."""
            self.name = name

        def init(self, _rng: jax.Array, _inputs: jax.Array) -> dict[str, jax.Array]:
            """Return dummy parameter dictionary."""
            return {"w": jnp.ones((1,))}

        def apply(self, _params: Mapping[str, jax.Array], seq: jax.Array) -> jax.Array:
            """Return logits favoring token 65 ('A')."""
            logits = jnp.zeros((1, seq.shape[1], 128), dtype=jnp.float32)
            return logits.at[:, :, 65].set(10.0)

    monkeypatch.setattr(inference, "Gemma4Model", DummyMaxTextModel)
    tokenizer = SQLTokenizer()

    status, sql, score = _execute_generate("test-model", [83, 69], 1, 2, 127, tokenizer)
    assert status == "success"
    assert "AA" in sql
    assert isinstance(score, float)

    # Second call hits _MODEL_CACHE
    status2, _, _ = _execute_generate("test-model", [83, 69], 1, 2, 127, tokenizer)
    assert status2 == "success"

    for i in range(inference.MAX_MODEL_CACHE_SIZE + 1):
        inference._get_or_create_maxtext_model(f"extra-model-{i}")
    assert len(inference._MODEL_CACHE) == inference.MAX_MODEL_CACHE_SIZE

    # Model without .apply or .init (direct callable) and output without __len__
    class CallableOnlyModel:
        """Model without .apply or .init methods."""

        def __init__(self, _name: str) -> None:
            """Initialize CallableOnlyModel."""

        def __call__(self, seq: jax.Array) -> jax.Array:
            """Return uniform logits."""
            return jnp.ones((1, seq.shape[1], 16), dtype=jnp.float32)

    monkeypatch.setattr(inference, "Gemma4Model", CallableOnlyModel)
    status_call, _, _ = _execute_generate("callable-model", [65], 1, 1, 15, tokenizer)
    assert status_call == "success"


def test_generate_sql_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql returns structured results, captures runtime errors, and checks dependencies."""
    monkeypatch.setattr(inference, "Gemma4Model", MagicMock())
    with patch("gemma_4_sql.backends.maxtext.inference._execute_generate", return_value=("success", "SELECT 1", 0.9)):
        result = generate_sql("test-model", "test prompt")
        assert result["sql"] == "SELECT 1"
        assert result["status"] == "success"
        assert result["confidence_score"] == 0.9

    with patch("gemma_4_sql.backends.maxtext.inference._execute_generate", side_effect=RuntimeError("Generation failed")):
        err_res = generate_sql("test-model", "test prompt")
        assert err_res["sql"] == ""
        assert "failed: Generation failed" in err_res["status"]

    monkeypatch.setattr(inference, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        generate_sql("test-model", "test prompt")
