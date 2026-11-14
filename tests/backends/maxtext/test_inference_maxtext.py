"""Unit and behavioral tests for MaxText inference, `(model, params)` caching, and Orbax checkpoint reloading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import pytest

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.maxtext import inference
from gemma_4_sql.backends.maxtext.export import export_model
from gemma_4_sql.backends.maxtext.inference import (
    _beam_search_step,
    _execute_generate,
    cache_maxtext_params,
    generate_sql,
    get_cached_maxtext_params,
    load_maxtext_orbax_params,
    maxtext_beam_search,
)
from gemma_4_sql.backends.maxtext.train import save_maxtext_checkpoint
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import PyTreeNode


@pytest.fixture(autouse=True)
def clear_maxtext_cache() -> None:
    """Clear the MaxText model and parameter caches before each test."""
    inference._MODEL_CACHE.clear()
    inference._PARAMS_ONLY_CACHE.clear()


def test_beam_search_step_real_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_beam_search_step` expands sequences using real `jax.nn.log_softmax` and `jnp.argsort`."""
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

    monkeypatch.delattr(inference.jax, "jit", raising=False)
    beams_no_jit = _beam_search_step(seq, 0.0, model_apply_1d, beam_width=1)
    assert bool(jnp.array_equal(jnp.asarray(beams_no_jit[0][0]), jnp.array([[1, 2, 3, 4]])))


def test_maxtext_beam_search_and_eos_handling() -> None:
    """Verify `maxtext_beam_search` decodes tokens and terminates early on EOS."""

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
        """Predict token 1 on step 1 and token 0 (EOS) on step 2.

        Returns:
            Logits favoring token 1 on first call and token 0 thereafter.

        """
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


def test_execute_generate_caches_params_without_reinit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_execute_generate` caches `(model, params)` so `model.init` runs only once across calls."""
    tokenizer = SQLTokenizer()
    a_token = tokenizer.byte_fallback.byte_to_token_ids(ord("A"))[0]
    init_calls = 0

    class DummyMaxTextModel:
        """Stateless MaxText model stub whose output depends on `params['token_bias']`."""

        def __init__(self, name: str) -> None:
            """Initialize `DummyMaxTextModel`."""
            self.name = name

        def init(self, _rng: jax.Array, _inputs: jax.Array) -> dict[str, PyTreeNode]:
            """Increment `init_calls` and return biased parameter dictionary.

            Returns:
                A parameter dictionary containing `'token_bias'`.

            """
            nonlocal init_calls
            init_calls += 1
            bias = jnp.zeros((260,), dtype=jnp.float32).at[a_token].set(10.0)
            return {"token_bias": bias}

        def apply(self, params: Mapping[str, PyTreeNode], seq: jax.Array) -> jax.Array:
            """Broadcast `params['token_bias']` across sequence positions.

            Returns:
                Vocabulary logits broadcast across `(1, seq_len, 260)`.

            """
            tb = params["token_bias"]
            assert isinstance(tb, jax.Array)
            return jnp.broadcast_to(tb[None, None, :], (1, seq.shape[1], 260))

    monkeypatch.setattr(inference, "Gemma4Model", DummyMaxTextModel)

    status, sql, score = _execute_generate("test-model", [83, 69], 1, 2, 127, tokenizer)
    assert status == "success"
    assert "AA" in sql
    assert isinstance(score, float)
    assert init_calls == 1

    # Second call reuses cached `(model, params)` without re-running `model.init`!
    status2, sql2, _ = _execute_generate("test-model", [83, 69], 1, 2, 127, tokenizer)
    assert status2 == "success"
    assert sql2 == sql
    assert init_calls == 1

    # Updating cached params via `cache_maxtext_params` updates existing cached `(model, params)` tuple
    b_token = tokenizer.byte_fallback.byte_to_token_ids(ord("B"))[0]
    b_bias = jnp.zeros((260,), dtype=jnp.float32).at[b_token].set(12.0)
    cache_maxtext_params("test-model", {"token_bias": b_bias})
    assert get_cached_maxtext_params("test-model") is not None
    status3, sql3, _ = _execute_generate("test-model", [83, 69], 1, 2, 127, tokenizer)
    assert status3 == "success"
    assert "BB" in sql3

    # Standalone `_PARAMS_ONLY_CACHE` hit when `_MODEL_CACHE` has not seen `"params-only"` yet
    cache_maxtext_params("params-only", {"token_bias": b_bias})
    assert get_cached_maxtext_params("params-only") is not None
    _, sql_po, _ = _execute_generate("params-only", [83], 1, 2, 127, tokenizer)
    assert "BB" in sql_po

    for i in range(inference.MAX_MODEL_CACHE_SIZE + 1):
        inference._get_or_create_maxtext_model(f"extra-model-{i}")
    assert len(inference._MODEL_CACHE) == inference.MAX_MODEL_CACHE_SIZE

    class CallableOnlyModel:
        """Model without `.apply` or `.init` methods."""

        def __init__(self, _name: str) -> None:
            """Initialize `CallableOnlyModel`."""

        def __call__(self, seq: jax.Array) -> jax.Array:
            """Return uniform logits."""
            return jnp.ones((1, seq.shape[1], 16), dtype=jnp.float32)

    monkeypatch.setattr(inference, "Gemma4Model", CallableOnlyModel)
    status_call, _, _ = _execute_generate("callable-model", [65], 1, 1, 15, tokenizer)
    assert status_call == "success"

    # Also cover scalar `output_ids[0].tolist()` edge case (0D row)
    with patch.object(
        inference, "maxtext_beam_search", return_value=(jnp.array([68], dtype=jnp.int32).reshape(1, 1)[:, 0], -0.1)
    ):
        # shape of `arr[:, 0]` is `(1,)` so `output_ids[0].tolist()` is an `int` scalar; mock `.shape` via 2D view?
        pass

    class _ZeroDimRowOutput:
        """2D-shaped array wrapper whose `[0].tolist()` returns a Python scalar."""

        shape = (1, 2)

        def __getitem__(self, _idx: int) -> jax.Array:
            """Return 0D scalar array."""
            return jnp.array(68, dtype=jnp.int32)

    with patch.object(inference, "maxtext_beam_search", return_value=(_ZeroDimRowOutput(), -0.1)):
        status_sc, _, _ = _execute_generate("callable-model", [65], 1, 1, 15, tokenizer)
        assert status_sc == "success"


def test_exported_and_trained_orbax_checkpoint_reloads_into_generate_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove exported and trained Orbax checkpoints reload into `generate_sql` with deterministic custom weights."""
    tokenizer = SQLTokenizer()
    default_tok = tokenizer.byte_fallback.byte_to_token_ids(ord("X"))[0]
    persisted_tok = tokenizer.byte_fallback.byte_to_token_ids(ord("S"))[0]

    class ParamSensitiveMaxTextModel:
        """Stateless model whose `init()` favors `'X'` unless custom Orbax weights favoring `'S'` are loaded."""

        def __init__(self, name: str) -> None:
            """Store model name."""
            self.name = name

        def init(self, _rng: jax.Array, _inputs: jax.Array) -> dict[str, PyTreeNode]:
            """Default `PRNGKey(0)` weights favor `'X'`.

            Returns:
                A parameter dictionary containing `'vocab_bias'`.

            """
            logits_vec = jnp.zeros((260,), dtype=jnp.float32).at[default_tok].set(10.0)
            return {"vocab_bias": logits_vec}

        def apply(self, params: Mapping[str, PyTreeNode], seq: jax.Array) -> jax.Array:
            """Return `params['vocab_bias']` at every sequence step."""
            vb = params["vocab_bias"]
            assert isinstance(vb, jax.Array)
            return jnp.broadcast_to(vb[None, None, :], (1, seq.shape[1], 260))

    monkeypatch.setattr(inference, "Gemma4Model", ParamSensitiveMaxTextModel)

    # 1. Default un-exported model predicts 'XXX'
    default_res = generate_sql("fresh-maxtext", "SELECT", beam_width=1, max_length=3)
    assert "XXX" in default_res["sql"]

    # 2. Export custom weights favoring 'S' to Orbax, clear all caches,
    # and reload via `checkpoint_path` AND `model_name=ckpt_dir`!
    custom_bias = jnp.zeros((260,), dtype=jnp.float32).at[persisted_tok].set(15.0)
    export_dir = tmp_path / "exported_s_model"
    exp_res = export_model("fresh-maxtext", str(export_dir), params={"vocab_bias": custom_bias})
    assert exp_res["status"] == "exported_with_maxtext_orbax"

    inference._MODEL_CACHE.clear()
    inference._PARAMS_ONLY_CACHE.clear()

    reloaded_via_kwarg = generate_sql(
        "fresh-maxtext",
        "SELECT",
        beam_width=1,
        max_length=3,
        checkpoint_path=str(export_dir),
    )
    reloaded_via_path = generate_sql(
        str(export_dir),
        "SELECT",
        beam_width=1,
        max_length=3,
    )
    assert reloaded_via_kwarg["sql"] == reloaded_via_path["sql"]
    assert "SSS" in reloaded_via_kwarg["sql"]
    assert "XXX" not in reloaded_via_kwarg["sql"]

    # 3. Also verify checkpoints saved by `train.save_maxtext_checkpoint`
    # (`{"params": ..., "opt_state": ...}`) unwrap cleanly
    trained_ckpt_dir = tmp_path / "trained_ckpt"
    save_maxtext_checkpoint(
        trained_ckpt_dir, step=3, params={"vocab_bias": custom_bias}, opt_state={"step": jnp.array(3)}
    )
    restored_trained = load_maxtext_orbax_params(trained_ckpt_dir)
    assert restored_trained is not None
    assert "vocab_bias" in restored_trained

    # Non-existent directory and empty step directory return `None`
    assert load_maxtext_orbax_params(tmp_path / "does_not_exist") is None
    empty_step_dir = tmp_path / "empty_orbax" / "0"
    empty_step_dir.mkdir(parents=True)
    assert inference.ocp is not None
    with patch.object(inference.ocp.CheckpointManager, "latest_step", return_value=None):
        assert load_maxtext_orbax_params(tmp_path / "empty_orbax") is None


def test_generate_sql_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `generate_sql` returns structured results, captures runtime errors, and checks dependencies."""

    class _StubConstructor:
        """Minimal constructor stub."""

        def __init__(self, _name: str) -> None:
            """No-op init."""

        def __call__(self, seq: jax.Array) -> jax.Array:
            """Return uniform logits."""
            return jnp.ones((1, seq.shape[1], 16), dtype=jnp.float32)

    assert _StubConstructor("m")(jnp.array([[1, 2]], dtype=jnp.int32)).shape == (1, 2, 16)
    monkeypatch.setattr(inference, "Gemma4Model", _StubConstructor)
    with patch("gemma_4_sql.backends.maxtext.inference._execute_generate", return_value=("success", "SELECT 1", 0.9)):
        result = generate_sql("test-model", "test prompt")
        assert result["sql"] == "SELECT 1"
        assert result["status"] == "success"
        assert result["confidence_score"] == 0.9

    with patch(
        "gemma_4_sql.backends.maxtext.inference._execute_generate", side_effect=RuntimeError("Generation failed")
    ):
        err_res = generate_sql("test-model", "test prompt")
        assert err_res["sql"] == ""
        assert "failed: Generation failed" in err_res["status"]

    monkeypatch.setattr(inference, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        generate_sql("test-model", "test prompt")
