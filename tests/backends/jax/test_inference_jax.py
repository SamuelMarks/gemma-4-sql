"""Unit tests for JAX inference, KV-cached beam search, multimodal forwarding, and checkpoint restoration."""

from __future__ import annotations

import pathlib

import jax
import pytest
from flax import nnx

import gemma_4_sql.backends.jax.inference as inf
from gemma_4_sql.backends.common_inference import BoundedLRUCache
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.backends.jax.gemma4.cache import Cache
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from tests.conftest import requires_jax

pytestmark = requires_jax


@pytest.fixture(autouse=True)
def _bypass_jax_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass XLA jit compilation during unit tests for fast, deterministic execution."""

    def _identity_jit[FnT](fn: FnT, *_args: int, **_kwargs: tuple[int, ...]) -> FnT:
        """Return `fn` unchanged."""
        return fn

    def _fast_forward(
        model: Gemma4ForCausalLM,
        cache: Cache,
        input_ids: jax.Array,
        positions: jax.Array,
        *,
        pixel_values: jax.Array | None = None,
        image_token_mask: jax.Array | None = None,
        input_features: jax.Array | None = None,
        input_features_mask: jax.Array | None = None,
        audio_token_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, Cache]:
        """Run un-jitted forward step returning last-token logits and updated cache.

        Returns:
            A tuple of last-token logits and the updated KV cache.

        """
        logits = model(
            input_ids=input_ids,
            positions=positions,
            cache=cache,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_token_mask=audio_token_mask,
        )
        return (logits[:, -1, :], cache)

    monkeypatch.setattr(inf.jax, "jit", _identity_jit)
    monkeypatch.setattr(inf, "forward_fn", _fast_forward)


def _make_tiny_jax_model() -> Gemma4ForCausalLM:
    """Instantiate a real tiny JAX Gemma4ForCausalLM model for behavioral testing.

    Returns:
        A tiny `Gemma4ForCausalLM` instance initialized with Flax NNX RNGs.

    """
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
    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

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


def test_beam_search_and_kv_cached_beam_search_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify jax_kv_cached_beam_search matches uncached jax_beam_search token-for-token and score-for-score."""
    import jax as real_jax

    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

    model = _make_tiny_jax_model()
    input_ids = jnp.array([[10, 11]], dtype=jnp.int32)

    beams = inf._beam_search_step(input_ids, 0.0, model, beam_width=2)
    assert len(beams) == 2
    assert beams[0][0].shape == (1, 3)

    best_seq_uncached, best_score_uncached = inf.jax_beam_search(
        model,
        input_ids,
        beam_width=2,
        max_length=3,
        eos_token_id=319,
    )
    best_seq_cached, best_score_cached = inf.jax_kv_cached_beam_search(
        model,
        input_ids,
        beam_width=2,
        max_length=3,
        eos_token_id=319,
    )
    assert jnp.array_equal(best_seq_cached, best_seq_uncached)
    assert best_score_cached == pytest.approx(best_score_uncached, abs=1e-4)

    # Verify early EOS stop and finished-beam carry-over in jax_kv_cached_beam_search
    eos_seq = jnp.array([[10, 319]], dtype=jnp.int32)
    stopped_seq, stopped_score = inf.jax_kv_cached_beam_search(
        model, eos_seq, beam_width=2, max_length=3, eos_token_id=319
    )
    assert stopped_seq.shape == (1, 2)
    assert stopped_score == 0.0

    # Force first generated token of top beam to match eos_token_id so step 1 carries finished beam
    first_tok = int(best_seq_cached[0, 2])
    early_eos_seq, _ = inf.jax_kv_cached_beam_search(
        model, input_ids, beam_width=2, max_length=3, eos_token_id=first_tok
    )
    assert early_eos_seq.shape[1] >= 3

    # Also test beam_width=1 where the single beam finishes on step 0 and loop breaks at all(eos)
    greedy_one, _ = inf.jax_kv_cached_beam_search(model, input_ids, beam_width=1, max_length=1, eos_token_id=319)
    greedy_first_tok = int(greedy_one[0, 2])
    single_eos_seq, _ = inf.jax_kv_cached_beam_search(
        model, input_ids, beam_width=1, max_length=3, eos_token_id=greedy_first_tok
    )
    assert int(single_eos_seq[0, -1]) == greedy_first_tok

    # Verify fallback when jax.jit is not present on the jax namespace
    class _NoJitJaxNamespace:
        """Typed `jax` namespace without `.jit`."""

        nn = real_jax.nn

    monkeypatch.setattr(inf, "jax", _NoJitJaxNamespace())
    beams_no_jit = inf._beam_search_step(input_ids, 0.0, model, beam_width=2)
    assert len(beams_no_jit) == 2


def test_generate_sql_real_tiny_model_and_caching(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify generate_sql initializes a tiny model, caches it, and decodes SQL output via KV cache."""
    tiny_model = _make_tiny_jax_model()

    def _return_tiny(_cfg: Gemma4Config, rngs: nnx.Rngs | None = None) -> Gemma4ForCausalLM:
        """Return `tiny_model`."""
        _ = rngs
        return tiny_model

    monkeypatch.setattr(inf.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_model.config))
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", _return_tiny)

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

    res_cached = inf.generate_sql(model_key, "SELECT 1", beam_width=1, max_length=1)
    assert res_cached["status"] == "success"


def test_generate_sql_checkpoint_restore_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify generate_sql handles Orbax checkpoint restoration success, None return, and errors."""
    tiny_model = _make_tiny_jax_model()

    def _return_tiny(_cfg: Gemma4Config, rngs: nnx.Rngs | None = None) -> Gemma4ForCausalLM:
        """Return `tiny_model`."""
        _ = rngs
        return tiny_model

    def _fast_kv_beam(
        _model: Gemma4ForCausalLM,
        ids: jax.Array,
        _bw: int,
        _ml: int,
        _eos: int,
        **_kw: jax.Array | None,
    ) -> tuple[jax.Array, float]:
        """Return `(ids, -0.5)`."""
        return (ids, -0.5)

    monkeypatch.setattr(inf.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_model.config))
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", _return_tiny)
    monkeypatch.setattr(inf, "jax_kv_cached_beam_search", _fast_kv_beam)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    restore_mode = "ok"
    update_calls: list[str] = []

    import orbax.checkpoint as ocp

    class _FakeCheckpointer:
        """Typed Orbax `PyTreeCheckpointer` double."""

        def restore(self, _path: pathlib.Path) -> str | None:
            """Return state string, `None`, or raise `RuntimeError` based on `restore_mode`.

            Returns:
                The restored state sentinel or `None`.

            Raises:
                RuntimeError: If `restore_mode` is set to `'err'`.

            """
            if restore_mode == "err":
                raise RuntimeError("corrupt ckpt")
            if restore_mode == "none":
                return None
            return "restored_state"

    class _FakeNNXNamespace:
        """Typed `nnx` double recording `update` calls."""

        Rngs = nnx.Rngs

        @staticmethod
        def update(_model: Gemma4ForCausalLM, state: str) -> None:
            """Record `state` passed to `nnx.update`."""
            update_calls.append(state)

    monkeypatch.setattr(ocp, "PyTreeCheckpointer", _FakeCheckpointer)
    monkeypatch.setattr(inf, "nnx", _FakeNNXNamespace)

    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res1 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res1["status"] == "success"
    assert update_calls == ["restored_state"]

    restore_mode = "none"
    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res2 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res2["status"] == "success"

    restore_mode = "err"
    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    res3 = inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1)
    assert res3["status"] == "success"

    inf._MODEL_CACHE.pop(str(ckpt_dir), None)
    with pytest.raises(InferenceError, match="corrupt ckpt"):
        inf.generate_sql(str(ckpt_dir), "SELECT 1", beam_width=1, max_length=1, strict_checkpoint=True)

    small_cache: BoundedLRUCache[int] = inf.BoundedLRUCache(maxsize=2)
    small_cache.put_model("a", 1)
    small_cache.put_model("a", 10)
    small_cache.put_model("b", 2)
    assert small_cache.get_model("a") == 10
    small_cache.put_model("c", 3)
    assert small_cache.get_model("b") is None
    assert small_cache.get_model("c") == 3


def test_generate_sql_multimodal_and_generic_callable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql processes image/audio paths and falls back to jax_beam_search for generic callables."""
    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

    def _proc_img(_p: str | bytes) -> dict[str, list[list[float]]]:
        """Return pixel_values dict."""
        return {"pixel_values": [[1.0, 2.0]]}

    def _proc_aud(_p: str | bytes) -> dict[str, list[list[float]]]:
        """Return audio_values dict."""
        return {"audio_values": [[0.5, -0.5]]}

    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.process_image", _proc_img)
    monkeypatch.setattr("gemma_4_sql.backends.common_multimodal.process_audio", _proc_aud)

    def strict_pos_only_model(seq: jax.Array, _pos: jax.Array | None = None) -> jax.Array:
        """Model callable that rejects keyword arguments, exercising generic callable beam search.

        Returns:
            A uniform float32 logits array of shape `(1, seq_len, 320)`.

        """
        return jnp.ones((1, seq.shape[1], 320), dtype=jnp.float32)

    monkeypatch.setitem(inf._MODEL_CACHE, "strict_pos_model", strict_pos_only_model)

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
