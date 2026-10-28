"""Tests for shared common_gemma4_config, common_inference, and common_peft utilities."""

from __future__ import annotations

import types
from collections.abc import Sequence

import pytest

from gemma_4_sql.backends.common_gemma4_config import (
    GEMMA4_E2B_SPEC,
    BaseGemma4ConfigFields,
    get_gemma4_preset_spec,
    resolve_gemma4_preset_name,
)
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    JaxNNBeamNamespaceProtocol,
    call_jax_causal_lm,
    clear_all_model_caches,
    compute_jax_step_probs,
    expand_jax_beams,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_peft import (
    apply_lora_wrapper,
    build_peft_result,
    compute_lora_matrices,
    compute_lora_scale,
    matches_target_module,
)
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, InferenceError, PEFTError
from gemma_4_sql.type_hints import ForwardArg


def test_common_gemma4_config_presets() -> None:
    """Verify resolve_gemma4_preset_name and get_gemma4_preset_spec handle all preset variants."""
    assert resolve_gemma4_preset_name("google/gemma-4-tiny") == "tiny"
    assert resolve_gemma4_preset_name("google/gemma-4-31B-it") == "31b"
    assert resolve_gemma4_preset_name("google/gemma-4-26B-A4B") == "26b_a4b"
    assert resolve_gemma4_preset_name("google/gemma-4-E4B") == "e4b"
    assert resolve_gemma4_preset_name("google/gemma-4-E2B") == "e2b"
    assert resolve_gemma4_preset_name("custom-model") == "e2b"

    assert get_gemma4_preset_spec("tiny")["num_hidden_layers"] == 1
    assert get_gemma4_preset_spec("31B")["num_hidden_layers"] == 60
    assert get_gemma4_preset_spec("26B-A4B")["num_experts"] == 128
    assert get_gemma4_preset_spec("E4B")["num_hidden_layers"] == 42
    assert get_gemma4_preset_spec("E2B") == GEMMA4_E2B_SPEC
    assert "layers=24" in BaseGemma4ConfigFields().cache_fingerprint()


def test_common_inference_and_peft_utilities() -> None:
    """Verify BoundedLRUCache, make_model_cache_key, beam search, inference wrapper, and PEFT helpers."""
    cache: BoundedLRUCache[str] = BoundedLRUCache(maxsize=2)
    assert cache.get_model("missing") is None
    cache.put_model("a", "model_a")
    cache.put_model("b", "model_b")
    assert cache.get_model("a") == "model_a"
    cache.put_model("a", "model_a2")
    cache.put_model("c", "model_c")
    assert cache.get_model("b") is None
    assert cache.get_model("a") == "model_a2"
    assert cache.get_model("c") == "model_c"
    clear_all_model_caches()
    assert cache.get_model("a") is None

    class _CustomFingerprint:
        """Config stub implementing cache_fingerprint()."""

        def cache_fingerprint(self) -> str:
            """Return deterministic fingerprint."""
            return "fp-v1"

    assert make_model_cache_key("m") == "m"
    assert "jax:m:fp16" in make_model_cache_key(
        "m",
        backend_name="jax",
        dtype="fp16",
        config={"k": 1},
        factories=(str,),
    )
    assert "fp-v1" in make_model_cache_key("m", backend_name="jax", config=_CustomFingerprint())
    assert "42" in make_model_cache_key("m", backend_name="jax", config=42)

    assert normalize_logprob_confidence([-0.1, -0.2], 0) == 0.0
    assert 0.0 < normalize_logprob_confidence([-0.5, -0.5], 2) <= 1.0
    assert 0.0 < normalize_logprob_confidence(-1.0, 2) <= 1.0

    class _Seq2D:
        """Pure-Python 2D sequence stub implementing `BeamSequenceArrayProtocol`."""

        def __init__(self, tokens: list[int], shape_override: tuple[int, ...] | None = None) -> None:
            """Initialize with 1D token list exposed as shape `(1, len(tokens))`."""
            self.tokens = tokens
            self._shape = shape_override if shape_override is not None else (1, len(tokens))

        @property
        def shape(self) -> tuple[int, ...]:
            """Provide tensor dimensions."""
            return self._shape

        def __getitem__(
            self,
            idx: int | slice | types.EllipsisType | tuple[int | slice | types.EllipsisType, ...],
            /,
        ) -> _Seq2D:
            """Index into token sequence and return scalar-compatible `_Seq2D`."""
            if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[1], int):
                return _Seq2D([self.tokens[idx[1]]], shape_override=(1,))
            if isinstance(idx, int):
                return _Seq2D([self.tokens[idx]], shape_override=(1,))
            return _Seq2D(list(self.tokens), shape_override=(len(self.tokens),))

        def reshape(self, *shape: int) -> _Seq2D:
            """Reshape the array."""
            return _Seq2D(list(self.tokens), shape_override=shape)

        def __float__(self) -> float:
            """Convert first element to float."""
            return float(self.tokens[0])

        def __int__(self) -> int:
            """Convert first element to int."""
            return int(self.tokens[0])

    seq0 = _Seq2D([1, 2])

    def _expand(
        seq: _Seq2D,
        score: float,
        bw: int,
    ) -> list[tuple[_Seq2D, float]]:
        """Expand beam with EOS token 9 on first candidate."""
        return [
            (_Seq2D([*seq.tokens, 9]), score - 0.1),
            (_Seq2D([*seq.tokens, 3]), score - 0.5),
        ][:bw]

    best_seq, best_score = run_array_beam_search(seq0, beam_width=2, max_length=2, eos_token_id=9, step_expand_fn=_expand)
    assert int(best_seq[0, -1]) == 9
    assert best_score < 0.0

    # Also test natural loop exhaustion without early break when no beam hits EOS
    best_no_eos, _ = run_array_beam_search(seq0, beam_width=2, max_length=1, eos_token_id=99, step_expand_fn=_expand)
    assert int(best_no_eos[0, -1]) == 9

    def _expand_eos(
        seq: _Seq2D,
        score: float,
        _bw: int,
    ) -> list[tuple[_Seq2D, float]]:
        """Expand beam with only EOS token 9."""
        return [(_Seq2D([*seq.tokens, 9]), score - 0.1)]

    # Also test early termination when all beams hit EOS on step 1
    best_eos, _ = run_array_beam_search(
        seq0,
        beam_width=1,
        max_length=3,
        eos_token_id=9,
        step_expand_fn=_expand_eos,
    )
    assert int(best_eos[0, -1]) == 9

    class _FakeJaxNN:
        """Stub `jax.nn` namespace implementing `JaxNNBeamNamespaceProtocol`."""

        def log_softmax(self, x: _Seq2D, axis: int = -1) -> _Seq2D:
            """Return input unchanged for deterministic top-k testing."""
            _ = axis
            return x

    class _FakeJax:
        """Pure-Python stub implementing `JaxModuleBeamProtocol`."""

        @property
        def nn(self) -> JaxNNBeamNamespaceProtocol[_Seq2D]:
            """Provide the `jax.nn` submodule."""
            return _FakeJaxNN()

    class _FakeJnp:
        """Pure-Python stub implementing `JaxNumpyBeamProtocol`."""

        def argsort(self, a: _Seq2D, axis: int = -1) -> _Seq2D:
            """Return descending sorted indices."""
            _ = axis

            def _token_at(idx: int) -> int:
                """Return token value at index `idx`."""
                return a.tokens[idx]

            ranked = sorted(range(len(a.tokens)), key=_token_at)
            return _Seq2D(ranked, shape_override=(len(ranked),))

        def concatenate(self, arrays: Sequence[_Seq2D], axis: int = -1) -> _Seq2D:
            """Concatenate `_Seq2D` instances along last axis."""
            _ = axis
            merged: list[int] = []
            for arr in arrays:
                merged.extend(arr.tokens)
            return _Seq2D(merged)

    fake_jax = _FakeJax()
    fake_jnp = _FakeJnp()
    logits_3d = _Seq2D([1, 20, 5, 40], shape_override=(1, 1, 4))
    top_idx, top_p = compute_jax_step_probs(logits_3d, 2, fake_jax, fake_jnp)
    assert top_idx.shape == (4,)
    logits_2d = _Seq2D([1, 9, 2], shape_override=(1, 3))
    assert compute_jax_step_probs(logits_2d, 2, fake_jax, fake_jnp)[0].shape == (3,)
    logits_1d = _Seq2D([1, 9, 2], shape_override=(3,))
    assert compute_jax_step_probs(logits_1d, 2, fake_jax, fake_jnp)[0].shape == (3,)

    expanded = expand_jax_beams(_Seq2D([1]), 0.0, top_idx, top_p, 2, fake_jnp)
    assert len(expanded) == 2

    res_inf = run_inference_wrapper("jax", "m", "SELECT", 2, lambda: ("SELECT 1", 0.9))
    assert res_inf["status"] == "success"
    assert res_inf["sql"] == "SELECT 1"

    def _fail_inf() -> tuple[str, float]:
        """Raise ValueError for testing inference error handling."""
        raise ValueError("inf error")

    res_fail = run_inference_wrapper("jax", "m", "SELECT", 2, _fail_inf)
    assert res_fail["status"] == "failed: inf error"

    with pytest.raises(InferenceError, match="inf error"):
        run_inference_wrapper("jax", "m", "SELECT", 2, _fail_inf, raise_on_error=True)

    with pytest.raises(InferenceError, match="custom inf"):
        run_inference_wrapper(
            "jax",
            "m",
            "SELECT",
            2,
            lambda: (_ for _ in ()).throw(InferenceError("custom inf")),
            raise_on_error=True,
        )

    with pytest.raises(DependencyMissingError):
        run_inference_wrapper(
            "jax",
            "m",
            "SELECT",
            2,
            lambda: (_ for _ in ()).throw(DependencyMissingError("missing")),
        )

    assert matches_target_module("layers.0.self_attn.q_proj", ["q_proj", "v_proj"]) is True
    assert matches_target_module("layers.0.mlp.down_proj", ["q_proj", "v_proj"]) is False
    assert compute_lora_scale(16, 8) == 2.0
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        compute_lora_scale(16, 0)

    lora_a, lora_b = compute_lora_matrices(4, 6, 2)
    assert len(lora_a) == 4
    assert len(lora_a[0]) == 2
    assert len(lora_b) == 2
    assert all(v == 0.0 for row in lora_b for v in row)
    with pytest.raises(ValueError, match="Invalid LoRA dimensions"):
        compute_lora_matrices(0, 6, 2)

    peft_ok = apply_lora_wrapper("jax", "m", ["q_proj"], 4, 8, 0.0, lambda: 2)
    assert peft_ok["status"] == "completed"
    assert peft_ok["injected_modules"] == 2

    def _fail_peft() -> int:
        """Raise RuntimeError for testing PEFT error handling."""
        raise RuntimeError("peft boom")

    peft_fail = apply_lora_wrapper("jax", "m", ["q_proj"], 4, 8, 0.0, _fail_peft)
    assert peft_fail["status"] == "failed: peft boom"

    with pytest.raises(PEFTError, match="peft boom"):
        apply_lora_wrapper("jax", "m", ["q_proj"], 4, 8, 0.0, _fail_peft, raise_on_error=True)

    with pytest.raises(DependencyMissingError):
        apply_lora_wrapper(
            "jax",
            "m",
            ["q_proj"],
            4,
            8,
            0.0,
            lambda: (_ for _ in ()).throw(DependencyMissingError("no peft")),
        )

    with pytest.raises(ConfigurationError, match="lora_r must be > 0"):
        apply_lora_wrapper("jax", "m", ["q_proj"], 0, 8, 0.0, lambda: 1)

    with pytest.raises(ConfigurationError, match="lora_dropout must be in"):
        apply_lora_wrapper("jax", "m", ["q_proj"], 4, 8, 1.5, lambda: 1)

    built = build_peft_result("jax", "m", ["q_proj"], 4, 8, 0.0, "completed", 1)
    assert built["status_code"] == "completed"

    def _kw_model(inputs: ForwardArg, positions: ForwardArg = None, **kwargs: ForwardArg) -> tuple[list[ForwardArg], None]:
        """Accept inputs, positions, and arbitrary kwargs."""
        return ([inputs, positions, kwargs.get("pixel_values"), kwargs.get("audio_values"), kwargs.get("attention_mask")], None)

    def _pos_model(inputs: ForwardArg) -> tuple[ForwardArg, None]:
        """Accept only inputs."""
        return (inputs, None)

    kw_logits, kw_cache = call_jax_causal_lm(
        _kw_model,
        [1, 2],
        positions=[0, 1],
        pixel_values="px",
        audio_values="au",
        attention_mask="am",
    )
    assert kw_logits == [[1, 2], [0, 1], "px", "au", "am"]
    assert kw_cache is None

    pos_logits, pos_cache = call_jax_causal_lm(_pos_model, [1, 2], pixel_values="px")
    assert pos_logits == [1, 2]
    assert pos_cache is None
