"""Tests for shared common_gemma4_config, common_inference, and common_peft utilities."""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence

import pytest

from gemma_4_sql.backends._jax_common import (
    BeamSequenceArrayProtocol,
    JaxNNBeamNamespaceProtocol,
    build_jax_beam_step_fn,
    call_jax_causal_lm,
    compute_jax_step_probs,
    expand_jax_beams,
    gather_jax_sequence_logps,
    run_jax_beam_search_step,
)
from gemma_4_sql.backends.common_gemma4_config import (
    GEMMA4_E2B_SPEC,
    BaseGemma4ConfigFields,
    get_gemma4_preset_spec,
    resolve_gemma4_preset_name,
)
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    clear_all_model_caches,
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
    assert resolve_gemma4_preset_name("model_prod_v1") == "e2b"
    assert resolve_gemma4_preset_name("test_candidate_e4b") == "e4b"

    assert get_gemma4_preset_spec("tiny")["num_hidden_layers"] == 1
    assert get_gemma4_preset_spec("31B")["num_hidden_layers"] == 60
    assert get_gemma4_preset_spec("26B-A4B")["num_experts"] == 128
    assert get_gemma4_preset_spec("E4B")["num_hidden_layers"] == 42
    assert get_gemma4_preset_spec("E2B") == GEMMA4_E2B_SPEC
    assert "layers=24" in BaseGemma4ConfigFields().cache_fingerprint()
    assert BaseGemma4ConfigFields.from_preset("tiny").num_hidden_layers == 1
    assert BaseGemma4ConfigFields.from_preset("31B").num_hidden_layers == 60
    assert BaseGemma4ConfigFields.from_preset("26B-A4B").num_experts == 128
    assert BaseGemma4ConfigFields.from_preset("E4B").num_hidden_layers == 42
    assert BaseGemma4ConfigFields.from_preset("E2B").num_hidden_layers == 35


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
    cache.set_maxsize(1)
    assert len(cache) == 1
    clear_all_model_caches()
    assert cache.get_model("c") is None

    class _CustomFingerprint:
        """Config stub implementing cache_fingerprint()."""

        def cache_fingerprint(self) -> str:
            """Return deterministic fingerprint."""
            return "fp-v1"

    assert make_model_cache_key("m") == "m"

    def _custom_factory() -> str:
        """Return 'custom'."""
        return "custom"

    assert _custom_factory() == "custom"

    class _SlottedPresetFactory:
        """Non-weakrefable `CacheFactoryProtocol` double (`__slots__ = ()`)."""

        __slots__ = ()

        def __hash__(self) -> int:
            """Return deterministic hash."""
            return 17

        def from_preset(self, _name: str) -> None:
            """Return None."""
            return

    class _SlottedCallableFactory:
        """Non-weakrefable callable factory (`__slots__ = ()`)."""

        __slots__ = ()

        def __call__(self) -> str:
            """Return 'slotted'."""
            return "slotted"

    slotted_factory = _SlottedPresetFactory()
    slotted_callable = _SlottedCallableFactory()
    assert hash(slotted_factory) == 17
    assert slotted_factory.from_preset("m") is None
    assert slotted_callable() == "slotted"
    key1 = make_model_cache_key(
        "m",
        backend_name="jax",
        dtype="fp16",
        config={"k": 1},
        factories=(_custom_factory, "v1", 7, str, slotted_factory, slotted_factory.from_preset, slotted_callable),
    )
    key2 = make_model_cache_key(
        "m",
        backend_name="jax",
        dtype="fp16",
        config={"k": 1},
        factories=(_custom_factory, "v1", 7, str, slotted_factory, slotted_factory.from_preset, slotted_callable),
    )
    assert key1 == key2
    assert "jax:m:fp16" in key1
    assert "val=v1" in key1
    assert "val=7" in key1
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
            idx: int | slice | types.EllipsisType | _Seq2D | tuple[int | slice | types.EllipsisType, ...],
            /,
        ) -> _Seq2D:
            """Index into token sequence and return scalar-compatible `_Seq2D`.

            Returns:
                Indexed `_Seq2D` slice or scalar wrapper.

            """
            if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[1], int):
                return _Seq2D([self.tokens[idx[1]]], shape_override=(1,))
            if isinstance(idx, int):
                return _Seq2D([self.tokens[idx]], shape_override=(1,))
            return _Seq2D(list(self.tokens), shape_override=(len(self.tokens),))

        def reshape(self, *shape: int) -> _Seq2D:
            """Reshape the array.

            Returns:
                Reshaped `_Seq2D` instance.

            """
            return _Seq2D(list(self.tokens), shape_override=shape)

        def __float__(self) -> float:
            """Convert first element to float.

            Returns:
                First token value as `float`.

            """
            return float(self.tokens[0])

        def __int__(self) -> int:
            """Convert first element to int.

            Returns:
                First token value as `int`.

            """
            return int(self.tokens[0])

    seq0 = _Seq2D([1, 2])

    def _expand(
        seq: _Seq2D,
        score: float,
        bw: int,
    ) -> list[tuple[_Seq2D, float]]:
        """Expand beam with EOS token 9 on first candidate.

        Returns:
            Top `bw` `(sequence, score)` beam candidates.

        """
        return [
            (_Seq2D([*seq.tokens, 9]), score - 0.1),
            (_Seq2D([*seq.tokens, 3]), score - 0.5),
        ][:bw]

    best_seq, best_score = run_array_beam_search(
        seq0, beam_width=2, max_length=2, eos_token_id=9, step_expand_fn=_expand
    )
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
        """Expand beam with only EOS token 9.

        Returns:
            Single-element list containing the EOS-terminated beam and score.

        """
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
            """Concatenate `_Seq2D` instances along last axis.

            Returns:
                Concatenated `_Seq2D` sequence.

            """
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

    step_no_jit = run_jax_beam_search_step(_Seq2D([1]), 0.0, logits_1d, 2, fake_jax, fake_jnp)
    assert len(step_no_jit) == 2

    class _FakeJaxWithJit(_FakeJax):
        """Stub `jax` module implementing `JaxJitBeamModuleProtocol` and counting `jit` calls."""

        def __init__(self) -> None:
            """Initialize `jit_calls` counter."""
            self.jit_calls = 0

        def jit[ArrayT: BeamSequenceArrayProtocol](
            self,
            fun: Callable[[ArrayT, int], tuple[ArrayT, ArrayT]],
            *,
            static_argnums: tuple[int, ...],
        ) -> Callable[[ArrayT, int], tuple[ArrayT, ArrayT]]:
            """Count compilation and return `fun` directly.

            Returns:
                Unmodified `fun` callable.

            """
            _ = static_argnums
            self.jit_calls += 1
            return fun

    jit_jax = _FakeJaxWithJit()
    step_with_jit = run_jax_beam_search_step(_Seq2D([1]), 0.0, logits_1d, 2, jit_jax, fake_jnp)
    assert len(step_with_jit) == 2
    assert jit_jax.jit_calls == 1

    hoisted_jax = _FakeJaxWithJit()
    hoisted_step = build_jax_beam_step_fn(hoisted_jax, fake_jnp)
    for _ in range(5):
        assert len(hoisted_step(_Seq2D([1]), 0.0, logits_1d, 2)) == 2
    assert hoisted_jax.jit_calls == 1

    res_inf = run_inference_wrapper("jax", "m", "SELECT", 2, lambda: ("SELECT 1", 0.9))
    assert res_inf["status"] == "success"
    assert res_inf["sql"] == "SELECT 1"

    def _fail_inf() -> tuple[str, float]:
        """Raise ValueError for testing inference error handling.

        Raises:
            ValueError: Always raised to simulate inference failure.

        """
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

    lora_a, lora_b = compute_lora_matrices(1200, 6, 2)
    assert len(lora_a) == 1200
    assert len(lora_a[0]) == 2
    assert lora_a[0] != lora_a[1009]
    assert len(lora_b) == 2
    assert all(v == 0.0 for row in lora_b for v in row)
    with pytest.raises(ValueError, match="Invalid LoRA dimensions"):
        compute_lora_matrices(0, 6, 2)

    peft_ok = apply_lora_wrapper("jax", "m", ["q_proj"], 4, 8, 0.0, lambda: 2)
    assert peft_ok["status"] == "completed"
    assert peft_ok.get("injected_modules") == 2

    def _fail_peft() -> int:
        """Raise RuntimeError for testing PEFT error handling.

        Raises:
            RuntimeError: Always raised to simulate PEFT injection failure.

        """
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
    assert built.get("status_code") == "completed"

    def _kw_model(
        inputs: ForwardArg, positions: ForwardArg = None, **kwargs: ForwardArg
    ) -> tuple[list[ForwardArg], None]:
        """Accept inputs, positions, and arbitrary kwargs.

        Returns:
            Tuple of `(collected_args, None)`.

        """
        return (
            [inputs, positions, kwargs.get("pixel_values"), kwargs.get("audio_values"), kwargs.get("attention_mask")],
            None,
        )

    def _pos_model(inputs: ForwardArg) -> tuple[ForwardArg, None]:
        """Accept only inputs.

        Returns:
            Tuple of `(inputs, None)`.

        """
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

    # Second call exercises the cached signature traits path
    kw_logits2, _ = call_jax_causal_lm(_kw_model, [3], positions=[0])
    assert kw_logits2[0] == [3]

    pos_logits, pos_cache = call_jax_causal_lm(_pos_model, [1, 2], positions=[0], pixel_values="px")
    assert pos_logits == [1, 2]
    assert pos_cache is None

    def _buggy_model(seq: Sequence[int], positions: Sequence[int] | None = None) -> Sequence[int]:
        """Raise an internal TypeError inside the model forward body.

        Raises:
            TypeError: Always raised to simulate an internal layer bug.

        """
        _ = (seq, positions)
        raise TypeError("internal layer bug")

    with pytest.raises(TypeError, match="internal layer bug"):
        call_jax_causal_lm(_buggy_model, [1, 2], positions=[0])

    class _BoundHolder:
        """Class whose bound method is passed to `call_jax_causal_lm`."""

        def forward(self, inputs: ForwardArg, positions: ForwardArg = None) -> tuple[ForwardArg, ForwardArg]:
            """Return `(inputs, positions)`."""
            return (inputs, positions)

    holder = _BoundHolder()
    assert call_jax_causal_lm(holder.forward, [7], positions=[0]) == ([7], [0])

    class _SlottedCallable:
        """Non-weakrefable callable (`__slots__ = ()`) to exercise non-weakrefable signature inspection."""

        __slots__ = ()

        def __call__(self, inputs: ForwardArg) -> ForwardArg:
            """Return `inputs`."""
            return inputs

    assert call_jax_causal_lm(_SlottedCallable(), [9], positions=[0]) == [9]

    import jax.nn as jnn

    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

    logits = jnp.array([[[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]], dtype=jnp.float32)
    labels = jnp.array([[-100, 1, 2]], dtype=jnp.int32)
    shifted_sum = gather_jax_sequence_logps(jnn, jnp, logits, labels)
    expected_t0 = float(jnn.log_softmax(logits[0, 0])[1])
    expected_t1 = float(jnn.log_softmax(logits[0, 1])[2])
    assert float(shifted_sum[0]) == pytest.approx(expected_t0 + expected_t1, rel=1e-5)

    # Masked prompt token (-100) at shifted position is excluded
    labels_with_prompt = jnp.array([[-100, -100, 2]], dtype=jnp.int32)
    masked_sum = gather_jax_sequence_logps(jnn, jnp, logits, labels_with_prompt)
    assert float(masked_sum[0]) == pytest.approx(expected_t1, rel=1e-5)

    # Single-step (seq_len == 1) unshifted fallback branch
    logits_1 = jnp.array([[[0.0, 2.0, 0.0]]], dtype=jnp.float32)
    labels_1 = jnp.array([[1]], dtype=jnp.int32)
    assert float(gather_jax_sequence_logps(jnn, jnp, logits_1, labels_1)[0]) == pytest.approx(expected_t0, rel=1e-5)
