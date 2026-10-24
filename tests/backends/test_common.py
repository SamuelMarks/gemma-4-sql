"""Tests for shared common_gemma4_config, common_inference, and common_peft utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gemma_4_sql.backends.common_gemma4_config import (
    get_gemma4_preset_spec,
    resolve_gemma4_preset_name,
)
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    compute_jax_step_probs,
    expand_jax_beams,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.exceptions import ConfigurationError, InferenceError, TrainingError


def test_common_gemma4_config_presets() -> None:
    """Verify resolve_gemma4_preset_name and get_gemma4_preset_spec resolve all Gemma 4 variants."""
    assert resolve_gemma4_preset_name("google/gemma-4-E2B") == "e2b"
    assert resolve_gemma4_preset_name("google/gemma-4-E4B") == "e4b"
    assert resolve_gemma4_preset_name("google/gemma-4-26B-A4B") == "26b_a4b"
    assert resolve_gemma4_preset_name("google/gemma-4-31B") == "31b"
    assert resolve_gemma4_preset_name("gemma-4-tiny") == "tiny"

    assert get_gemma4_preset_spec("gemma-4-e2b")["hidden_size"] == 1024
    assert get_gemma4_preset_spec("gemma-4-e4b")["hidden_size"] == 2560
    assert get_gemma4_preset_spec("gemma-4-26b")["num_experts"] == 128
    assert get_gemma4_preset_spec("gemma-4-31b")["hidden_size"] == 5376
    assert get_gemma4_preset_spec("gemma-4-tiny")["hidden_size"] == 64


def test_common_inference_and_peft_utilities() -> None:
    """Verify BoundedLRUCache, normalize_logprob_confidence, run_array_beam_search, and apply_lora_wrapper."""
    cache: BoundedLRUCache[int] = BoundedLRUCache(maxsize=2)
    assert cache.get_model("missing") is None
    cache.put_model("a", 1)
    cache.put_model("b", 2)
    cache.put_model("a", 10)
    assert cache.get_model("a") == 10
    cache.put_model("c", 3)
    assert cache.get_model("b") is None
    assert cache.get_model("c") == 3

    assert normalize_logprob_confidence([-0.1, -0.2], 0) == 0.0
    assert 0.0 < normalize_logprob_confidence([-0.1, -0.2], 2) <= 1.0

    init_seq = np.array([[1, 2]])

    def _expand(seq: Any, score: float, _bw: int) -> list[tuple[Any, float]]:
        """Expand beam with one EOS candidate and one non-EOS candidate."""
        return [
            (np.concatenate([seq, [[99]]], axis=1), score - 0.1),
            (np.concatenate([seq, [[3]]], axis=1), score - 0.5),
        ]

    best_seq, best_score = run_array_beam_search(init_seq, beam_width=2, max_length=2, eos_token_id=99, step_expand_fn=_expand)
    assert best_seq[0, -1] == 99
    assert best_score == pytest.approx(-0.1)

    eos_only_seq = np.array([[1, 99]])
    best_eos, _ = run_array_beam_search(eos_only_seq, beam_width=1, max_length=2, eos_token_id=99, step_expand_fn=_expand)
    assert best_eos[0, -1] == 99

    best_zero_len, _ = run_array_beam_search(init_seq, beam_width=1, max_length=0, eos_token_id=99, step_expand_fn=_expand)
    assert best_zero_len.shape == (1, 2)

    res_ok = apply_lora_wrapper("jax", "m", ["q_proj"], 8, 16, 0.05, lambda: 4)
    assert res_ok["injected_modules"] == 4
    assert res_ok["status"] == "completed"

    def _fail() -> int:
        """Raise ValueError for testing error handling."""
        raise ValueError("lora boom")

    res_fail = apply_lora_wrapper("jax", "m", ["q_proj"], 8, 16, 0.05, _fail)
    assert res_fail["status"] == "failed: lora boom"

    with pytest.raises(TrainingError, match="lora boom"):
        apply_lora_wrapper("jax", "m", ["q_proj"], 8, 16, 0.05, _fail, raise_on_error=True)

    def _cfg_err() -> int:
        """Raise ConfigurationError to verify passthrough."""
        raise ConfigurationError("bad lora r")

    with pytest.raises(ConfigurationError, match="bad lora r"):
        apply_lora_wrapper("jax", "m", ["q_proj"], 0, 16, 0.05, _cfg_err)

    class FakeJaxNN:
        """Stub jax.nn namespace."""

        @staticmethod
        def log_softmax(x: Any, axis: int = -1) -> Any:
            """Return input array unchanged."""
            _ = axis
            return x

    class FakeJax:
        """Stub jax namespace."""

        nn = FakeJaxNN

    for logits in (
        np.array([[[0.1, 0.9, 0.5]]]),
        np.array([[0.1, 0.9, 0.5]]),
        np.array([0.1, 0.9, 0.5]),
    ):
        top_idx, top_p = compute_jax_step_probs(logits, 2, FakeJax, np)
        assert list(top_idx) == [1, 2]
        beams = expand_jax_beams(np.array([[10]]), 0.0, top_idx, top_p, 2, np)
        assert len(beams) == 2

    res_inf_ok = run_inference_wrapper("jax", "m", "p", 2, lambda: ("SELECT 1;", 0.9))
    assert res_inf_ok["status"] == "success"
    assert res_inf_ok["sql"] == "SELECT 1;"

    res_inf_fail = run_inference_wrapper("jax", "m", "p", 2, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert res_inf_fail["status"] == "failed: boom"

    with pytest.raises(InferenceError, match="boom"):
        run_inference_wrapper("jax", "m", "p", 2, lambda: (_ for _ in ()).throw(RuntimeError("boom")), raise_on_error=True)

    with pytest.raises(InferenceError, match="direct err"):
        run_inference_wrapper(
            "jax",
            "m",
            "p",
            2,
            lambda: (_ for _ in ()).throw(InferenceError("direct err")),
            raise_on_error=True,
        )
