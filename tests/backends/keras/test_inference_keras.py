"""Unit and behavioral tests for Keras inference, beam sampling, and confidence scoring without MagicMock."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable

import pytest

import gemma_4_sql.backends.keras.inference as inf
from gemma_4_sql._types_models import KerasSamplerProtocol
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

    class _ScoreTensorStub:
        """Typed `_ToListOnlyScoreProtocol` score tensor stub."""

        def tolist(self) -> list[float]:
            """Return `[0.1, 0.2]`."""
            return [0.1, 0.2]

    assert inf.compute_keras_confidence(_ScoreTensorStub(), 2) == pytest.approx(0.15)
    assert inf.compute_keras_confidence([-1.0, -2.0], 2) == pytest.approx(math.exp(-1.5))
    assert inf.compute_keras_confidence(0.5, 2) == 0.5
    assert inf.compute_keras_confidence(-1.0, 2) == pytest.approx(math.exp(-0.5))
    assert inf.compute_keras_confidence([], 2) == 1.0
    assert inf.compute_keras_confidence(None, 10) == 1.0


def test_configure_beam_sampler() -> None:
    """Verify configure_beam_sampler attaches BeamSampler via model.compile or model.sampler."""
    from gemma_4_sql.type_hints import ForwardArg, TensorProtocol, TfTensorProtocol

    class _FakeBeamSampler:
        """Callable `KerasSamplerProtocol` test double."""

        def compute_next_token(
            self,
            probabilities: TfTensorProtocol | TensorProtocol,
        ) -> TfTensorProtocol | TensorProtocol:
            """Return `probabilities` unchanged.

            Returns:
                Unmodified `probabilities` tensor.

            """
            return probabilities

        def __call__(self, *_a: ForwardArg, **_kw: ForwardArg) -> TfTensorProtocol | TensorProtocol:
            """Sample next token.

            Returns:
                Tensor token IDs from `probabilities` or `ops.zeros((1, 1))`.

            """
            from tests.utils.tensor_helpers import get_typed_keras_ops

            return get_typed_keras_ops().zeros((1, 1), dtype="int32")

    sampler_obj = _FakeBeamSampler()
    tok_out = sampler_obj()
    assert sampler_obj.compute_next_token(tok_out) is tok_out
    sampler_ret: _FakeBeamSampler | str = sampler_obj
    raise_sampler_err = False

    from unittest.mock import patch

    def _make_beam_sampler(*, num_beams: int = 1) -> _FakeBeamSampler | str:
        """Return `sampler_ret` or raise `ValueError`.

        Raises:
            ValueError: When `raise_sampler_err` is enabled.

        """
        _ = num_beams
        if raise_sampler_err:
            raise ValueError("invalid beam width")
        return sampler_ret

    class _FakeSamplersModule:
        """Typed `keras_nlp.samplers` holder."""

        BeamSampler: Callable[..., _FakeBeamSampler | str] | None = staticmethod(_make_beam_sampler)

    class _FakeKerasNlpPkg:
        """Typed `keras_nlp` module double."""

        samplers = _FakeSamplersModule

    with patch.dict(sys.modules, {"keras_nlp": _FakeKerasNlpPkg}):

        class _CompileCapableModel:
            """Model supporting `.compile(sampler=...)`."""

            def __init__(self, *, fail_compile: bool = False) -> None:
                """Initialize model."""
                self.compiled_sampler: KerasSamplerProtocol | None = None
                self.sampler: KerasSamplerProtocol | None = None
                self.fail_compile = fail_compile

            def compile(self, *, sampler: KerasSamplerProtocol) -> None:
                """Record `sampler` or raise `TypeError`.

                Raises:
                    TypeError: If `fail_compile` is enabled.

                """
                if self.fail_compile:
                    raise TypeError("compile rejected sampler")
                self.compiled_sampler = sampler

        model1 = _CompileCapableModel()
        inf.configure_beam_sampler(model1, 2)
        assert model1.compiled_sampler is sampler_obj

        model_fallback = _CompileCapableModel(fail_compile=True)
        inf.configure_beam_sampler(model_fallback, 2)
        assert model_fallback.sampler is sampler_obj

        inf.configure_beam_sampler("no_sampler_attr", 2)

        sampler_ret = "non_callable_invalid_sampler"
        assert inf.configure_beam_sampler("no_sampler_attr", 2) is None

        raise_sampler_err = True
        assert inf.configure_beam_sampler("no_sampler_attr", 2) is None
        raise_sampler_err = False

        _FakeSamplersModule.BeamSampler = None
        assert inf.configure_beam_sampler("no_sampler_attr", 2) is None

    with patch.dict(sys.modules, {"keras_nlp": None}):
        assert inf.configure_beam_sampler("no_sampler_attr", 2) is None


def test_generate_sql_and_output_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql handles string, dict, tuple, and integer generation outputs and missing dependencies."""
    real_keras = inf.keras
    monkeypatch.setattr(inf, "keras", None)
    monkeypatch.setattr(inf, "tf", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        inf.generate_sql("m", "p")

    def _noop_beam_sampler(_m: inf.KerasCausalLMProtocol | str | None, _bw: int) -> None:
        """No-op beam sampler configurator."""

    monkeypatch.setattr(inf, "keras", real_keras)
    monkeypatch.setattr(inf, "tf", real_keras)
    monkeypatch.setattr(inf, "configure_beam_sampler", _noop_beam_sampler)
    inf._MODEL_CACHE.clear_cache()

    def setup_stub(
        ret_val: str | int | dict[str, str | float] | tuple[str, float],
        *,
        with_last_scores: bool = False,
    ) -> None:
        """Install a typed GemmaCausalLM double returning `ret_val` on `.generate()`."""
        inf._MODEL_CACHE.clear_cache()

        class _StubKerasLM:
            """Minimal Keras causal LM double."""

            def generate(
                self, _prompt: str, *, max_length: int = 50
            ) -> str | int | dict[str, str | float] | tuple[str, float]:
                """Return `ret_val`."""
                _ = max_length
                return ret_val

        class _StubKerasLMWithScores(_StubKerasLM):
            """Keras causal LM double also exposing `last_scores`."""

            last_scores = 0.75

        class _StubLoader:
            """Typed `GemmaCausalLM` loader double."""

            @staticmethod
            def from_preset(_name: str) -> _StubKerasLM:
                """Return `_StubKerasLMWithScores` or `_StubKerasLM`."""
                return _StubKerasLMWithScores() if with_last_scores else _StubKerasLM()

        monkeypatch.setattr(inf, "GemmaCausalLM", _StubLoader)

    setup_stub("prompt SELECT 1")
    res = inf.generate_sql("m", "prompt ")
    assert res["status"] == "success"
    assert res["sql"] == "SELECT 1"

    setup_stub("prompt SELECT 2", with_last_scores=True)
    res_scores = inf.generate_sql("m", "prompt ")
    assert res_scores["status"] == "success"
    assert res_scores["confidence_score"] == pytest.approx(0.75)

    setup_stub({"text": "prompt SEL", "scores": 0.5})
    assert inf.generate_sql("m", "prompt ")["status"] == "success"

    setup_stub(("prompt SEL", 0.5))
    assert inf.generate_sql("m", "prompt ")["status"] == "success"

    setup_stub("prompt ")
    assert "failed" in inf.generate_sql("m", "prompt ")["status"]

    setup_stub(12345)
    res_num = inf.generate_sql("m", "")
    assert res_num["status"] == "success"
    assert "12345" in res_num["sql"]

    def _none_attr(*_a: str | type) -> None:
        """Return None for import_optional_symbol."""

    monkeypatch.setattr(inf, "GemmaCausalLM", None)
    monkeypatch.setattr(inf, "import_optional_symbol", _none_attr)
    res_missing_cls = inf.generate_sql("m", "SELECT")
    assert res_missing_cls["status"].startswith("failed:")
    assert "keras_nlp.models.GemmaCausalLM is not available" in res_missing_cls["status"]
