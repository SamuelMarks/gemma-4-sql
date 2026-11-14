"""Unit and behavioral tests for Keras model quantization without MagicMock."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Never

import pytest

import gemma_4_sql.backends.keras.quantize as q
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    QuantizationError,
    UnsupportedQuantizationMethodError,
)
from gemma_4_sql.type_hints import KerasVariableProtocol
from tests.conftest import requires_keras
from tests.utils.tensor_helpers import TypedKerasModelInstanceProtocol, get_typed_keras, get_typed_keras_ops

pytestmark = requires_keras


def test_quantize_layer_weights_real_native_and_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_layer_weights converts Dense kernels to int8 / packed int4 and supports simulated mode."""
    keras = get_typed_keras()
    ops = get_typed_keras_ops()
    x = ops.ones((2, 4), dtype="float32")
    orig_w = ops.reshape(ops.linspace(-2.0, 2.0, 16), (4, 4))

    # 1. Native Keras 3 INT8 quantization: kernel variable genuinely becomes int8 + float32 kernel_scale!
    dense_int8 = keras.layers.Dense(4, use_bias=True, name="dense_int8")
    _ = dense_int8(x)
    assert isinstance(dense_int8.kernel, KerasVariableProtocol)
    dense_int8.kernel.assign(orig_w)
    out_before_8 = ops.stop_gradient(dense_int8(x))

    count_int8 = q.quantize_layer_weights(dense_int8, "int8")
    assert count_int8 == 1
    assert dense_int8.kernel.dtype == "int8"
    out_after_8 = ops.stop_gradient(dense_int8(x))
    assert bool(ops.all(ops.isclose(out_after_8, out_before_8, atol=0.1)))

    # 2. Native Keras 3 INT4 quantization: kernel variable genuinely becomes packed 4-bit int8 (shape (4, 2))!
    dense_int4 = keras.layers.Dense(4, use_bias=False, name="dense_int4")
    _ = dense_int4(x)
    assert isinstance(dense_int4.kernel, KerasVariableProtocol)
    dense_int4.kernel.assign(orig_w)
    out_before_4 = ops.stop_gradient(dense_int4(x))

    count_int4 = q.quantize_layer_weights(dense_int4, "int4")
    assert count_int4 == 1
    assert "int8" in str(dense_int4.kernel.dtype)
    assert any(tuple(w.shape) == (4, 2) for w in dense_int4.weights)
    out_after_4 = ops.stop_gradient(dense_int4(x))
    assert bool(ops.all(ops.isclose(out_after_4, out_before_4, atol=0.6)))

    # 3. Explicit simulated=True quantization-aware dequantization
    dense_sim = keras.layers.Dense(4, use_bias=False, name="dense_sim")
    _ = dense_sim(x)
    assert isinstance(dense_sim.kernel, KerasVariableProtocol)
    dense_sim.kernel.assign(orig_w)
    assert q.quantize_layer_weights(dense_sim, "int8", simulated=True) == 1
    assert "float32" in str(dense_sim.kernel.dtype)

    # 4. Custom layer stubs: native quantize NotImplementedError fallback,
    # all-zero 2D weight without .assign(), assignable Variable, 1D bias skip
    class _UnsupportedNativeLayer:
        """Layer whose `.quantize` raises `NotImplementedError`, falling back to `.weights`."""

        def __init__(self) -> None:
            """Initialize zero 2D tensor, assignable 2D Variable, and 1D vector."""
            var_cls = keras.Variable
            self.var_w = var_cls(ops.ones((2, 2), dtype="float32"), name="var_w")  # check ctor below
            self.weights = [ops.zeros((2, 2), dtype="float32"), self.var_w, ops.zeros((2,), dtype="float32")]

        def quantize(self, _mode: str) -> Never:
            """Raise `NotImplementedError`."""
            raise NotImplementedError("no native quantize")

    assert q.quantize_layer_weights(_UnsupportedNativeLayer(), "int4") == 2

    class _InvalidWeightsLayer:
        """Layer whose `.weights` entry raises inside `ops.stop_gradient`."""

        weights = ["invalid_weight"]

    assert q.quantize_layer_weights(_InvalidWeightsLayer(), "int8") == 0

    class _NoWeightsCausalStub:
        """Minimal `CausalLMProtocol` without `.quantize` or `.weights`."""

        def __call__(self, *_a: q.BackendKwargValue, **_k: q.BackendKwargValue) -> float:
            """Return `0.0`."""
            return 0.0

    no_w_stub = _NoWeightsCausalStub()
    assert no_w_stub() == 0.0
    assert q.quantize_layer_weights(no_w_stub, "int8") == 0
    assert q.quantize_layer_weights("str_input", "int8") == 0

    monkeypatch.setattr(q, "ops", None)
    assert q.quantize_layer_weights(dense_int8, "int8") == 0


def test_quantize_model_real_and_fail_fast_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify quantize_model quantizes Keras models and raises QuantizationError when preset loading fails."""
    keras = get_typed_keras()
    ops = get_typed_keras_ops()
    inp = keras.Input(shape=(4,))
    dense_proj = keras.layers.Dense(4, name="proj")
    out = dense_proj(inp)
    real_model = keras.Model(inputs=inp, outputs=out)
    orig_w = ops.reshape(ops.linspace(-1.5, 1.5, 16), (4, 4))
    assert isinstance(dense_proj.kernel, KerasVariableProtocol)
    dense_proj.kernel.assign(orig_w)

    export_dir = tmp_path / "export"
    res = q.quantize_model("m", "int8", model=real_model, export_path=str(export_dir))
    assert res["status"] == "quantized_int8"
    assert res["memory_reduction_factor"] == 0.5
    assert res.get("quantized_layers_count") == 1
    assert dense_proj.kernel.dtype == "int8"
    assert Path(str(res.get("export_path"))).exists()

    # Simulated quantization reports memory_reduction_factor == 0.0 and simulated_quantized_int8
    inp_sim = keras.Input(shape=(4,))
    dense_sim = keras.layers.Dense(4, name="proj_sim")
    sim_model = keras.Model(inputs=inp_sim, outputs=dense_sim(inp_sim))
    res_sim = q.quantize_model("m", "int8", model=sim_model, simulated=True)
    assert res_sim["status"] == "simulated_quantized_int8"
    assert res_sim["memory_reduction_factor"] == 0.0

    # Custom layer container without native .quantize also honestly reports simulated_quantized_int8 / 0.0
    class _SimOnlyLayer:
        """Layer with 2D weight tensor and no native `.quantize`."""

        def __init__(self) -> None:
            """Initialize 2D weight."""
            self.weights = [ops.ones((2, 2), dtype="float32")]

    class _SimOnlyModel:
        """Model container exposing non-native `_SimOnlyLayer`."""

        def __init__(self) -> None:
            """Initialize layers list."""
            self.layers = [_SimOnlyLayer()]

    res_custom_sim = q.quantize_model("m", "int8", model=_SimOnlyModel())
    assert res_custom_sim["status"] == "simulated_quantized_int8"
    assert res_custom_sim["memory_reduction_factor"] == 0.0

    # Model without .layers that exposes direct .quantize(mode)
    class _DirectQuantizableModel:
        """Model without `.layers` supporting direct `.quantize(mode)`."""

        def __init__(self) -> None:
            """Initialize tracking list."""
            self.modes: list[str] = []

        def quantize(self, mode: str) -> None:
            """Record `mode`."""
            self.modes.append(mode)

    direct_m = _DirectQuantizableModel()
    res_direct = q.quantize_model("m", "int4", model=direct_m, export_path=str(export_dir))
    assert res_direct["status"] == "quantized_int4"
    assert res_direct["memory_reduction_factor"] == 0.75
    assert direct_m.modes == ["int4"]

    # Unsupported method raises UnsupportedQuantizationMethodError
    with pytest.raises(UnsupportedQuantizationMethodError):
        q.quantize_model("m", "unknown")

    # Preset loading success vs FAIL-FAST QuantizationError when preset returns None, is missing, or raises ValueError!
    inp2 = keras.Input(shape=(4,))
    d2 = keras.layers.Dense(4, name="proj2")
    preset_model = keras.Model(inputs=inp2, outputs=d2(inp2))
    preset_mode = "ok"

    class _FakeGemmaCausalLMLoader:
        """Typed `GemmaCausalLM` preset loader double."""

        @staticmethod
        def from_preset(_name: str) -> TypedKerasModelInstanceProtocol | None:
            """Return `preset_model`, `None`, or raise `ValueError`.

            Returns:
                The prebuilt `preset_model` or `None`.

            Raises:
                ValueError: If `preset_mode` is `'err'`.

            """
            if preset_mode == "err":
                raise ValueError("sim")
            if preset_mode == "none":
                return None
            return preset_model

    def _return_gemma_cls(*_a: str | type) -> type[_FakeGemmaCausalLMLoader]:
        """Return `_FakeGemmaCausalLMLoader`."""
        return _FakeGemmaCausalLMLoader

    def _none_attr(*_a: str | type) -> None:
        """Return `None`."""

    monkeypatch.setattr(q, "import_optional_symbol", _return_gemma_cls)
    res_preset = q.quantize_model("m", "int4")
    assert res_preset["status"] == "quantized_int4"
    assert res_preset["memory_reduction_factor"] == 0.75

    preset_mode = "none"
    res_none = q.quantize_model("m", "int8")
    assert str(res_none["status"]).startswith("failed:")
    assert res_none["memory_reduction_factor"] == 0.0
    with pytest.raises(QuantizationError, match="returned None"):
        q.quantize_model("m", "int8", raise_on_error=True)

    monkeypatch.setattr(q, "import_optional_symbol", _none_attr)
    res_unavail = q.quantize_model("m", "int8")
    assert str(res_unavail["status"]).startswith("failed:")
    assert res_unavail["memory_reduction_factor"] == 0.0
    with pytest.raises(QuantizationError, match="unavailable"):
        q.quantize_model("m", "int8", raise_on_error=True)

    preset_mode = "err"
    monkeypatch.setattr(q, "import_optional_symbol", _return_gemma_cls)
    res_val_err = q.quantize_model("m", "int8")
    assert str(res_val_err["status"]).startswith("failed:")
    assert res_val_err["memory_reduction_factor"] == 0.0
    with pytest.raises(QuantizationError, match="Failed to load Keras model"):
        q.quantize_model("m", "int8", raise_on_error=True)

    # Keras dtype_policies fallback and bare keras stub without config/dtype_policies
    policies_set: list[str] = []

    class _DTypePoliciesStub:
        """Stub `dtype_policies` namespace."""

        @staticmethod
        def set_dtype_policy(pol: str) -> None:
            """Record `pol`."""
            policies_set.append(str(pol))

    class _KerasWithDTypePoliciesOnly:
        """Keras module stub exposing only `dtype_policies`."""

        dtype_policies = _DTypePoliciesStub()

    class _BareKerasStub:
        """Keras module stub exposing neither `config` nor `dtype_policies`."""

    monkeypatch.setattr(q, "keras", _KerasWithDTypePoliciesOnly())
    q.quantize_model("m", "int4", model=direct_m)
    assert "int4_from_float32" in policies_set

    monkeypatch.setattr(q, "keras", _BareKerasStub())
    q.quantize_model("m", "int4", model=direct_m)

    # Save failure handling
    class _FailingSaveModel:
        """Model whose `.save` raises `RuntimeError`."""

        layers: tuple[()] = ()

        def quantize(self, _mode: str) -> None:
            """No-op quantize."""

        def save(self, _path: str) -> Never:
            """Raise `RuntimeError`.

            Raises:
                RuntimeError: Always raised to simulate model save failure.

            """
            raise RuntimeError("save failed")

    res_fail = q.quantize_model("m", "int8", model=_FailingSaveModel(), export_path=str(export_dir))
    assert "failed" in str(res_fail["status"])
    assert res_fail["memory_reduction_factor"] == 0.0

    keras.config.set_dtype_policy("float32")
    monkeypatch.setattr(q, "keras", None)
    with pytest.raises(DependencyMissingError):
        q.quantize_model("m", "int8")
    _ = gc.collect()
