"""Behavioral and unit tests for MLX model quantization (INT8, INT4, AWQ, GPTQ) without MagicMock."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from unittest.mock import patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import MlxArrayProtocol
from tests.conftest import requires_mlx
from tests.utils.tensor_helpers import (
    TypedMlxSequentialProtocol,
    get_typed_mlx_lm,
    get_typed_mlx_nn,
    get_typed_mx,
    mx_eval,
)

pytestmark = requires_mlx


def test_mlx_quantize_imports_and_fallback_base() -> None:
    """Verify mlx.quantize gracefully sets optional imports and fallback MLXAWQLinear when unavailable."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    mx = get_typed_mx()
    mlx_nn = get_typed_mlx_nn()

    lin = mlx_nn.Linear(4, 4)
    scales = mx.ones((4,)) * 2.0

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None}):
        importlib.reload(quantize_module)
        assert quantize_module.mlx is None
        assert quantize_module._get_native_mlx_awq_cls() is None
        fb = quantize_module.MLXAWQLinear(lin, scales)
        res_fb = fb(mx.ones((1, 4)) * 4.0)
        assert float(res_fb.sum()) == pytest.approx(8.0)
    importlib.reload(quantize_module)


def test_calibrate_awq_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibrate_awq_scales computes grid-searched per-channel scales and handles missing mlx."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    mx = get_typed_mx()

    with pytest.raises(ValueError, match="alpha_range"):
        quantize_module.calibrate_awq_scales([[1.0]], [[1.0]], alpha_range=[])

    class _ShapeOnlyStub:
        """Typed `_ShapeHolderProtocol` stub."""

        shape = (2, 10)

    monkeypatch.setattr(quantize_module, "mlx", None)
    res = quantize_module.calibrate_awq_scales(_ShapeOnlyStub(), None)
    assert res == [1.0] * 10
    assert quantize_module.calibrate_awq_scales("no_shape", None) == [1.0]
    assert quantize_module._collect_embedding_activations(None, ["SELECT 1"], 4) is None

    monkeypatch.setattr(quantize_module, "mlx", mx)
    mx.random.seed(42)
    w = mx.random.normal((10, 5))
    acts = mx.random.normal((100, 5))

    scales = quantize_module.calibrate_awq_scales(w, acts)
    assert isinstance(scales, MlxArrayProtocol)
    assert scales.shape == (5,)

    acts_1d = mx.random.normal((5,))
    scales2 = quantize_module.calibrate_awq_scales(w, acts_1d)
    assert isinstance(scales2, MlxArrayProtocol)
    assert scales2.shape == (5,)
    scales_none = quantize_module.calibrate_awq_scales(w, None)
    assert isinstance(scales_none, MlxArrayProtocol)
    assert scales_none.shape == (5,)


def test_calibrate_gptq_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibrate_gptq_weights computes Hessian-compensated weights and handles missing mlx."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    mx = get_typed_mx()

    monkeypatch.setattr(quantize_module, "mlx", None)
    res = quantize_module.calibrate_gptq_weights("w", "a")
    assert res == "w"

    monkeypatch.setattr(quantize_module, "mlx", mx)
    mx.random.seed(42)
    w = mx.random.normal((10, 5))
    acts = mx.random.normal((100, 5))

    w_q = quantize_module.calibrate_gptq_weights(w, acts)
    assert isinstance(w_q, MlxArrayProtocol)
    assert w_q.shape == (10, 5)

    acts_1d = mx.random.normal((5,))
    w_q2 = quantize_module.calibrate_gptq_weights(w, acts_1d)
    assert isinstance(w_q2, MlxArrayProtocol)
    assert w_q2.shape == (10, 5)
    w_q_none = quantize_module.calibrate_gptq_weights(w, None)
    assert isinstance(w_q_none, MlxArrayProtocol)
    assert w_q_none.shape == (10, 5)


def test_quantize_model_real_mlx_with_awq_forward_preservation() -> None:
    """Verify quantize_model quantizes real MLX models and MLXAWQLinear preserves forward outputs via X / awq_scales."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module
    from gemma_4_sql.type_hints import get_quantized_artifact

    mx = get_typed_mx()
    mlx_nn = get_typed_mlx_nn()

    def _build_calib_model() -> TypedMlxSequentialProtocol:
        """Construct an MLX `Sequential` model with `Embedding(128, 64)` and `Linear(64, 64)`.

        Returns:
            Initialized `TypedMlxSequentialProtocol` container.
        """
        return mlx_nn.Sequential(mlx_nn.Embedding(128, 64), mlx_nn.Linear(64, 64))

    # INT8 real quantization
    m_int8 = mlx_nn.Sequential(mlx_nn.Linear(64, 64))
    assert m_int8(mx.ones((1, 64))).shape == (1, 64)
    res_int8 = quantize_module.quantize_model("tiny-mlx", "int8", model=m_int8, group_size=64)
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == 0.5
    q_int8 = m_int8.layers[0]
    assert isinstance(q_int8, mlx_nn.QuantizedLinear)
    assert q_int8.scales.shape == (64, 1)

    # INT4 real quantization
    m_int4 = mlx_nn.Sequential(mlx_nn.Linear(64, 64))
    res_int4 = quantize_module.quantize_model("tiny-mlx", "int4", model=m_int4, group_size=32)
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == 0.75
    q_int4 = m_int4.layers[0]
    assert isinstance(q_int4, mlx_nn.QuantizedLinear)
    assert q_int4.scales.shape == (64, 2)

    # AWQ real quantization with real Embedding calibration pass AND forward-output verification!
    mx.random.seed(7)
    m_awq = _build_calib_model()
    x_probe = mx.random.normal((2, 64))
    y_before_awq = m_awq.layers[1](x_probe)
    mx_eval(y_before_awq)

    res_awq = quantize_module.quantize_model(
        "tiny-mlx",
        "awq",
        model=m_awq,
        calib_data=["SELECT * FROM users;"],
        group_size=32,
    )
    assert res_awq["status"] == "quantized_awq"
    proj_after = m_awq.layers[1]
    assert isinstance(proj_after, mlx_nn.QuantizedLinear)
    assert isinstance(proj_after, quantize_module.MLXAWQLinear)
    y_after_awq = proj_after(x_probe)
    mx_eval(y_after_awq)
    assert float(mx.max(mx.abs(y_after_awq - y_before_awq))) < 0.35

    awq_artifact = get_quantized_artifact(m_awq)
    assert awq_artifact is not None
    assert awq_artifact.method == "awq"
    assert "layer_0" in awq_artifact.scales

    # GPTQ real quantization with Hessian-compensated weights
    m_gptq = _build_calib_model()
    res_gptq = quantize_module.quantize_model(
        "tiny-mlx", "gptq", model=m_gptq, calib_data=["SELECT 1;"], group_size="32"
    )
    assert res_gptq["status"] == "quantized_gptq"
    assert isinstance(m_gptq.layers[1], mlx_nn.QuantizedLinear)
    gptq_artifact = get_quantized_artifact(m_gptq)
    assert gptq_artifact is not None
    assert gptq_artifact.method == "gptq"

    # Direct weight attribute extraction on a single Linear layer, non-dict weight layer, 1D weight, and string
    direct_lin = mlx_nn.Linear(32, 32, bias=False)
    assert quantize_module._extract_first_2d_weight(direct_lin) is not None
    emb_1d = mlx_nn.RMSNorm(32)
    assert quantize_module._get_layer_weight(emb_1d) is None
    assert quantize_module._extract_first_2d_weight("no_weights") is None

    class _AttrWeightLayer:
        """Non-dict `_WeightLayerProtocol` exposing 2D `.weight`."""

        def __init__(self) -> None:
            """Initialize 2D weight."""
            self.weight = mx.ones((4, 4))

    class _Attr1DWeightLayer:
        """Non-dict `_WeightLayerProtocol` exposing 1D `.weight`."""

        def __init__(self) -> None:
            """Initialize 1D weight."""
            self.weight = mx.ones((4,))

    assert quantize_module._get_layer_weight(_AttrWeightLayer()) is not None
    assert quantize_module._get_layer_weight(_Attr1DWeightLayer()) is None
    assert quantize_module._collect_embedding_activations(m_awq, ["SELECT 1"], 999) is None
    with patch.object(quantize_module.SQLTokenizer, "encode", return_value=[]):
        assert quantize_module._collect_embedding_activations(_build_calib_model(), ["SELECT 1"], 64) is None

    with patch.object(mlx_nn, "quantize"):
        res_gptq_attr = quantize_module.quantize_model("tiny-mlx", "gptq", model=_AttrWeightLayer())
        assert res_gptq_attr["status"] == "quantized_gptq"

    # Test _replace_submodule_at_path across empty path, nested dict/list, non-module values,
    # and _UpdateModulesModelProtocol.
    quantize_module._replace_submodule_at_path(
        None, "", quantize_module.MLXAWQLinear(direct_lin, mx.ones((32,)), group_size=32)
    )
    awq_inst = quantize_module.MLXAWQLinear(direct_lin, mx.ones((32,)), group_size=32)
    quantize_module._replace_submodule_at_path("non_mutable", "leaf", awq_inst)
    quantize_module._replace_submodule_at_path({"a": "plain_str"}, "a.b", awq_inst)
    quantize_module._replace_submodule_at_path({"a": "plain_str"}, "missing.b", awq_inst)

    class _CustomUpdater:
        """Container implementing `_UpdateModulesModelProtocol`."""

        def __init__(self) -> None:
            """Initialize updated dict."""
            self.updated: Mapping[str, quantize_module.MLXAWQLinear] | None = None

        def update_modules(self, modules: Mapping[str, quantize_module.MLXAWQLinear]) -> None:
            """Record `modules`."""
            self.updated = modules

    updater = _CustomUpdater()
    inner_list: list[quantize_module._SubmoduleNodeType | quantize_module.MLXAWQLinear] = [updater, updater]
    nested_list_dict: dict[str, quantize_module._SubmoduleNodeType | quantize_module.MLXAWQLinear] = {
        "blocks": inner_list
    }
    quantize_module._replace_submodule_at_path(nested_list_dict, "blocks.0.proj", awq_inst)
    quantize_module._replace_submodule_at_path(nested_list_dict, "blocks.1", awq_inst)
    assert updater.updated is not None and "proj" in updater.updated
    assert inner_list[1] is awq_inst


def test_quantize_model_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_model error handling, unsupported methods, model loading, and missing dependencies."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    mlx_nn = get_typed_mlx_nn()
    mlx_lm = get_typed_mlx_lm()

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_module.quantize_model("model", "invalid", raise_on_error=True)

    class SlottedModel:
        """Model with __slots__ returning empty named_modules."""

        __slots__ = ()

        def named_modules(self) -> list[tuple[str, quantize_module._WeightLayerProtocol]]:
            """Return empty module list."""
            return []

    with patch.object(mlx_nn, "quantize"):
        res_slot_awq = quantize_module.quantize_model("model", "awq", model=SlottedModel())
        assert res_slot_awq["status"] == "quantized_awq"
        res_slot_gptq = quantize_module.quantize_model("model", "gptq", model=SlottedModel())
        assert res_slot_gptq["status"] == "quantized_gptq"
        res_bool_arg = quantize_module.quantize_model("model", "int8", model=True)
        assert res_bool_arg["status"] == "quantized_int8"

    with (
        patch.object(mlx_lm, "load", return_value=(SlottedModel(), "tok")),
        patch.object(mlx_nn, "quantize"),
    ):
        res_loaded = quantize_module.quantize_model("model", "int8")
        assert res_loaded["status"] == "quantized_int8"

    with patch.object(mlx_lm, "load", side_effect=RuntimeError("load fail")):
        res_fail = quantize_module.quantize_model("model", "int8")
        assert "failed:" in res_fail["status"]

    # Missing nn.quantize attribute via custom mlx.nn module stub without .quantize
    from gemma_4_sql.backends.lazy_loader import LazyLoader, override_lazy_module
    from tests.utils.test_utils import FakeModuleNamespace

    _ = LazyLoader("mlx.nn")
    nn_without_quantize = FakeModuleNamespace("mlx.nn")
    with override_lazy_module("mlx.nn", nn_without_quantize):
        res_no_q = quantize_module.quantize_model("model", "int8", model=SlottedModel())
        assert "failed: MLX quantization failed: mlx.nn.quantize is not available" in res_no_q["status"]

    _ = LazyLoader("mlx_lm")
    with override_lazy_module("mlx_lm", None), pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("model", "int8", model=SlottedModel(), raise_on_error=True)

    monkeypatch.setattr(quantize_module, "mlx", None)
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("model", "int8")
