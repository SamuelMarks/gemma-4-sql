"""Tests for mlx quantize."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError


def test_mlx_quantize_imports():
    """Test mlx quantize imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "numpy": None}):
        import gemma_4_sql.backends.mlx.quantize as quantize_module

        importlib.reload(quantize_module)
        assert quantize_module.mlx is None
        assert quantize_module.np is None
    importlib.reload(quantize_module)


def test_calibrate_awq_scales():
    """Test calibrate_awq_scales."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    with pytest.raises(ValueError):
        quantize_module.calibrate_awq_scales(MagicMock(), MagicMock(), alpha_range=[])

    quantize_module.np = None
    res = quantize_module.calibrate_awq_scales(MagicMock(shape=(2, 10)), MagicMock())
    assert res == [1.0] * 10

    import numpy as np

    quantize_module.np = np

    w = np.random.randn(10, 5)
    acts = np.random.randn(100, 5)

    scales = quantize_module.calibrate_awq_scales(w, acts)
    assert scales.shape == (5,)

    acts_1d = np.random.randn(5)
    scales2 = quantize_module.calibrate_awq_scales(w, acts_1d)
    assert scales2.shape == (5,)


def test_calibrate_gptq_weights():
    """Test calibrate_gptq_weights."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    quantize_module.np = None
    res = quantize_module.calibrate_gptq_weights("w", "a")
    assert res == "w"

    import numpy as np

    quantize_module.np = np

    w = np.random.randn(10, 5)
    acts = np.random.randn(100, 5)

    w_q = quantize_module.calibrate_gptq_weights(w, acts)
    assert w_q.shape == (10, 5)

    acts_1d = np.random.randn(5)
    w_q2 = quantize_module.calibrate_gptq_weights(w, acts_1d)
    assert w_q2.shape == (10, 5)

    # Test linalg error fallback
    with patch("numpy.linalg.inv") as mock_inv:
        mock_inv.side_effect = np.linalg.LinAlgError("error")
        quantize_module.calibrate_gptq_weights(w, acts)


def test_quantize_model():
    """Test quantize_model."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    quantize_module.mlx = MagicMock()

    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}

        res = quantize_module.quantize_model("model", "int8")
        assert res == {"status": "ok"}

        mock_wrapper.call_args[1]["apply_fn"]

        # Test unsupported method
        with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper"):  # don't care
            pass

        quantize_module.quantize_model = quantize_module.quantize_model  # refresh closures? No, apply_fn captures kwargs
        # We need to call apply_fn and see what happens inside it.
        # But wait, apply_fn uses `method`, `kwargs`, `model_name` from outer scope.


def test_quantize_model_apply_fn():
    """Test quantize apply_fn."""
    import gemma_4_sql.backends.mlx.quantize as quantize_module

    quantize_module.mlx = MagicMock()

    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        quantize_module.quantize_model("model", "invalid")
        apply_fn = mock_wrapper.call_args[1]["apply_fn"]
        with pytest.raises(UnsupportedQuantizationMethodError):
            apply_fn()

    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        mock_model = MagicMock()
        quantize_module.quantize_model("model", "int8", model=mock_model)
        apply_fn = mock_wrapper.call_args[1]["apply_fn"]

        with patch.dict(sys.modules, {"mlx_lm": MagicMock(), "mlx.nn": MagicMock()}):
            import mlx.nn as mlx_nn

            mlx_nn.quantize = MagicMock()

            red, stat = apply_fn()
            assert stat == "quantized_int8"

            # missing quantize
            del mlx_nn.quantize
            with pytest.raises(RuntimeError, match="not available"):
                apply_fn()

            # missing deps
            with patch.dict(sys.modules, {"mlx_lm": None}):
                with pytest.raises(DependencyMissingError):
                    apply_fn()

    # Test load model inside apply_fn
    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        quantize_module.quantize_model("model", "int8")
        apply_fn = mock_wrapper.call_args[1]["apply_fn"]
        with patch.dict(sys.modules, {"mlx_lm": MagicMock(), "mlx.nn": MagicMock()}):
            import mlx.nn as mlx_nn
            import mlx_lm

            mlx_nn.quantize = MagicMock()
            mlx_lm.load.return_value = ("model_obj", "tok")

            red, stat = apply_fn()
            assert stat == "quantized_int8"

            # load fails
            mlx_lm.load.side_effect = Exception("error")
            with pytest.raises(RuntimeError):
                apply_fn()

    # Test awq / gptq
    import numpy as np

    quantize_module.np = np
    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        quantize_module.quantize_model("model", "awq", model=MagicMock(), calib_data=["a"])
        apply_fn = mock_wrapper.call_args[1]["apply_fn"]
        with patch.dict(sys.modules, {"mlx_lm": MagicMock(), "mlx.nn": MagicMock()}):
            import mlx.nn as mlx_nn

            mlx_nn.quantize = MagicMock()
            with patch("gemma_4_sql.backends.mlx.quantize.calibrate_awq_scales"):
                red, stat = apply_fn()
                assert stat == "quantized_awq"

    with patch("gemma_4_sql.backends.mlx.quantize.quantize_model_wrapper") as mock_wrapper:
        quantize_module.quantize_model("model", "gptq", model=MagicMock(), group_size="32")
        apply_fn = mock_wrapper.call_args[1]["apply_fn"]
        with patch.dict(sys.modules, {"mlx_lm": MagicMock(), "mlx.nn": MagicMock()}):
            import mlx.nn as mlx_nn

            mlx_nn.quantize = MagicMock()
            with patch("gemma_4_sql.backends.mlx.quantize.calibrate_gptq_weights"):
                red, stat = apply_fn()
                assert stat == "quantized_gptq"

    quantize_module.mlx = None
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("model", "int8")
