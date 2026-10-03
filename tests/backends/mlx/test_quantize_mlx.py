from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gemma_4_sql.backends.mlx import quantize
from gemma_4_sql.backends.mlx.quantize import (
    calibrate_awq_scales,
    calibrate_gptq_weights,
    quantize_model,
)
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    UnsupportedQuantizationMethodError,
)


# We patch quantize_model_wrapper so that it just calls apply_fn directly.
# This way, exceptions raised in apply_fn are propagated to our tests,
# and we can test its logic easily without testing the wrapper's exception handling.
@pytest.fixture(autouse=True)
def bypass_wrapper(monkeypatch):
    def mock_wrapper(**kwargs):
        return kwargs["apply_fn"]()

    monkeypatch.setattr(quantize, "quantize_model_wrapper", mock_wrapper)


def test_calibrate_awq_scales_empty_alpha():
    with pytest.raises(ValueError, match="alpha_range must contain at least one value"):
        calibrate_awq_scales(MagicMock(), MagicMock(), alpha_range=[])


def test_calibrate_awq_scales_no_np(monkeypatch):
    monkeypatch.setattr(quantize, "np", None)
    weight_matrix = MagicMock()
    weight_matrix.shape = (1, 10)
    result = calibrate_awq_scales(weight_matrix, MagicMock())
    assert result == [1.0] * 10


def test_calibrate_awq_scales_1d_activations():
    w = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(5).astype(np.float32)
    scales = calibrate_awq_scales(w, x)
    assert len(scales) == 5


def test_calibrate_awq_scales_2d_activations():
    w = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(3, 5).astype(np.float32)
    scales = calibrate_awq_scales(w, x)
    assert len(scales) == 5


def test_calibrate_gptq_weights_no_np(monkeypatch):
    monkeypatch.setattr(quantize, "np", None)
    w = MagicMock()
    res = calibrate_gptq_weights(w, MagicMock())
    assert res is w


def test_calibrate_gptq_weights_1d_activations():
    w = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(5).astype(np.float32)
    res = calibrate_gptq_weights(w, x)
    assert res.shape == (5, 5)


def test_calibrate_gptq_weights_2d_activations():
    w = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(3, 5).astype(np.float32)
    res = calibrate_gptq_weights(w, x)
    assert res.shape == (5, 5)


def test_calibrate_gptq_weights_singular_matrix():
    w = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(3, 5).astype(np.float32)
    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError):
        res = calibrate_gptq_weights(w, x)
    assert res.shape == (5, 5)


def test_quantize_model_no_mlx(monkeypatch):
    monkeypatch.setattr(quantize, "mlx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        quantize_model("dummy")


class MockNN:
    def quantize(self, model, group_size, bits):
        pass


@pytest.fixture
def mock_mlx_env():
    import sys

    mock_mlx = MagicMock()
    mock_mlx_lm = MagicMock()

    mock_nn = MockNN()
    mock_mlx.nn = mock_nn

    with patch.dict(sys.modules, {"mlx": mock_mlx, "mlx_lm": mock_mlx_lm, "mlx.nn": mock_nn}):
        yield mock_mlx, mock_mlx_lm, mock_nn


def test_quantize_model_unsupported_method(monkeypatch):
    monkeypatch.setattr(quantize, "mlx", MagicMock())
    with pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method"):
        quantize_model("dummy", method="invalid")


def test_quantize_model_missing_mlx_lm(monkeypatch, mock_mlx_env):
    import sys

    monkeypatch.setattr(quantize, "mlx", MagicMock())
    with patch.dict(sys.modules, {"mlx_lm": None}), pytest.raises(DependencyMissingError, match="mlx and mlx_lm are required for MLX quantization"):
        quantize_model("dummy", method="int8")


def test_quantize_model_load_fails(monkeypatch, mock_mlx_env):
    _mock_mlx, mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    mock_mlx_lm.load.side_effect = Exception("Load failed")

    with pytest.raises(RuntimeError, match="MLX quantization failed: Load failed"):
        quantize_model("dummy", method="int8")


def test_quantize_model_no_nn_quantize(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    # Remove quantize from class
    del MockNN.quantize
    try:
        with pytest.raises(RuntimeError, match="mlx.nn.quantize is not available"):
            quantize_model("dummy", method="int8", model=MagicMock())
    finally:
        # Restore for other tests
        MockNN.quantize = lambda self, model, group_size, bits: None


def test_quantize_model_awq(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    res = quantize_model("dummy", method="awq", model=MagicMock())
    assert res == (0.75, "quantized_awq")


def test_quantize_model_gptq(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    res = quantize_model("dummy", method="gptq", model=MagicMock())
    assert res == (0.75, "quantized_gptq")


def test_quantize_model_int4(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    res = quantize_model("dummy", method="int4", model=MagicMock(), group_size="32", calib_data=["a"])
    assert res == (0.75, "quantized_int4")


def test_quantize_model_int8(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    res = quantize_model("dummy", method="int8", model=MagicMock(), calib_data="not a sequence")
    assert res == (0.5, "quantized_int8")


def test_quantize_model_load_tuple(monkeypatch, mock_mlx_env):
    _mock_mlx, mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())

    res = quantize_model("dummy", method="int8")
    assert res == (0.5, "quantized_int8")


def test_quantize_model_load_not_tuple(monkeypatch, mock_mlx_env):
    _mock_mlx, mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())

    mock_mlx_lm.load.return_value = MagicMock()

    res = quantize_model("dummy", method="int8")
    assert res == (0.5, "quantized_int8")


def test_quantize_model_awq_no_np(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())
    monkeypatch.setattr(quantize, "np", None)

    res = quantize_model("dummy", method="awq", model=MagicMock())
    assert res == (0.75, "quantized_awq")


def test_quantize_model_gptq_no_np(monkeypatch, mock_mlx_env):
    _mock_mlx, _mock_mlx_lm, _mock_nn = mock_mlx_env
    monkeypatch.setattr(quantize, "mlx", MagicMock())
    monkeypatch.setattr(quantize, "np", None)

    res = quantize_model("dummy", method="gptq", model=MagicMock())
    assert res == (0.75, "quantized_gptq")
