from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras import quantize
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError


@pytest.fixture
def mock_np():
    np_mock = MagicMock()
    np_mock.max.return_value = 10.0
    return np_mock


@pytest.fixture
def mock_keras():
    return MagicMock()


class TestQuantizeLayerWeights:
    def test_np_none(self):
        with patch.object(quantize, "np", None):
            assert quantize.quantize_layer_weights(object()) == 0

    def test_method_int8(self, mock_np):
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.return_value.ndim = 2
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 1
            w1.assign.assert_called_once()
            mock_np.clip.assert_called()
            # check call args for clip
            clip_args = mock_np.clip.call_args[0]
            assert clip_args[1] == -128
            assert clip_args[2] == 127

    def test_method_int4(self, mock_np):
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.return_value.ndim = 2
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int4")
            assert count == 1
            w1.assign.assert_called_once()
            clip_args = mock_np.clip.call_args[0]
            assert clip_args[1] == -8
            assert clip_args[2] == 7

    def test_numpy_fallback(self, mock_np):
        layer = MagicMock()
        w1 = MagicMock()
        del w1.numpy
        mock_np.array.return_value.ndim = 2
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 1
            mock_np.array.assert_called_once_with(w1)

    def test_ndim_less_than_2(self, mock_np):
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.return_value.ndim = 1
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 0
            w1.assign.assert_not_called()

    def test_max_abs_zero(self, mock_np):
        mock_np.max.return_value = 0.0
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.return_value.ndim = 2
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 1
            w1.assign.assert_called_once()

    def test_no_assign_method(self, mock_np):
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.return_value.ndim = 2
        del w1.assign
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 1

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError, TypeError, AttributeError])
    def test_exceptions_caught(self, mock_np, exc):
        layer = MagicMock()
        w1 = MagicMock()
        w1.numpy.side_effect = exc("test")
        layer.weights = [w1]

        with patch.object(quantize, "np", mock_np):
            count = quantize.quantize_layer_weights(layer, "int8")
            assert count == 0


class TestQuantizeModel:
    def test_keras_none(self):
        with patch.object(quantize, "keras", None), pytest.raises(DependencyMissingError, match="Keras dependencies are missing."):
            quantize.quantize_model("model_name")

    def test_unsupported_method(self, mock_keras):
        with patch.object(quantize, "keras", mock_keras), pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method"):
            quantize.quantize_model("model_name", method="int16")

    def test_set_dtype_policy_via_dtype_policies(self, mock_keras):
        # Setup mock_keras to have dtype_policies
        mock_keras.dtype_policies = MagicMock()
        del mock_keras.config

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("model_name", method="int8")
            mock_keras.dtype_policies.set_dtype_policy.assert_called_once_with("int8_from_float32")
            assert result["status"] == "quantized_int8"

    def test_set_dtype_policy_via_config(self, mock_keras):
        del mock_keras.dtype_policies
        mock_keras.config = MagicMock()

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("model_name", method="int8")
            mock_keras.config.set_dtype_policy.assert_called_once_with("int8_from_float32")
            assert result["status"] == "quantized_int8"

    def test_set_dtype_policy_neither(self, mock_keras):
        del mock_keras.dtype_policies
        del mock_keras.config

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("model_name", method="int8")
            assert result["status"] == "quantized_int8"

    def test_model_provided_in_kwargs(self, mock_keras):
        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]

        with patch.object(quantize, "keras", mock_keras), patch.object(quantize, "quantize_layer_weights", return_value=1) as mock_qlw:
            result = quantize.quantize_model("model_name", method="int4", model=mock_model)
            assert result["status"] == "quantized_int4"
            assert result["memory_reduction_factor"] == 0.75
            assert result["quantized_layers_count"] == 1
            mock_qlw.assert_called_once_with(mock_model.layers[0], method="int4")

    def test_model_not_provided_import_success(self, mock_keras):
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_cls.from_preset.return_value = mock_model_instance
        mock_model_instance.layers = []

        mock_keras_nlp = MagicMock()
        mock_keras_nlp.models.GemmaCausalLM = mock_model_cls

        def fake_import(name, fromlist=None):
            if name == "keras_nlp.models" and "GemmaCausalLM" in fromlist:
                return mock_keras_nlp.models
            raise ImportError(name)

        with patch.object(quantize, "keras", mock_keras), patch("builtins.__import__", side_effect=fake_import):
            result = quantize.quantize_model("preset_name")
            assert result["status"] == "quantized_int8"
            mock_model_cls.from_preset.assert_called_once_with("preset_name")

    @pytest.mark.parametrize("exc", [ImportError, ValueError, RuntimeError, AttributeError, OSError])
    def test_model_not_provided_import_fails(self, mock_keras, exc):
        def fake_import(name, fromlist=None):
            raise exc("import failed")

        with patch.object(quantize, "keras", mock_keras), patch("builtins.__import__", side_effect=fake_import):
            result = quantize.quantize_model("preset_name")
            assert result["status"] == "quantized_int8"
            assert "quantized_layers_count" not in result

    def test_export_path_provided_with_save(self, mock_keras, tmp_path):
        mock_model = MagicMock()
        mock_model.layers = []
        export_dir = tmp_path / "export"

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("my_model", method="int8", model=mock_model, export_path=str(export_dir))

            assert "export_path" in result
            expected_file = export_dir / "my_model_int8.keras"
            assert result["export_path"] == str(expected_file)
            mock_model.save.assert_called_once_with(str(expected_file))
            assert export_dir.exists()

    def test_export_path_provided_without_save(self, mock_keras, tmp_path):
        mock_model = MagicMock()
        mock_model.layers = []
        del mock_model.save
        export_dir = tmp_path / "export"

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("my_model", method="int8", model=mock_model, export_path=str(export_dir))

            assert "export_path" in result
            # directory should still be created
            assert export_dir.exists()

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError])
    def test_quantize_model_exception_caught(self, mock_keras, exc):
        # We can trigger an exception by making keras.dtype_policies.set_dtype_policy raise it
        mock_keras.dtype_policies = MagicMock()
        mock_keras.dtype_policies.set_dtype_policy.side_effect = exc("test exception")

        with patch.object(quantize, "keras", mock_keras):
            result = quantize.quantize_model("my_model")

            assert result["status"].startswith("failed:")
            assert result["memory_reduction_factor"] == 0.0
