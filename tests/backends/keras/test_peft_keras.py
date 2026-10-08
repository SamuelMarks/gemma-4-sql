"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.peft import (
    KerasLoRADense,
    apply_lora,
    count_parameters,
    inject_lora,
    merge_lora_weights,
)
from gemma_4_sql.exceptions import DependencyMissingError


class DummyOps:
    """Docstring for DummyOps."""

    def matmul(self, a, b):
        """Docstring for matmul."""

        class MockTensor:
            """Docstring for MockTensor."""

            def __add__(self, other):
                """Docstring for __add__."""
                return self

            def __radd__(self, other):
                """Docstring for __radd__."""
                return self

            def __mul__(self, other):
                """Docstring for __mul__."""
                return self

            def __rmul__(self, other):
                """Docstring for __rmul__."""
                return self

        return MockTensor()

    def size(self, w):
        """Docstring for size."""
        return len(w)


def get_mock_keras_ops():
    """Docstring for get_mock_keras_ops."""
    mock_keras = MagicMock()
    mock_keras.ops = DummyOps()

    class DummyDropout:
        """Docstring for DummyDropout."""

        def __init__(self, rate):
            """Docstring for __init__."""
            self.rate = rate

        def __call__(self, x, training=None):
            """Docstring for __call__."""
            return x

    mock_keras.layers.Dropout = DummyDropout

    class Layer:
        """Docstring for Layer."""

        def __init__(self, **kwargs):
            """Docstring for __init__."""

        def build(self, input_shape):
            """Docstring for build."""

    class Dense(Layer):
        """Docstring for Dense."""

        def __init__(self, kernel, bias=None, name=""):
            """Docstring for __init__."""
            self.kernel = kernel
            self.bias = bias
            self.name = name
            self.trainable = True

        def __call__(self, inputs):
            """Docstring for __call__."""
            return inputs

    mock_keras.layers.Layer = Layer
    mock_keras.layers.Dense = Dense

    return mock_keras, DummyOps()


def test_keras_lora_dense_missing_keras():
    """Docstring for test_keras_lora_dense_missing_keras."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        KerasLoRADense(dense=MagicMock())


def test_keras_lora_dense_invalid_rank():
    """Docstring for test_keras_lora_dense_invalid_rank."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops), pytest.raises(ValueError):
        KerasLoRADense(dense=MagicMock(), r=0)


def test_keras_lora_dense_build_and_call():
    """Docstring for test_keras_lora_dense_build_and_call."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel, bias=1)

        lora = KerasLoRADense(dense=dense, r=2, lora_alpha=4.0, lora_dropout=0.1)

        def add_weight_mock(shape, initializer, trainable, name):
            """Docstring for add_weight_mock."""
            return shape

        lora.add_weight = add_weight_mock

        lora.build((None, 4))
        lora.build((None, 4))  # Test already built

        assert lora.kernel == mock_kernel
        assert lora.bias == 1
        assert lora.W == mock_kernel
        assert lora.A == (4, 2)
        assert lora.B == (2, 4)

        out = lora.call(10.0, training=True)
        assert out is not None


def test_keras_lora_dense_no_dropout():
    """Docstring for test_keras_lora_dense_no_dropout."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel)

        lora = KerasLoRADense(dense=dense, r=2, lora_dropout=0.0)
        assert lora.dropout is None

        def add_weight_mock(shape, initializer, trainable, name):
            """Docstring for add_weight_mock."""
            return 2.0

        lora.add_weight = add_weight_mock
        lora.build((None, 4))

        out = lora.call(10.0)
        assert out is not None


def test_keras_lora_dense_merge_weights():
    """Docstring for test_keras_lora_dense_merge_weights."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        mock_kernel.__add__ = lambda self, other: self
        dense = mock_keras.layers.Dense(mock_kernel)

        lora = KerasLoRADense(dense=dense, r=2)

        # Merge before build
        assert lora.merge_weights() == dense

        def add_weight_mock(shape, initializer, trainable, name):
            """Docstring for add_weight_mock."""
            return 1.0

        lora.add_weight = add_weight_mock
        lora.build()

        merged = lora.merge_weights()
        assert merged == dense


def test_inject_lora_missing_deps():
    """Docstring for test_inject_lora_missing_deps."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        inject_lora(MagicMock(), ["target"])


def test_inject_lora_empty():
    """Docstring for test_inject_lora_empty."""
    mock_keras, _mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras):
        _model, count = inject_lora(MagicMock(), [])
        assert count == 0


def test_inject_lora_basic():
    """Docstring for test_inject_lora_basic."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class SubModel:
            """Docstring for SubModel."""

        sub_model = SubModel()
        sub_model.l3 = mock_keras.layers.Dense(MagicMock(), name="target_mod")

        class Model:
            """Docstring for Model."""

            def __init__(self):
                """Docstring for __init__."""
                self._tracker = MagicMock()
                self._modules = {"mod_l4": mock_keras.layers.Dense(MagicMock(), name="target_mod")}
                self.l1 = mock_keras.layers.Dense(MagicMock(), name="target_mod")
                self.l2 = mock_keras.layers.Dense(MagicMock(), name="other")
                self.layer_list = [
                    mock_keras.layers.Dense(MagicMock(), name="target_mod"),
                    sub_model,  # Cover object inside list with __dict__
                    None,  # Cover None inside list
                ]
                self.other_list = [mock_keras.layers.Dense(MagicMock(), name="ignore")]
                self.sub_model = sub_model
                self.self_ref = self  # Cover circular reference (visited)
                self._private = mock_keras.layers.Dense(MagicMock(), name="target_mod")  # Should be skipped
                self.none_attr = None

            def __getattr__(self, name):
                """Docstring for __getattr__."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model = Model()
        adapted, count = inject_lora(model, ["target_mod"])

        assert count == 4
        assert type(adapted.l1).__name__ == "KerasLoRADense"
        assert adapted.l2.trainable is False
        assert type(adapted.layer_list[0]).__name__ == "KerasLoRADense"
        assert adapted.other_list[0].trainable is False

        # Test without __dict__
        class NoDictModel:
            """Docstring for NoDictModel."""

            __slots__ = ["l1"]

            def __init__(self):
                """Docstring for __init__."""
                self.l1 = mock_keras.layers.Dense(MagicMock(), name="target")

        _no_dict, count = inject_lora(NoDictModel(), ["target"])
        assert count == 0


def test_merge_lora_weights_basic():
    """Docstring for test_merge_lora_weights_basic."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        dense_kernel = MagicMock()
        dense_kernel.shape = (4, 4)
        dense_kernel.__add__ = lambda self, other: self

        class SubModel:
            """Docstring for SubModel."""

        sub_model = SubModel()
        sub_model.l3 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        class Model:
            """Docstring for Model."""

            def __init__(self):
                """Docstring for __init__."""
                self._tracker = MagicMock()
                self._modules = {"mod_l4": KerasLoRADense(mock_keras.layers.Dense(dense_kernel))}
                self.l1 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))
                self.l2 = "not a layer"
                self.layer_list = [KerasLoRADense(mock_keras.layers.Dense(dense_kernel)), sub_model, None]
                self.sub_model = sub_model
                self.self_ref = self
                self._private = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))
                self.none_attr = None

            def __getattr__(self, name):
                """Docstring for __getattr__."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model = Model()
        merged = merge_lora_weights(model)

        assert type(merged.l1).__name__ != "KerasLoRADense"
        assert not isinstance(merged.layer_list[0], KerasLoRADense)
        assert not isinstance(merged.sub_model.l3, KerasLoRADense)

        # Test without __dict__
        class NoDictModel:
            """Docstring for NoDictModel."""

            __slots__ = ["l1"]

            def __init__(self):
                """Docstring for __init__."""
                self.l1 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        no_dict = merge_lora_weights(NoDictModel())
        assert isinstance(no_dict.l1, KerasLoRADense)


def test_count_parameters():
    """Docstring for test_count_parameters."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class Model:
            """Docstring for Model."""

            weights = [[1, 2], [3]]
            trainable_weights = [[3]]

        total, trainable = count_parameters(Model())
        assert total == 3
        assert trainable == 1


def test_apply_lora_missing_deps():
    """Docstring for test_apply_lora_missing_deps."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        apply_lora("model", ["target"])


def test_apply_lora_with_backbone():
    """Docstring for test_apply_lora_with_backbone."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.backbone.enable_lora = MagicMock()
        mock_layer = MagicMock()
        mock_layer.trainable_variables = []
        mock_model.backbone.layers = [mock_layer]

        res = apply_lora("test", ["target"], model=mock_model)

        mock_model.backbone.enable_lora.assert_called_with(rank=8)
        assert mock_layer.trainable is False
        assert res["injected_modules"] == 1


def test_apply_lora_inject_and_merge():
    """Docstring for test_apply_lora_inject_and_merge."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.l1 = mock_keras.layers.Dense(MagicMock(), name="target")
        del mock_model.backbone  # Make sure no backbone

        res = apply_lora("test", ["target"], model=mock_model, merge=True)
        assert res["status"] == "completed"


def test_apply_lora_from_preset():
    """Docstring for test_apply_lora_from_preset."""
    mock_keras, mock_ops = get_mock_keras_ops()
    mock_model = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model
    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        res = apply_lora("preset_model", ["target"])
        assert res["status"] == "completed"


def test_apply_lora_exception():
    """Docstring for test_apply_lora_exception."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.backbone.enable_lora.side_effect = ValueError("Some error")

        res = apply_lora("test", ["target"], model=mock_model)
        assert "failed: Some error" in res["status"]
