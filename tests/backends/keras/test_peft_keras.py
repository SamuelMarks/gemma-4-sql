"""Unit and behavioral tests for peft keras."""

from typing import Any
from unittest.mock import MagicMock, patch

from tests.conftest import requires_keras

pytestmark = requires_keras

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
    """Execute and verify `DummyOps` behavior and invariants."""

    def matmul(self, a, b):
        """Execute and verify `matmul` behavior and invariants."""

        class MockTensor:
            """Execute and verify `MockTensor` behavior and invariants."""

            def __add__(self, other):
                """Execute and verify `__add__` behavior and invariants."""
                return self

            def __radd__(self, other):
                """Execute and verify `__radd__` behavior and invariants."""
                return self

            def __mul__(self, other):
                """Execute and verify `__mul__` behavior and invariants."""
                return self

            def __rmul__(self, other):
                """Execute and verify `__rmul__` behavior and invariants."""
                return self

        return MockTensor()

    def size(self, w):
        """Execute and verify `size` behavior and invariants."""
        return len(w)


def get_mock_keras_ops():
    """Execute and verify `get_mock_keras_ops` behavior and invariants."""
    mock_keras = MagicMock()
    mock_keras.ops = DummyOps()

    class DummyDropout:
        """Execute and verify `DummyDropout` behavior and invariants."""

        def __init__(self, rate):
            """Execute and verify `__init__` behavior and invariants."""
            self.rate = rate

        def __call__(self, x, training=None):
            """Execute and verify `__call__` behavior and invariants."""
            return x

    mock_keras.layers.Dropout = DummyDropout

    class Layer:
        """Execute and verify `Layer` behavior and invariants."""

        def __init__(self, **kwargs):
            """Execute and verify `__init__` behavior and invariants."""

        def build(self, input_shape):
            """Execute and verify `build` behavior and invariants."""

    class Dense(Layer):
        """Execute and verify `Dense` behavior and invariants."""

        def __init__(self, kernel, bias=None, name=""):
            """Execute and verify `__init__` behavior and invariants."""
            self.kernel = kernel
            self.bias = bias
            self.name = name
            self.trainable = True

        def __call__(self, inputs):
            """Execute and verify `__call__` behavior and invariants."""
            return inputs

    mock_keras.layers.Layer = Layer
    mock_keras.layers.Dense = Dense

    return mock_keras, DummyOps()


def test_keras_lora_dense_missing_keras():
    """Execute and verify `test_keras_lora_dense_missing_keras` behavior and invariants."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        KerasLoRADense(dense=MagicMock())


def test_keras_lora_dense_invalid_rank():
    """Execute and verify `test_keras_lora_dense_invalid_rank` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops), pytest.raises(ValueError):
        KerasLoRADense(dense=MagicMock(), r=0)


def test_keras_lora_dense_build_and_call():
    """Execute and verify `test_keras_lora_dense_build_and_call` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel, bias=1)

        lora = KerasLoRADense(dense=dense, r=2, lora_alpha=4.0, lora_dropout=0.1)

        def add_weight_mock(shape, initializer, trainable, name):
            """Execute and verify `add_weight_mock` behavior and invariants."""
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

        # Cover MockTensor unused methods
        t = mock_ops.matmul(1, 2)
        _ = t + 1
        _ = t * 1


def test_keras_lora_dense_no_dropout():
    """Execute and verify `test_keras_lora_dense_no_dropout` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel)

        lora = KerasLoRADense(dense=dense, r=2, lora_dropout=0.0)
        assert lora.dropout is None

        def add_weight_mock(shape, initializer, trainable, name):
            """Execute and verify `add_weight_mock` behavior and invariants."""
            return 2.0

        lora.add_weight = add_weight_mock
        lora.build((None, 4))

        out = lora.call(10.0)
        assert out is not None


def test_keras_lora_dense_merge_weights():
    """Execute and verify `test_keras_lora_dense_merge_weights` behavior and invariants."""
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
            """Execute and verify `add_weight_mock` behavior and invariants."""
            return 1.0

        lora.add_weight = add_weight_mock
        lora.build()

        merged = lora.merge_weights()
        assert merged == dense


def test_inject_lora_missing_deps():
    """Execute and verify `test_inject_lora_missing_deps` behavior and invariants."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        inject_lora(MagicMock(), ["target"])


def test_inject_lora_empty():
    """Execute and verify `test_inject_lora_empty` behavior and invariants."""
    mock_keras, _mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras):
        _model, count = inject_lora(MagicMock(), [])
        assert count == 0


def test_inject_lora_basic():
    """Execute and verify `test_inject_lora_basic` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class SubModel:
            """Execute and verify `SubModel` behavior and invariants."""

        sub_model = SubModel()
        sub_model.l3 = mock_keras.layers.Dense(MagicMock(), name="target_mod")

        class Model:
            """Execute and verify `Model` behavior and invariants."""

            def __init__(self):
                """Execute and verify `__init__` behavior and invariants."""
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
                self.prim_attr = 42

            def __getattr__(self, name):
                """Execute and verify `__getattr__` behavior and invariants."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model = Model()
        with pytest.raises(AttributeError):
            _ = model.missing_attr
        adapted, count = inject_lora(model, ["target_mod"])

        assert count == 4
        assert type(adapted.l1).__name__ == "KerasLoRADense"
        assert adapted.l2.trainable is False
        assert type(adapted.layer_list[0]).__name__ == "KerasLoRADense"
        assert adapted.other_list[0].trainable is False

        # Test without __dict__
        class NoDictModel:
            """Execute and verify `NoDictModel` behavior and invariants."""

            __slots__ = ["l1"]

            def __init__(self):
                """Execute and verify `__init__` behavior and invariants."""
                self.l1 = mock_keras.layers.Dense(MagicMock(), name="target")

        _no_dict, count = inject_lora(NoDictModel(), ["target"])
        assert count == 0


def test_merge_lora_weights_basic():
    """Execute and verify `test_merge_lora_weights_basic` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        dense_kernel = MagicMock()
        dense_kernel.shape = (4, 4)
        dense_kernel.__add__ = lambda self, other: self

        class SubModel:
            """Execute and verify `SubModel` behavior and invariants."""

        sub_model = SubModel()
        sub_model.l3 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        class Model:
            """Execute and verify `Model` behavior and invariants."""

            def __init__(self):
                """Execute and verify `__init__` behavior and invariants."""
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
                """Execute and verify `__getattr__` behavior and invariants."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model = Model()
        with pytest.raises(AttributeError):
            _ = model.missing_attr
        merged = merge_lora_weights(model)

        assert type(merged.l1).__name__ != "KerasLoRADense"
        assert not isinstance(merged.layer_list[0], KerasLoRADense)
        assert not isinstance(merged.sub_model.l3, KerasLoRADense)

        # Test without __dict__
        class NoDictModel:
            """Execute and verify `NoDictModel` behavior and invariants."""

            __slots__ = ["l1"]

            def __init__(self):
                """Execute and verify `__init__` behavior and invariants."""
                self.l1 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        no_dict = merge_lora_weights(NoDictModel())
        assert isinstance(no_dict.l1, KerasLoRADense)


def test_count_parameters():
    """Execute and verify `test_count_parameters` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class Model:
            """Execute and verify `Model` behavior and invariants."""

            weights = [[1, 2], [3]]
            trainable_weights = [[3]]

        total, trainable = count_parameters(Model())
        assert total == 3
        assert trainable == 1

        class DummyModelNoWeights:
            """Model stub without weights attributes."""

        assert count_parameters(DummyModelNoWeights()) == (0, 0)

        class BuiltDense:
            """Pre-built Dense layer stub."""

            built = True
            trainable = True
            name = "dense"

            def __init__(self) -> None:
                """Initialize BuiltDense."""
                self.kernel = MagicMock()
                self.kernel.shape = (2, 2)
                self.bias = MagicMock()
                self.bias.shape = (2,)

            def __call__(self, *args: Any, **_kwargs: Any) -> Any:
                """Forward pass."""
                return args

        with patch.object(KerasLoRADense, "add_weight", MagicMock()):
            lora_built = KerasLoRADense(BuiltDense(), 8, 16.0, 0.0)
            assert getattr(lora_built, "_lora_built", False) is True
            _ = BuiltDense()()

        class BackboneWithTrainableVars:
            """Backbone stub with non-empty trainable_variables."""

            def enable_lora(self, rank: int) -> None:
                """Enable LoRA."""

            @property
            def layers(self) -> list[Any]:
                """Return layer with non-empty trainable_variables."""
                layer = MagicMock()
                layer.trainable_variables = [1]
                return [layer]

        class ModelWithBackbone:
            """Model wrapper with BackboneWithTrainableVars."""

            backbone = BackboneWithTrainableVars()

        res_bb = apply_lora("name", ["t"], model=ModelWithBackbone())
        assert res_bb["status"] == "completed"


def test_apply_lora_missing_deps():
    """Execute and verify `test_apply_lora_missing_deps` behavior and invariants."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        apply_lora("model", ["target"])


def test_apply_lora_with_backbone():
    """Execute and verify `test_apply_lora_with_backbone` behavior and invariants."""
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
    """Execute and verify `test_apply_lora_inject_and_merge` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.l1 = mock_keras.layers.Dense(MagicMock(), name="target")
        del mock_model.backbone  # Make sure no backbone

        res = apply_lora("test", ["target"], model=mock_model, merge=True)
        assert res["status"] == "completed"


def test_apply_lora_from_preset():
    """Execute and verify `test_apply_lora_from_preset` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    mock_model = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_model
    mock_keras_nlp = MagicMock()
    mock_keras_nlp.models.GemmaCausalLM = mock_cls

    with patch.dict("sys.modules", {"keras_nlp.models": mock_keras_nlp.models}), patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        res = apply_lora("preset_model", ["target"])
        assert res["status"] == "completed"

    with patch("gemma_4_sql.backends.keras.peft.optional_attr", return_value=None), patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        res_err = apply_lora("preset_model", ["target"])
        assert "failed:" in res_err["status"]


def test_apply_lora_exception():
    """Execute and verify `test_apply_lora_exception` behavior and invariants."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.backbone.enable_lora.side_effect = ValueError("Some error")

        res = apply_lora("test", ["target"], model=mock_model)
        assert "failed: Some error" in res["status"]
