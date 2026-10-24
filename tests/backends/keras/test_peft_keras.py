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
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError


class DummyOps:
    """Test stub for `DummyOps`."""

    def matmul(self, a: Any, b: Any):
        """Provide matmul test helper."""

        class MockTensor:
            """Test stub for `MockTensor`."""

            def __add__(self, other: Any):
                """Provide add   test helper."""
                return self

            def __radd__(self, other: Any):
                """Provide radd   test helper."""
                return self

            def __mul__(self, other: Any):
                """Provide mul   test helper."""
                return self

            def __rmul__(self, other: Any):
                """Provide rmul   test helper."""
                return self

        return MockTensor()

    def size(self, w: Any):
        """Provide size test helper."""
        return len(w)


def get_mock_keras_ops():
    """Provide get mock keras ops test helper."""
    mock_keras = MagicMock()
    mock_keras.ops = DummyOps()

    class DummyDropout:
        """Test stub for `DummyDropout`."""

        def __init__(self, rate: Any):
            """Initialize test stub instance."""
            self.rate = rate

        def __call__(self, x: Any, training: Any = None):
            """Execute callable test stub."""
            return x

    mock_keras.layers.Dropout = DummyDropout

    class Layer:
        """Test stub for `Layer`."""

        def __init__(self, **kwargs: Any):
            """Initialize test stub instance."""

        def build(self, input_shape: Any):
            """Provide build test helper."""

    class Dense(Layer):
        """Test stub for `Dense`."""

        def __init__(self, kernel: Any, bias: Any = None, name: Any = ""):
            """Initialize test stub instance."""
            self.kernel = kernel
            self.bias = bias
            self.name = name
            self.trainable = True

        def __call__(self, inputs: Any):
            """Execute callable test stub."""
            return inputs

    mock_keras.layers.Layer = Layer
    mock_keras.layers.Dense = Dense

    return mock_keras, DummyOps()


def test_keras_lora_dense_missing_keras():
    """Verify keras lora dense missing keras behavior."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        KerasLoRADense(dense=MagicMock())


def test_keras_lora_dense_invalid_rank():
    """Verify keras lora dense invalid rank behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops), pytest.raises(ValueError):
        KerasLoRADense(dense=MagicMock(), r=0)


def test_keras_lora_dense_build_and_call():
    """Verify keras lora dense build and call behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel, bias=1)

        lora = KerasLoRADense(dense=dense, r=2, lora_alpha=4.0, lora_dropout=0.1)

        def add_weight_mock(shape: Any, initializer: Any, trainable: Any, name: Any):
            """Provide add weight mock test helper."""
            return shape

        lora.add_weight = add_weight_mock

        lora.build((None, 4))
        lora.build((None, 4))  # Test already built

        assert lora.kernel == mock_kernel
        assert lora.bias == 1
        assert lora.W == mock_kernel
        assert lora.A == (4, 2)
        assert lora.B == (2, 4)

        x_in: Any = 10.0
        out = lora.call(x_in, training=True)
        assert out is not None

        # Cover MockTensor unused methods
        t = mock_ops.matmul(1, 2)
        _ = t + 1
        _ = t * 1


def test_keras_lora_dense_no_dropout():
    """Verify keras lora dense no dropout behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        dense = mock_keras.layers.Dense(mock_kernel)

        lora = KerasLoRADense(dense=dense, r=2, lora_dropout=0.0)
        assert lora.dropout is None

        def add_weight_mock(shape: Any, initializer: Any, trainable: Any, name: Any):
            """Provide add weight mock test helper."""
            return 2.0

        lora.add_weight = add_weight_mock
        lora.build((None, 4))

        x_in: Any = 10.0
        out = lora.call(x_in)
        assert out is not None


def test_keras_lora_dense_merge_weights():
    """Verify keras lora dense merge weights behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_kernel = MagicMock()
        mock_kernel.shape = (4, 4)
        mock_kernel.__add__ = lambda self, other: self
        dense = mock_keras.layers.Dense(mock_kernel)

        lora = KerasLoRADense(dense=dense, r=2)

        # Merge before build
        assert lora.merge_weights() == dense

        def add_weight_mock(shape: Any, initializer: Any, trainable: Any, name: Any):
            """Provide add weight mock test helper."""
            return 1.0

        lora.add_weight = add_weight_mock
        lora.build()

        merged = lora.merge_weights()
        assert merged == dense


def test_inject_lora_missing_deps():
    """Verify inject lora missing deps behavior."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        inject_lora(MagicMock(), ["target"])


def test_inject_lora_empty():
    """Verify inject lora empty behavior."""
    mock_keras, _mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras):
        _model, count = inject_lora(MagicMock(), [])
        assert count == 0


def test_inject_lora_basic():
    """Verify inject lora basic behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class SubModel:
            """Test stub for `SubModel`."""

            l3: Any = None

        sub_model = SubModel()
        sub_model.l3 = mock_keras.layers.Dense(MagicMock(), name="target_mod")

        class Model:
            """Test stub for `Model`."""

            def __init__(self):
                """Initialize test stub instance."""
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

            def __getattr__(self, name: Any):
                """Provide getattr   test helper."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model: Any = Model()
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
            """Test stub for `NoDictModel`."""

            __slots__ = ["l1"]

            def __init__(self):
                """Initialize test stub instance."""
                self.l1 = mock_keras.layers.Dense(MagicMock(), name="target")

        no_dict_m: Any = NoDictModel()
        _no_dict, count = inject_lora(no_dict_m, ["target"])
        assert count == 0


def test_merge_lora_weights_basic():
    """Verify merge lora weights basic behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        dense_kernel = MagicMock()
        dense_kernel.shape = (4, 4)
        dense_kernel.__add__ = lambda self, other: self

        class SubModel:
            """Test stub for `SubModel`."""

            l3: Any = None

        sub_model = SubModel()
        sub_model.l3 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        class Model:
            """Test stub for `Model`."""

            def __init__(self):
                """Initialize test stub instance."""
                self._tracker = MagicMock()
                self._modules = {"mod_l4": KerasLoRADense(mock_keras.layers.Dense(dense_kernel))}
                self.l1 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))
                self.l2 = "not a layer"
                self.layer_list = [KerasLoRADense(mock_keras.layers.Dense(dense_kernel)), sub_model, None]
                self.sub_model = sub_model
                self.self_ref = self
                self._private = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))
                self.none_attr = None

            def __getattr__(self, name: Any):
                """Provide getattr   test helper."""
                if name in self._modules:
                    return self._modules[name]
                raise AttributeError()

        model: Any = Model()
        with pytest.raises(AttributeError):
            _ = model.missing_attr
        merged = merge_lora_weights(model)

        assert type(merged.l1).__name__ != "KerasLoRADense"
        assert not isinstance(merged.layer_list[0], KerasLoRADense)
        assert not isinstance(merged.sub_model.l3, KerasLoRADense)

        # Test without __dict__
        class NoDictModel:
            """Test stub for `NoDictModel`."""

            __slots__ = ["l1"]

            def __init__(self):
                """Initialize test stub instance."""
                self.l1 = KerasLoRADense(mock_keras.layers.Dense(dense_kernel))

        no_dict_m: Any = NoDictModel()
        no_dict = merge_lora_weights(no_dict_m)
        assert isinstance(no_dict.l1, KerasLoRADense)


def test_count_parameters():
    """Verify count parameters behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):

        class Model:
            """Test stub for `Model`."""

            weights = [[1, 2], [3]]
            trainable_weights = [[3]]

        m_inst: Any = Model()
        total, trainable = count_parameters(m_inst)
        assert total == 3
        assert trainable == 1

        class DummyModelNoWeights:
            """Model stub without weights attributes."""

        no_w_inst: Any = DummyModelNoWeights()
        assert count_parameters(no_w_inst) == (0, 0)

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
    """Verify apply lora missing deps behavior."""
    with patch("gemma_4_sql.backends.keras.peft.keras", None), pytest.raises(DependencyMissingError):
        apply_lora("model", ["target"])


def test_apply_lora_with_backbone():
    """Verify apply lora with backbone behavior."""
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
    """Verify apply lora inject and merge behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.l1 = mock_keras.layers.Dense(MagicMock(), name="target")
        del mock_model.backbone  # Make sure no backbone

        res = apply_lora("test", ["target"], model=mock_model, merge=True)
        assert res["status"] == "completed"


def test_apply_lora_from_preset():
    """Verify apply lora from preset behavior."""
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
    """Verify apply lora exception behavior."""
    mock_keras, mock_ops = get_mock_keras_ops()
    with patch("gemma_4_sql.backends.keras.peft.keras", mock_keras), patch("gemma_4_sql.backends.keras.peft.ops", mock_ops):
        mock_model = MagicMock()
        mock_model.backbone.enable_lora.side_effect = ValueError("Some error")

        res = apply_lora("test", ["target"], model=mock_model)
        assert "failed: Some error" in res["status"]
        with pytest.raises(TrainingError, match="Some error"):
            apply_lora("test", ["target"], model=mock_model, raise_on_error=True)
