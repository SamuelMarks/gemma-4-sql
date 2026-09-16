"""Tests for Keras PEFT / LoRA implementation."""

from __future__ import annotations

from typing import Any

import keras
import pytest
from keras import ops

import gemma_4_sql.backends.keras.peft as pt
from gemma_4_sql.backends.keras.peft import (
    KerasLoRADense,
    apply_lora,
    count_parameters,
    inject_lora,
    merge_lora_weights,
)
from gemma_4_sql.exceptions import DependencyMissingError


class MockBackbone:
    """Mock backbone layer for testing native enable_lora."""

    lora_enabled: bool = False
    layers: list[Any]

    def __init__(self) -> None:
        """Initialize mock backbone with dummy frozen layers."""
        dummy_layer = keras.layers.Dense(4)
        dummy_layer.build((None, 4))
        activation_layer = keras.layers.Activation("relu")
        self.layers = [dummy_layer, activation_layer]

    def enable_lora(self, rank: int | None = None) -> None:
        """Enable LoRA on the backbone.

        Args:
            rank: LoRA rank.
        """
        self.lora_enabled = True


class MockNativeModel:
    """Mock model containing a backbone with native enable_lora."""

    backbone: MockBackbone

    def __init__(self) -> None:
        """Initialize mock model with a backbone."""
        self.backbone = MockBackbone()


class CustomTransformerBlock(keras.layers.Layer):
    """Custom transformer layer block containing dense projections."""

    q_proj: keras.layers.Dense
    v_proj: keras.layers.Dense
    out_proj: keras.layers.Dense

    def __init__(self, units: int = 16, **kwargs: object) -> None:
        """Initialize projection layers.

        Args:
            units: Hidden units dimension.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.q_proj = keras.layers.Dense(units, name="q_proj")
        self.v_proj = keras.layers.Dense(units, name="v_proj")
        self.out_proj = keras.layers.Dense(units, name="out_proj")

    def call(self, inputs: Any) -> Any:
        """Forward pass through projections.

        Args:
            inputs: Input tensor.

        Returns:
            Projected output tensor.
        """
        return self.out_proj(self.q_proj(inputs) + self.v_proj(inputs))


class CustomKerasModel(keras.Model):
    """Custom Keras model without native enable_lora."""

    block: CustomTransformerBlock
    layers_list: list[Any]

    def __init__(self, units: int = 16, **kwargs: object) -> None:
        """Initialize custom model architecture.

        Args:
            units: Hidden units dimension.
            **kwargs: Model keyword arguments.
        """
        super().__init__(**kwargs)
        self.block = CustomTransformerBlock(units)
        self.layers_list = [keras.layers.Dense(units, name="list_dense_0")]

    def call(self, inputs: Any) -> Any:
        """Forward pass through custom model.

        Args:
            inputs: Input tensor.

        Returns:
            Model prediction tensor.
        """
        h = self.block(inputs)
        return self.layers_list[0](h)


def test_keras_lora_dense_init_and_properties() -> None:
    """Test initialization, attributes, and property accessors of KerasLoRADense."""
    dense = keras.layers.Dense(32, use_bias=True)
    lora_dense = KerasLoRADense(dense, r=4, lora_alpha=8.0, lora_dropout=0.1)

    assert lora_dense.r == 4
    assert lora_dense.lora_alpha == 8.0
    assert lora_dense.scale == 2.0
    assert lora_dense.dense is dense
    assert dense.trainable is False
    assert lora_dense.dropout is not None

    # Build layer
    x = ops.ones((2, 16))
    _ = lora_dense(x)
    assert lora_dense.lora_a.shape == (16, 4)
    assert lora_dense.lora_b.shape == (4, 32)
    assert lora_dense.kernel.shape == (16, 32)
    assert lora_dense.bias is not None
    assert lora_dense.bias.shape == (32,)

    # Test alias properties W, A, B
    assert lora_dense.W is lora_dense.kernel
    assert lora_dense.A is lora_dense.lora_a
    assert lora_dense.B is lora_dense.lora_b


def test_keras_lora_dense_zero_dropout_and_no_bias() -> None:
    """Test KerasLoRADense with 0 dropout and without bias."""
    dense = keras.layers.Dense(16, use_bias=False)
    lora_dense = KerasLoRADense(dense, r=2, lora_dropout=0.0)
    assert lora_dense.dropout is None

    x = ops.ones((2, 8))
    _ = lora_dense(x)
    assert lora_dense.bias is None


def test_keras_lora_dense_invalid_rank() -> None:
    """Test KerasLoRADense raises ValueError for rank <= 0."""
    dense = keras.layers.Dense(16)
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        KerasLoRADense(dense, r=0)

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        KerasLoRADense(dense, r=-1)


def test_keras_lora_dense_forward_pass() -> None:
    """Test mathematical correctness of KerasLoRADense forward pass."""
    dense = keras.layers.Dense(4, use_bias=True)
    dense.build((None, 4))
    dense.kernel.assign(ops.ones((4, 4)))
    dense.bias.assign(ops.full((4,), 2.0))

    lora_dense = KerasLoRADense(dense, r=2, lora_alpha=4.0, lora_dropout=0.0)
    x = ops.ones((2, 4))
    _ = lora_dense(x)

    # Initially lora_b is zero: output equals base dense output: 4 * 1.0 + 2.0 = 6.0
    y_base = lora_dense(x, training=False)
    assert bool(ops.all(ops.isclose(y_base, ops.full((2, 4), 6.0))))

    # Set lora_a and lora_b to ones:
    # (x @ A) = 4 in shape (2, 2)
    # (x @ A) @ B = 8 in shape (2, 4)
    # scale * 8 = (4 / 2) * 8 = 16.0
    # y = 6.0 + 16.0 = 22.0
    lora_dense.lora_a.assign(ops.ones((4, 2)))
    lora_dense.lora_b.assign(ops.ones((2, 4)))
    y_lora = lora_dense(x, training=False)
    assert bool(ops.all(ops.isclose(y_lora, ops.full((2, 4), 22.0))))


def test_keras_lora_dense_merge_weights() -> None:
    """Test merge_weights folds adapter matrices into the base Dense kernel."""
    dense = keras.layers.Dense(4, use_bias=False)
    dense.build((None, 4))
    dense.kernel.assign(ops.ones((4, 4)))

    lora_dense = KerasLoRADense(dense, r=2, lora_alpha=4.0)
    x = ops.ones((1, 4))
    _ = lora_dense(x)
    lora_dense.lora_a.assign(ops.ones((4, 2)))
    lora_dense.lora_b.assign(ops.ones((2, 4)))

    merged_dense = lora_dense.merge_weights()
    assert merged_dense is dense
    # 1.0 + (4/2) * (ones(4, 2) @ ones(2, 4)) = 1.0 + 2.0 * 2.0 = 5.0
    assert bool(ops.all(ops.isclose(merged_dense.kernel, ops.full((4, 4), 5.0))))


def test_inject_lora_custom_model() -> None:
    """Test inject_lora wraps designated dense layers and freezes non-adapter layers."""
    model = CustomKerasModel(units=8)
    x = ops.ones((2, 8))
    _ = model(x)

    targets = ["q_proj", "v_proj", "layers_list_0"]
    adapted_model, count = inject_lora(model, target_modules=targets, lora_r=4, lora_alpha=8.0)
    assert count == 3
    assert isinstance(adapted_model.block.q_proj, KerasLoRADense)
    assert isinstance(adapted_model.block.v_proj, KerasLoRADense)
    assert isinstance(adapted_model.layers_list[0], KerasLoRADense)
    # out_proj should remain base Dense
    assert isinstance(adapted_model.block.out_proj, keras.layers.Dense)
    assert adapted_model.block.out_proj.trainable is False

    # Trainable weights must only be lora_a and lora_b
    trainable_names = [w.name for w in adapted_model.trainable_weights]
    assert len(trainable_names) == 6  # 3 layers * 2 adapter matrices
    for name in trainable_names:
        assert "lora_a" in name or "lora_b" in name
        assert "kernel" not in name
        assert "bias" not in name

    # Test forward pass on adapted model
    y = adapted_model(x)
    assert y.shape == (2, 8)


def test_inject_lora_empty_targets_or_non_object() -> None:
    """Test inject_lora with empty targets and non-object inputs."""
    model = CustomKerasModel(units=4)
    m, c = inject_lora(model, [])
    assert c == 0
    assert m is model

    m2, c2 = inject_lora(12345, ["q_proj"])
    assert c2 == 0
    assert m2 == 12345


def test_merge_lora_weights_model() -> None:
    """Test merge_lora_weights folds all LoRA layers back across the model."""
    model = CustomKerasModel(units=8)
    x = ops.ones((2, 8))
    _ = model(x)

    inject_lora(model, target_modules=["q_proj", "layers_list_0"], lora_r=2)
    assert isinstance(model.block.q_proj, KerasLoRADense)

    merged_model = merge_lora_weights(model)
    assert isinstance(merged_model.block.q_proj, keras.layers.Dense)
    assert isinstance(merged_model.layers_list[0], keras.layers.Dense)

    # Non-dict/object test
    assert merge_lora_weights("not_a_model") == "not_a_model"


def test_count_parameters() -> None:
    """Test count_parameters computes total and trainable weights accurately."""
    model = CustomKerasModel(units=4)
    x = ops.ones((2, 4))
    _ = model(x)

    total_before, trainable_before = count_parameters(model)
    assert total_before > 0
    assert trainable_before == total_before

    inject_lora(model, target_modules=["q_proj"], lora_r=2)
    total_after, trainable_after = count_parameters(model)
    # Only lora_a and lora_b should be trainable
    assert trainable_after < total_after
    assert trainable_after == 4 * 2 + 2 * 4  # 16

    # Model without weights attribute
    assert count_parameters(object()) == (0, 0)


def test_apply_lora_native_enable_lora() -> None:
    """Test apply_lora with model supporting native enable_lora."""
    native_model = MockNativeModel()
    res = apply_lora(
        model_name="gemma-keras",
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=16,
        model=native_model,
    )
    assert res["status"] == "completed"
    assert res["backend"] == "keras"
    assert res["injected_modules"] == 2
    assert native_model.backbone.lora_enabled is True


def test_apply_lora_custom_model_fallback() -> None:
    """Test apply_lora fallback on custom model lacking enable_lora."""
    model = CustomKerasModel(units=8)
    x = ops.ones((2, 8))
    _ = model(x)

    res = apply_lora(
        model_name="custom-keras",
        target_modules=["q_proj", "v_proj"],
        lora_r=2,
        lora_alpha=4.0,
        model=model,
        merge=True,
    )
    assert res["status"] == "completed"
    assert res["injected_modules"] == 2


def test_apply_lora_preset_failure() -> None:
    """Test apply_lora captures error when preset loading fails without model in kwargs."""
    res = apply_lora(model_name="non_existent_model", target_modules=["q_proj"])
    assert "failed" in str(res["status"])


def test_apply_lora_keras_nlp_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora when keras_nlp is available to load model presets."""
    import sys

    class MockPresetModel:
        """Mock model created from preset."""

        backbone: MockBackbone

        def __init__(self) -> None:
            """Initialize with mock backbone."""
            self.backbone = MockBackbone()

        @classmethod
        def from_preset(cls, _name: str) -> MockPresetModel:
            """Simulate from_preset constructor."""
            return cls()

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(
        sys.modules,
        "keras_nlp.models",
        type("MockModels", (), {"GemmaCausalLM": MockPresetModel}),
    )

    res = apply_lora(model_name="gemma-preset", target_modules=["q_proj"])
    assert res["status"] == "completed"


def test_keras_lora_dense_unbuilt_and_double_build() -> None:
    """Test KerasLoRADense handling of unbuilt layers, double build, and merge while unbuilt."""
    unbuilt_dense = keras.layers.Dense(4)
    lora_unbuilt = KerasLoRADense(unbuilt_dense, r=2)

    # Initially unbuilt
    assert getattr(lora_unbuilt, "_lora_built", False) is False

    # Merging while unbuilt returns base dense
    assert lora_unbuilt.merge_weights() is unbuilt_dense

    # Forward pass lazily builds lora_a and lora_b
    x = ops.ones((2, 4))
    y = lora_unbuilt(x)
    assert y.shape == (2, 4)
    assert getattr(lora_unbuilt, "_lora_built", False) is True

    # Calling build again returns early
    lora_unbuilt.build((None, 4))


def test_inject_and_merge_edge_cases() -> None:
    """Test inject_lora and merge_lora_weights with None attributes and nested layers in lists."""

    class DummyContainer:
        """Container holding dummy layers, None attributes, and lists of layers."""

        def __init__(self) -> None:
            """Initialize dummy attributes."""
            self.none_attr = None
            self.unrelated_layer = keras.layers.Activation("relu")
            self.layer_list = [keras.layers.Activation("relu")]
            self.cycle = self

    container = DummyContainer()
    _m, count = inject_lora(container, ["q_proj"])
    assert count == 0

    merged = merge_lora_weights(container)
    assert merged is container


def test_apply_lora_native_with_merge() -> None:
    """Test apply_lora on native model with merge=True."""
    native_model = MockNativeModel()
    res = apply_lora(
        model_name="gemma-keras",
        target_modules=["q_proj"],
        lora_r=4,
        model=native_model,
        merge=True,
    )
    assert res["status"] == "completed"


def test_missing_dependencies_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test functions raise DependencyMissingError when Keras is missing."""
    monkeypatch.setattr(pt, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        apply_lora("model", ["q_proj"])

    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        KerasLoRADense(object(), 4)

    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        inject_lora(object(), ["q_proj"])

    monkeypatch.setattr(pt, "keras", keras)
    monkeypatch.setattr(pt, "ops", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        KerasLoRADense(object(), 4)


def test_peft_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test reloading module when Keras is absent."""
    import importlib
    import sys

    mdl = sys.modules["gemma_4_sql.backends.keras.peft"]
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
