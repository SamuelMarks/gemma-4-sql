"""Unit and behavioral tests for Keras PEFT / LoRA."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.keras.peft import (
    KerasLoRADense,
    apply_lora,
    count_parameters,
    get_keras_lora_adapter,
    inject_lora,
    merge_lora_weights,
)
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import KerasLayerProtocol, get_quantized_artifact
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_keras_lora_dense_build_and_call() -> None:
    """Test KerasLoRADense layer creation, forward pass, and weight merging with real Keras tensors."""
    import keras
    from keras import ops

    dense = keras.layers.Dense(6, name="q_proj")
    x = ops.ones((2, 4), dtype="float32")
    base_out = ops.stop_gradient(dense(x))

    lora_layer = KerasLoRADense(dense=dense, r=2, lora_alpha=4.0, lora_dropout=0.0)
    assert lora_layer.scale == 2.0
    assert dense.trainable is False
    assert lora_layer.lora_a.shape == (4, 2)
    assert lora_layer.lora_b.shape == (2, 6)

    # With B initialized to zeros, output matches base_out
    out_init = ops.stop_gradient(lora_layer(x))
    assert bool(ops.all(ops.isclose(out_init, base_out, rtol=1e-5, atol=1e-5)))

    # Assign non-zero weights to B and verify LoRA delta is applied
    lora_layer.lora_b.assign(ops.ones((2, 6), dtype="float32") * 0.5)
    out_updated = ops.stop_gradient(lora_layer(x))
    assert not bool(ops.all(ops.isclose(out_updated, base_out)))

    # Merge weights into base dense kernel and verify merged dense matches lora_layer output
    merged_dense = lora_layer.merge_weights()
    assert merged_dense is dense
    assert merged_dense.trainable is True
    merged_base_out = ops.stop_gradient(dense(x))
    assert bool(ops.all(ops.isclose(merged_base_out, out_updated, rtol=1e-5, atol=1e-5)))


def test_keras_lora_dense_lazy_build_and_dropout() -> None:
    """Test KerasLoRADense when wrapped Dense is unbuilt at init and dropout > 0."""
    import keras
    from keras import ops

    unbuilt_dense = keras.layers.Dense(5, use_bias=False, name="v_proj")
    assert not unbuilt_dense.built

    lora_layer = KerasLoRADense(dense=unbuilt_dense, r=3, lora_alpha=6.0, lora_dropout=0.1)
    assert lora_layer._lora_built is False

    x = ops.ones((2, 4), dtype="float32")
    out_train = lora_layer(x, training=True)
    assert lora_layer._lora_built is True
    assert out_train.shape == (2, 5)
    assert lora_layer.kernel is unbuilt_dense.kernel
    assert lora_layer.bias is None
    assert lora_layer.W is unbuilt_dense.kernel
    assert tuple(lora_layer.A.shape) == (4, 3)
    assert tuple(lora_layer.B.shape) == (3, 5)

    # Calling build() again is a no-op
    lora_layer.build((2, 4))
    lora_layer.merge_weights()


def test_keras_lora_dense_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test KerasLoRADense validation and missing dependency errors."""
    import keras

    dense = keras.layers.Dense(4)
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        KerasLoRADense(dense=dense, r=0)

    unbuilt_dense = keras.layers.Dense(4)
    lora_unbuilt = KerasLoRADense(dense=unbuilt_dense, r=2, lora_dropout=0.0)
    # merge_weights before build is a safe no-op
    assert lora_unbuilt.merge_weights() is unbuilt_dense

    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.keras", None)
    with pytest.raises(DependencyMissingError):
        KerasLoRADense(dense=dense, r=2)


def test_inject_and_merge_lora_on_model() -> None:
    """Test inject_lora, merge_lora_weights, forward-pass modification, and count_parameters on a composite Keras model."""
    import keras
    from keras import ops

    q_layer = keras.layers.Dense(4, name="q_proj")
    k_layer = keras.layers.Dense(4, name="k_proj")
    v_layer = keras.layers.Dense(4, name="v_proj")
    x = ops.ones((1, 3), dtype="float32")
    base_q_out = ops.stop_gradient(q_layer(x))
    k_layer(x)
    base_v_out = ops.stop_gradient(v_layer(x))

    class SubBlock:
        """Nested sub-block containing v_proj."""

        def __init__(self) -> None:
            """Initialize sub-block."""
            self.layers = [v_layer]

    class CompositeModel:
        """Composite model with top-level and nested layers."""

        def __init__(self) -> None:
            """Initialize composite model."""
            self.layers = [q_layer, k_layer, SubBlock(), q_layer]
            self.weights = [*q_layer.weights, *k_layer.weights, *v_layer.weights]
            self.trainable_weights = [w for w in self.weights if w.trainable]

    model = CompositeModel()
    _, injected_zero = inject_lora(model, target_modules=[], lora_r=2)
    assert injected_zero == 0

    adapted_model, count = inject_lora(model, target_modules=["q_proj", "v_proj"], lora_r=2, lora_alpha=4.0, lora_dropout=0.0)
    assert count == 2
    adapter = get_keras_lora_adapter(adapted_model)  # ty: ignore[invalid-argument-type]
    assert adapter is not None
    assert len(adapter.layers) == 2
    assert k_layer.trainable is False

    # Verify forward pass through injected LoRA layers: when lora_b == 0, output matches base layer output
    q_lora = adapter.layers["q_proj"].lora_layer
    v_lora = adapter.layers["v_proj"].lora_layer
    assert bool(ops.all(ops.isclose(ops.stop_gradient(q_lora(x)), base_q_out, rtol=1e-5, atol=1e-5)))
    assert bool(ops.all(ops.isclose(ops.stop_gradient(v_lora(x)), base_v_out, rtol=1e-5, atol=1e-5)))

    # When lora_b is non-zero, forward-pass output changes
    q_lora.lora_b.assign(ops.full(q_lora.lora_b.shape, 0.5, dtype="float32"))
    q_out_nonzero = ops.stop_gradient(q_lora(x))
    assert not bool(ops.all(ops.isclose(q_out_nonzero, base_q_out)))

    artifact = get_quantized_artifact(adapted_model)
    assert artifact is not None
    assert artifact.method == "lora"

    total_p, trainable_p = count_parameters(q_layer)
    assert total_p > 0
    assert trainable_p == 0

    # Also test list/tuple input to inject_lora
    _, list_count = inject_lora([q_layer, k_layer], target_modules=["k_proj"], lora_r=2)
    assert list_count == 1

    # Merge weights via adapter registry
    merge_lora_weights(adapted_model)
    assert bool(ops.all(ops.isclose(ops.stop_gradient(q_layer(x)), q_out_nonzero, rtol=1e-5, atol=1e-5)))

    # Also test merge_lora_weights when model.layers directly contains KerasLoRADense
    direct_lora = KerasLoRADense(dense=k_layer, r=2)

    class DirectContainer:
        """Model container directly holding a KerasLoRADense layer."""

        def __init__(self) -> None:
            """Initialize DirectContainer."""
            self.layers = [direct_lora]

    merge_lora_weights(DirectContainer())


def test_apply_lora_backbone_and_custom_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora with KerasNLP backbone.enable_lora, custom models, and merge=True."""
    import keras
    from keras import ops

    dense = keras.layers.Dense(4, name="q_proj")
    dense(ops.ones((1, 3), dtype="float32"))

    class CustomModel:
        """Custom model without backbone."""

        def __init__(self) -> None:
            """Initialize CustomModel."""
            self.layers: list[KerasLayerProtocol] = [dense]

        @property
        def weights(self) -> list[KerasLayerProtocol]:
            """Provide empty weights list."""
            return []

        @property
        def trainable_weights(self) -> list[KerasLayerProtocol]:
            """Provide empty trainable_weights list."""
            return []

    custom_model = CustomModel()
    assert custom_model.weights == []
    assert custom_model.trainable_weights == []
    res = apply_lora("custom", ["q_proj"], lora_r=2, model=custom_model, merge=True)
    assert res["status"] == "completed"
    assert res["injected_modules"] == 1

    # Test with KerasNLP-style backbone.enable_lora where backbone also has layers
    backbone_dense = keras.layers.Dense(4, name="v_proj")
    backbone_dense(ops.ones((1, 3), dtype="float32"))
    mock_backbone = MagicMock()
    mock_backbone.layers = [backbone_dense]
    mock_gemma = MagicMock()
    mock_gemma.backbone = mock_backbone

    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = mock_gemma
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.optional_attr", lambda *_a: mock_cls)  # pyright: ignore[reportUnknownLambdaType]

    res_bb = apply_lora("gemma-preset", ["v_proj"], lora_r=4)
    assert res_bb["status"] == "completed"
    assert res_bb["injected_modules"] == 1
    mock_backbone.enable_lora.assert_called_once_with(rank=4)

    # When backbone has no matching Dense layers, count defaults to len(target_modules)
    mock_backbone.layers = []
    res_bb_empty = apply_lora("gemma-preset", ["q_proj", "v_proj"], lora_r=4)
    assert res_bb_empty["injected_modules"] == 2

    # When GemmaCausalLM is not available and no model is passed
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.optional_attr", lambda *_a: None)  # pyright: ignore[reportUnknownLambdaType]
    res_fail = apply_lora("gemma-preset", ["q_proj"])
    assert "failed:" in res_fail["status"]

    with pytest.raises(TrainingError, match="GemmaCausalLM is not available"):
        apply_lora("gemma-preset", ["q_proj"], raise_on_error=True)

    # When inject_lora raises an error
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.inject_lora", MagicMock(side_effect=RuntimeError("inject error")))
    res_err = apply_lora("custom", ["target"], model=custom_model)
    assert "failed: inject error" in res_err["status"]
    with pytest.raises(TrainingError, match="inject error"):
        apply_lora("custom", ["target"], model=custom_model, raise_on_error=True)

    # Missing keras dependency
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.keras", None)
    with pytest.raises(DependencyMissingError):
        apply_lora("model", ["target"])
    with pytest.raises(DependencyMissingError):
        inject_lora(custom_model, ["target"])
