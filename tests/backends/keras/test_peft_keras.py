"""Unit and behavioral tests for Keras PEFT / LoRA."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gemma_4_sql.backends.keras.peft import (
    KerasLoRADense,
    apply_lora,
    count_parameters,
    inject_lora,
    merge_lora_weights,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_keras

pytestmark = requires_keras

keras: Any = optional_import("keras")


def test_keras_lora_dense_missing_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify KerasLoRADense raises DependencyMissingError when keras is None."""
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.keras", None)
    with pytest.raises(DependencyMissingError):
        KerasLoRADense(dense=MagicMock())


def test_keras_lora_dense_invalid_rank() -> None:
    """Verify KerasLoRADense rejects non-positive rank r."""
    dense = keras.layers.Dense(4)
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        KerasLoRADense(dense=dense, r=0)


def test_keras_lora_dense_build_and_call() -> None:
    """Verify KerasLoRADense builds real weights, computes low-rank updates, and merges weights accurately."""
    dense = keras.layers.Dense(4, use_bias=True, name="q_proj")
    x = keras.ops.ones((2, 4), dtype="float32")
    base_out = dense(x)

    lora = KerasLoRADense(dense=dense, r=2, lora_alpha=4.0, lora_dropout=0.1)
    assert lora._lora_built is True

    # Calling build again is a safe no-op
    lora.build((2, 4))

    assert lora.kernel is dense.kernel
    assert lora.bias is dense.bias
    assert lora.W is dense.kernel
    assert tuple(lora.A.shape) == (4, 2)
    assert tuple(lora.B.shape) == (2, 4)

    # Initially B == 0, so output matches base_out
    out_init = lora(x, training=False)
    assert np.allclose(keras.ops.convert_to_numpy(out_init), keras.ops.convert_to_numpy(base_out), atol=1e-6)

    # Set non-zero lora_b and verify low-rank update math: base_out + scale * ((x @ A) @ B)
    lora.lora_a.assign(np.full((4, 2), 0.5, dtype=np.float32))
    lora.lora_b.assign(np.full((2, 4), 0.25, dtype=np.float32))
    out_updated = lora(x, training=False)
    expected_delta = (4.0 / 2.0) * (np.ones((2, 4), dtype=np.float32) @ np.full((4, 2), 0.5, dtype=np.float32) @ np.full((2, 4), 0.25, dtype=np.float32))
    expected_out = keras.ops.convert_to_numpy(base_out) + expected_delta
    assert np.allclose(keras.ops.convert_to_numpy(out_updated), expected_out, atol=1e-5)

    # Training pass with dropout active
    out_train = lora(x, training=True)
    assert tuple(out_train.shape) == (2, 4)

    # Merge weights back into base Dense layer and verify numerical equivalence
    merged = lora.merge_weights()
    assert merged is dense
    assert merged.trainable is True
    merged_out = merged(x)
    assert np.allclose(keras.ops.convert_to_numpy(merged_out), keras.ops.convert_to_numpy(out_updated), atol=1e-5)


def test_keras_lora_dense_unbuilt_and_no_dropout() -> None:
    """Verify KerasLoRADense handles unbuilt Dense layers, no-dropout mode, and pre-build merge."""
    dense = keras.layers.Dense(4, use_bias=False)
    lora = KerasLoRADense(dense=dense, r=2, lora_dropout=0.0)
    assert lora.dropout is None
    assert lora._lora_built is False

    # Merge before build returns the original dense layer unchanged
    assert lora.merge_weights() is dense

    # Build via explicit input_shape
    lora.build((None, 4))
    assert lora._lora_built is True
    assert lora.bias is None

    x = keras.ops.ones((2, 4), dtype="float32")
    out = lora.call(x, training=False)
    assert tuple(out.shape) == (2, 4)

    # Pre-built dense layer where build() is called without input_shape
    dense2 = keras.layers.Dense(3)
    dense2.build((None, 4))
    lora2 = KerasLoRADense(dense=dense2, r=2)
    assert lora2._lora_built is True


def test_inject_and_merge_lora_hierarchy() -> None:
    """Verify inject_lora and merge_lora_weights traverse nested models, lists, and trackers."""
    x = keras.ops.ones((1, 4), dtype="float32")

    def make_built_dense(name: str) -> Any:
        """Create and build a real Keras Dense layer."""
        d = keras.layers.Dense(4, name=name)
        _ = d(x)
        return d

    class SubModel:
        """Nested container holding a target Dense layer."""

        def __init__(self) -> None:
            """Initialize SubModel."""
            self.l3 = make_built_dense("target_mod")

    sub_model = SubModel()

    class CompositeModel:
        """Model container exercising _tracker, _modules, lists, circular refs, and non-layer attributes."""

        def __init__(self) -> None:
            """Initialize CompositeModel."""
            self._tracker = MagicMock()
            self._modules = {"mod_l4": make_built_dense("target_mod")}
            self.l1 = make_built_dense("target_mod")
            self.l2 = make_built_dense("other")
            self.layer_list: list[Any] = [
                make_built_dense("target_mod"),
                sub_model,
                None,
            ]
            self.other_list: list[Any] = [make_built_dense("ignore")]
            self.sub_model = sub_model
            self.self_ref = self
            self._private = make_built_dense("target_mod")
            self.none_attr = None
            self.prim_attr = 42

        def __getattr__(self, name: str) -> Any:
            """Lookup dynamic modules in _modules."""
            if name in self._modules:
                return self._modules[name]
            raise AttributeError(name)

    model: Any = CompositeModel()
    with pytest.raises(AttributeError):
        _ = model.missing_attr

    adapted, count = inject_lora(model, ["target_mod"], lora_r=2)
    assert count == 4
    assert isinstance(adapted.l1, KerasLoRADense)
    assert adapted.l2.trainable is False
    assert isinstance(adapted.layer_list[0], KerasLoRADense)
    assert adapted.other_list[0].trainable is False
    assert isinstance(adapted.sub_model.l3, KerasLoRADense)

    merged = merge_lora_weights(adapted)
    assert isinstance(merged.l1, keras.layers.Dense)
    assert not isinstance(merged.layer_list[0], KerasLoRADense)
    assert not isinstance(merged.sub_model.l3, KerasLoRADense)

    # Slot-only class without __dict__
    class NoDictModel:
        """Slot-only class without __dict__."""

        __slots__ = ["l1"]

        def __init__(self) -> None:
            """Initialize NoDictModel."""
            self.l1 = make_built_dense("target_mod")

    no_dict_m: Any = NoDictModel()
    _, count_no_dict = inject_lora(no_dict_m, ["target_mod"])
    assert count_no_dict == 0
    assert merge_lora_weights(no_dict_m) is no_dict_m

    # Empty target modules
    _, count_empty = inject_lora(model, [])
    assert count_empty == 0


def test_inject_lora_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify inject_lora raises DependencyMissingError when keras is None."""
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.keras", None)
    with pytest.raises(DependencyMissingError):
        inject_lora(MagicMock(), ["target"])


def test_count_parameters_real() -> None:
    """Verify count_parameters accurately counts total and trainable parameters on a real Keras model."""
    inp = keras.Input(shape=(4,))
    out = keras.layers.Dense(3, use_bias=True, name="q_proj")(inp)
    model: Any = keras.Model(inputs=inp, outputs=out)

    total_before, trainable_before = count_parameters(model)
    assert total_before == 4 * 3 + 3
    assert trainable_before == 15

    model.layers[1].trainable = False
    total_frozen, trainable_frozen = count_parameters(model)
    assert total_frozen == 15
    assert trainable_frozen == 0

    class DummyModelNoWeights:
        """Model stub without weights attributes."""

    no_w_inst: Any = DummyModelNoWeights()
    assert count_parameters(no_w_inst) == (0, 0)


def test_apply_lora_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_lora handles backbones, preset loading, merge=True, missing deps, and errors."""
    x = keras.ops.ones((1, 4), dtype="float32")
    d = keras.layers.Dense(4, name="target")
    _ = d(x)

    class SimpleContainer:
        """Container with a single Dense layer."""

        def __init__(self) -> None:
            """Initialize SimpleContainer."""
            self.l1 = d

    container: Any = SimpleContainer()
    res_merge = apply_lora("test", ["target"], model=container, merge=True)
    assert res_merge["status"] == "completed"
    assert res_merge["injected_modules"] == 1
    assert isinstance(container.l1, keras.layers.Dense)

    class BackboneStub:
        """KerasNLP backbone stub supporting enable_lora."""

        def __init__(self) -> None:
            """Initialize BackboneStub."""
            self.lora_rank: int | None = None
            layer_empty = MagicMock()
            layer_empty.trainable_variables = []
            layer_nonempty = MagicMock()
            layer_nonempty.trainable_variables = [1]
            self.layers = [layer_empty, layer_nonempty]

        def enable_lora(self, rank: int) -> None:
            """Record enabled LoRA rank."""
            self.lora_rank = rank

    class ModelWithBackbone:
        """Model wrapper with BackboneStub."""

        def __init__(self) -> None:
            """Initialize ModelWithBackbone."""
            self.backbone = BackboneStub()

    m_bb = ModelWithBackbone()
    res_bb = apply_lora("test", ["target"], lora_r=8, model=m_bb)
    assert res_bb["status"] == "completed"
    assert m_bb.backbone.lora_rank == 8
    assert m_bb.backbone.layers[0].trainable is False

    # Preset loading
    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = ModelWithBackbone()
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.optional_attr", lambda *_a, **_k: mock_cls)
    res_preset = apply_lora("preset_model", ["target"])
    assert res_preset["status"] == "completed"

    # Missing preset class
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.optional_attr", lambda *_a, **_k: None)
    res_missing_cls = apply_lora("preset_model", ["target"])
    assert "failed:" in res_missing_cls["status"]

    # Exception handling and raise_on_error
    bad_model = MagicMock()
    bad_model.backbone.enable_lora.side_effect = ValueError("Some error")
    res_err = apply_lora("test", ["target"], model=bad_model)
    assert "failed: Some error" in res_err["status"]
    with pytest.raises(TrainingError, match="Some error"):
        apply_lora("test", ["target"], model=bad_model, raise_on_error=True)

    # Missing keras dependency
    monkeypatch.setattr("gemma_4_sql.backends.keras.peft.keras", None)
    with pytest.raises(DependencyMissingError):
        apply_lora("model", ["target"])
