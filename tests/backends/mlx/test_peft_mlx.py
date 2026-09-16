"""Tests for MLX PEFT / LoRA implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from gemma_4_sql.backends.mlx.peft import (
    MLXLoRALinear,
    apply_lora,
    inject_lora,
    load_adapter_weights,
    save_adapter_weights,
)
from gemma_4_sql.exceptions import DependencyMissingError


class SimpleSubModule(nn.Module):
    """Submodule containing linear projections for testing LoRA injection."""

    q_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(self, dim: int = 16) -> None:
        """Initialize projection layers.

        Args:
            dim: Input and output dimension.
        """
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=True)


class SimpleModel(nn.Module):
    """Test model containing multiple submodules in a list."""

    layers: list[SimpleSubModule]

    def __init__(self, num_layers: int = 2, dim: int = 16) -> None:
        """Initialize model with a sequence of submodules.

        Args:
            num_layers: Number of transformer-like layers.
            dim: Feature dimension.
        """
        super().__init__()
        self.layers = [SimpleSubModule(dim=dim) for _ in range(num_layers)]


def test_mlx_lora_linear_init() -> None:
    """Test initialization and properties of MLXLoRALinear."""
    layer = MLXLoRALinear(in_features=16, out_features=32, r=4, lora_alpha=8.0, lora_dropout=0.1, bias=True)
    assert layer.in_features == 16
    assert layer.out_features == 32
    assert layer.r == 4
    assert layer.lora_alpha == 8.0
    assert layer.scale == 2.0
    assert layer.weight.shape == (32, 16)
    assert layer.bias is not None
    assert layer.bias.shape == (32,)
    assert layer.lora_a.shape == (16, 4)
    assert layer.lora_b.shape == (4, 32)
    assert isinstance(layer.dropout, nn.Dropout)

    # Test property accessors
    assert layer.W is layer.weight
    assert layer.A is layer.lora_a
    assert layer.B is layer.lora_b

    # Test property setters
    new_w = mx.ones((32, 16))
    layer.W = new_w
    assert layer.weight is new_w

    new_a = mx.ones((16, 4))
    layer.A = new_a
    assert layer.lora_a is new_a

    new_b = mx.ones((4, 32))
    layer.B = new_b
    assert layer.lora_b is new_b


def test_mlx_lora_linear_zero_dropout() -> None:
    """Test MLXLoRALinear with dropout 0.0 uses nn.Identity."""
    layer = MLXLoRALinear(in_features=8, out_features=16, r=2, lora_dropout=0.0, bias=False)
    assert layer.bias is None
    assert isinstance(layer.dropout, nn.Identity)


def test_mlx_lora_linear_invalid_rank() -> None:
    """Test MLXLoRALinear raises ValueError for non-positive rank."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        MLXLoRALinear(in_features=8, out_features=8, r=0)

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        MLXLoRALinear(in_features=8, out_features=8, r=-1)


def test_mlx_lora_linear_from_linear() -> None:
    """Test construction of MLXLoRALinear from an existing nn.Linear."""
    # Linear with bias
    lin_with_bias = nn.Linear(12, 24, bias=True)
    lora_from_lin = MLXLoRALinear.from_linear(lin_with_bias, r=4, lora_alpha=16.0, lora_dropout=0.0)
    assert lora_from_lin.in_features == 12
    assert lora_from_lin.out_features == 24
    assert lora_from_lin.weight is lin_with_bias.weight
    assert lora_from_lin.bias is lin_with_bias.bias

    # Linear without bias
    lin_no_bias = nn.Linear(8, 16, bias=False)
    lora_no_bias = MLXLoRALinear.from_linear(lin_no_bias, r=2)
    assert lora_no_bias.bias is None
    assert lora_no_bias.weight is lin_no_bias.weight


def test_mlx_lora_linear_forward() -> None:
    """Test forward pass of MLXLoRALinear matches expected mathematical output."""
    # Test without bias
    layer_no_bias = MLXLoRALinear(in_features=4, out_features=6, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=False)
    layer_no_bias.weight = mx.ones((6, 4))
    layer_no_bias.lora_a = mx.ones((4, 2))
    layer_no_bias.lora_b = mx.zeros((2, 6))
    x_no_bias = mx.ones((2, 3, 4))
    y_no_bias = layer_no_bias(x_no_bias)
    assert mx.allclose(y_no_bias, mx.full((2, 3, 6), 4.0)).item()

    # Test with bias
    layer = MLXLoRALinear(in_features=4, out_features=6, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=True)
    layer.weight = mx.ones((6, 4))
    layer.bias = mx.full((6,), 0.5)
    layer.lora_a = mx.ones((4, 2))
    layer.lora_b = mx.zeros((2, 6))

    x = mx.ones((2, 3, 4))
    # When lora_b is zeros, output is strictly base: x @ W.T + bias = 4 * 1.0 + 0.5 = 4.5
    y = layer(x)
    assert y.shape == (2, 3, 6)
    expected_base = mx.full((2, 3, 6), 4.5)
    assert mx.allclose(y, expected_base).item()

    # Now set lora_b to ones: scale = 4.0 / 2 = 2.0.
    # (x @ A) = 4 * 1.0 = 4 in shape (..., 2).
    # (x @ A) @ B = 4 * 2 = 8 in shape (..., 6).
    # lora_term = 2.0 * 8 = 16.0.
    # Total = 4.5 + 16.0 = 20.5.
    layer.lora_b = mx.ones((2, 6))
    y_with_lora = layer(x)
    expected_lora = mx.full((2, 3, 6), 20.5)
    assert mx.allclose(y_with_lora, expected_lora).item()


def test_mlx_lora_linear_adapter_save_load(tmp_path: Path) -> None:
    """Test per-layer adapter save and load functionality."""
    layer = MLXLoRALinear(in_features=4, out_features=4, r=2)
    layer.lora_a = mx.ones((4, 2)) * 3.0
    layer.lora_b = mx.ones((2, 4)) * 7.0

    save_file = tmp_path / "adapters" / "sub_layer.safetensors"
    layer.save_adapters(save_file)
    assert save_file.exists()

    # Load into another layer
    layer2 = MLXLoRALinear(in_features=4, out_features=4, r=2)
    layer2.load_adapters(save_file)
    assert mx.allclose(layer2.lora_a, layer.lora_a).item()
    assert mx.allclose(layer2.lora_b, layer.lora_b).item()

    # Test error when required keys are missing
    invalid_file = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(invalid_file), {"unrelated": mx.zeros((2, 2))})
    with pytest.raises(KeyError, match="Safetensors file missing 'lora_a' or 'lora_b' keys"):
        layer2.load_adapters(invalid_file)


def test_inject_lora_modules_and_freezing() -> None:
    """Test inject_lora properly replaces modules and freezes base parameters."""
    model = SimpleModel(num_layers=2, dim=8)
    initial_trainable = tree_flatten(model.trainable_parameters())
    assert len(initial_trainable) > 0

    adapted_model, count = inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    # 2 layers * 2 targets = 4 injected LoRA modules
    assert count == 4
    assert isinstance(adapted_model.layers[0].q_proj, MLXLoRALinear)
    assert isinstance(adapted_model.layers[0].v_proj, MLXLoRALinear)
    # out_proj should remain original Linear
    assert isinstance(adapted_model.layers[0].out_proj, nn.Linear)

    # Trainable parameters should ONLY include lora_a and lora_b for the 4 adapted modules
    trainable = dict(tree_flatten(adapted_model.trainable_parameters()))
    assert len(trainable) == 8  # 4 modules * 2 matrices
    for key in trainable:
        assert key.endswith((".lora_a", ".lora_b"))
        assert "out_proj" not in key
        assert "weight" not in key
        assert "bias" not in key


def test_inject_lora_direct_list() -> None:
    """Test inject_lora when linear layers are direct elements of a list."""

    class ListModel(nn.Module):
        """Model with linear layers stored directly in a list."""

        blocks: list[nn.Linear]

        def __init__(self) -> None:
            """Initialize list of linear layers."""
            super().__init__()
            self.blocks = [nn.Linear(8, 8), nn.Linear(8, 8)]

    model = ListModel()
    adapted_model, count = inject_lora(model, target_modules=["0", "1"], lora_r=2)
    assert count == 2
    assert isinstance(adapted_model.blocks[0], MLXLoRALinear)
    assert isinstance(adapted_model.blocks[1], MLXLoRALinear)


def test_inject_lora_empty_targets_or_no_modules() -> None:
    """Test inject_lora handles empty targets and models without named_modules."""
    model = SimpleModel(num_layers=1, dim=8)
    adapted_model, count = inject_lora(model, target_modules=[])
    assert count == 0
    assert adapted_model is model

    # Object without named_modules
    class DummyObj:
        """Dummy object with freeze method for testing."""

        frozen: bool = False

        def freeze(self) -> None:
            """Mark object as frozen."""
            self.frozen = True

    dummy = DummyObj()
    res, count = inject_lora(dummy, target_modules=["q_proj"])
    assert count == 0
    assert res is dummy
    assert dummy.frozen is True


def test_save_and_load_adapter_weights(tmp_path: Path) -> None:
    """Test model-level save_adapter_weights and load_adapter_weights."""
    model = SimpleModel(num_layers=1, dim=8)
    adapted_model, _ = inject_lora(model, target_modules=["q_proj"], lora_r=2)
    adapted_model.layers[0].q_proj.lora_a = mx.ones((8, 2)) * 4.0
    adapted_model.layers[0].q_proj.lora_b = mx.ones((2, 8)) * 5.0

    # Save to directory (should automatically create adapter.safetensors)
    save_dir = tmp_path / "model_adapters"
    save_adapter_weights(adapted_model, save_dir)
    expected_file = save_dir / "adapter.safetensors"
    assert expected_file.exists()

    # Also test saving with explicit .safetensors filename
    explicit_file = tmp_path / "explicit_dir" / "my_lora.safetensors"
    save_adapter_weights(adapted_model, explicit_file)
    assert explicit_file.exists()

    # Load into another model
    model2 = SimpleModel(num_layers=1, dim=8)
    adapted_model2, _ = inject_lora(model2, target_modules=["q_proj"], lora_r=2)
    load_adapter_weights(adapted_model2, expected_file)
    assert mx.allclose(adapted_model2.layers[0].q_proj.lora_a, adapted_model.layers[0].q_proj.lora_a).item()
    assert mx.allclose(adapted_model2.layers[0].q_proj.lora_b, adapted_model.layers[0].q_proj.lora_b).item()

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Adapter weights not found"):
        load_adapter_weights(adapted_model2, tmp_path / "non_existent.safetensors")

    # Test loading into model without load_weights method
    dummy_model: Any = object()
    load_adapter_weights(dummy_model, expected_file)


def test_apply_lora_with_model_instance(tmp_path: Path) -> None:
    """Test apply_lora with pre-existing model passed via kwargs."""
    model = SimpleModel(num_layers=1, dim=8)
    output_dir = tmp_path / "apply_lora_out"

    res = apply_lora(
        model_name="custom_model",
        target_modules=["q_proj"],
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        model=model,
        output_dir=str(output_dir),
    )
    assert res["status"] == "completed"
    assert res["backend"] == "mlx"
    assert res["action"] == "apply_lora"
    assert res["lora_r"] == 4
    assert res["lora_alpha"] == 16
    assert (output_dir / "adapter.safetensors").exists()


def test_apply_lora_with_load_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora loading via mlx_lm.load mock."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    model = SimpleModel(num_layers=1, dim=8)
    monkeypatch.setattr(mpeft, "load", lambda name: (model, None))

    res = mpeft.apply_lora(
        model_name="mlx-community/test-gemma",
        target_modules=["v_proj"],
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    assert res["status"] == "completed"
    assert isinstance(model.layers[0].v_proj, MLXLoRALinear)


def test_apply_lora_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora captures runtime exceptions cleanly."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    def faulty_load(name: str) -> Any:
        """Simulate a failure during model loading.

        Args:
            name: Model name.

        Raises:
            RuntimeError: Always raised to simulate error.
        """
        raise RuntimeError("Failed to load checkpoint")

    monkeypatch.setattr(mpeft, "load", faulty_load)
    res = mpeft.apply_lora(model_name="bad-model", target_modules=["q_proj"])
    assert "failed: Failed to load checkpoint" in str(res["status"])


def test_dependency_missing_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test functions raise DependencyMissingError when MLX modules are absent."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    # Test apply_lora
    monkeypatch.setattr(mpeft, "load", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.apply_lora("m", ["q_proj"])

    monkeypatch.setattr(mpeft, "load", lambda name: None)
    monkeypatch.setattr(mpeft, "nn", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.apply_lora("m", ["q_proj"])

    monkeypatch.setattr(mpeft, "nn", type("NN", (), {}))
    monkeypatch.setattr(mpeft, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.apply_lora("m", ["q_proj"])

    # Test MLXLoRALinear init
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.MLXLoRALinear(8, 8)

    # Test inject_lora
    monkeypatch.setattr(mpeft, "nn", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.inject_lora(object(), ["q_proj"])

    # Test save_adapter_weights
    monkeypatch.setattr(mpeft, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.save_adapter_weights(object(), tmp_path / "out")

    # Test load_adapter_weights
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        mpeft.load_adapter_weights(object(), tmp_path / "out")

    # Test save_adapters and load_adapters on MLXLoRALinear
    monkeypatch.setattr(mpeft, "nn", nn)
    monkeypatch.setattr(mpeft, "mx", mx)
    layer = mpeft.MLXLoRALinear(4, 4)
    monkeypatch.setattr(mpeft, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        layer.save_adapters(tmp_path / "f.safetensors")
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        layer.load_adapters(tmp_path / "f.safetensors")
