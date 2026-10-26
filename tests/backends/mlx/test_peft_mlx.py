"""Unit and behavioral tests for MLX PEFT / LoRA."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import gemma_4_sql.backends.mlx.peft as peft_module
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_mlx

pytestmark = requires_mlx

mx: Any = optional_import("mlx.core")
nn: Any = optional_import("mlx.nn")


def test_mlx_peft_imports() -> None:
    """Test MLX PEFT optional import fallback when MLX packages are absent."""
    import importlib

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx_lm": None}):
        reloaded_peft = importlib.reload(peft_module)
        assert reloaded_peft.mx is None
        assert reloaded_peft.nn is None

    restored_peft = importlib.reload(peft_module)
    assert restored_peft.mx is not None


def test_mlx_lora_linear_init_and_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLXLoRALinear initializes real MLX arrays and computes low-rank updates."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.MLXLoRALinear(4, 4, r=0)

    layer: Any = peft_module.MLXLoRALinear(4, 6, r=2, lora_alpha=4.0, lora_dropout=0.1, bias=True)
    assert layer.in_features == 4
    assert layer.out_features == 6
    assert layer.bias is not None
    assert tuple(layer.W.shape) == (6, 4)
    assert tuple(layer.A.shape) == (4, 2)
    assert tuple(layer.B.shape) == (2, 6)

    w_new = mx.ones((6, 4))
    a_new = mx.full((4, 2), 0.5)
    b_new = mx.full((2, 6), 0.25)
    layer.W = w_new
    assert mx.array_equal(layer.weight, w_new)
    layer.A = a_new
    assert mx.array_equal(layer.lora_a, a_new)
    layer.B = b_new
    assert mx.array_equal(layer.lora_b, b_new)
    layer.other = "custom_attr"
    assert layer.other == "custom_attr"

    layer_no_drop: Any = peft_module.MLXLoRALinear(4, 6, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=False)
    layer_no_drop.W = w_new
    layer_no_drop.A = a_new
    layer_no_drop.B = b_new
    x = mx.ones((2, 4))
    out = layer_no_drop(x)
    # y = x @ W.T + (4/2) * ((x @ A) @ B) = 4.0 + 2.0 * (2.0 * 0.5) = 6.0
    assert tuple(out.shape) == (2, 6)
    assert np.allclose(np.array(out), np.full((2, 6), 6.0), atol=1e-5)

    # Also test forward pass with bias and dropout
    out_bias = layer(x)
    assert tuple(out_bias.shape) == (2, 6)

    monkeypatch.setattr(peft_module, "nn", None)
    with pytest.raises(DependencyMissingError):
        peft_module.MLXLoRALinear(4, 4)


def test_mlx_lora_linear_from_linear() -> None:
    """Verify MLXLoRALinear.from_linear wraps real mlx.nn.Linear layers with and without bias."""
    base_lin = nn.Linear(4, 6, bias=True)
    lora_lin = peft_module.MLXLoRALinear.from_linear(base_lin, r=2, lora_alpha=8.0, lora_dropout=0.0)
    assert mx.array_equal(lora_lin.weight, base_lin.weight)
    assert lora_lin.bias is not None

    x = mx.ones((2, 4))
    # Initially lora_b is zeros, so output equals base_lin(x)
    assert np.allclose(np.array(lora_lin(x)), np.array(base_lin(x)), atol=1e-5)

    base_no_bias = nn.Linear(4, 6, bias=False)
    lora_no_bias = peft_module.MLXLoRALinear.from_linear(base_no_bias, r=2)
    assert lora_no_bias.bias is None


def test_mlx_lora_linear_save_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLXLoRALinear saves and loads real safetensors adapter weights."""
    layer: Any = peft_module.MLXLoRALinear(4, 6, r=2)
    layer.A = mx.full((4, 2), 1.5)
    layer.B = mx.full((2, 6), -0.5)

    path = tmp_path / "adapters.safetensors"
    layer.save_adapters(path)
    assert path.exists()

    layer2 = peft_module.MLXLoRALinear(4, 6, r=2)
    layer2.load_adapters(path)
    assert np.allclose(np.array(layer2.lora_a), 1.5)
    assert np.allclose(np.array(layer2.lora_b), -0.5)

    bad_path = tmp_path / "incomplete.safetensors"
    mx.save_safetensors(str(bad_path), {"lora_a": mx.zeros((4, 2))})
    with pytest.raises(KeyError, match="missing 'lora_a' or 'lora_b'"):
        layer2.load_adapters(bad_path)

    monkeypatch.setattr(peft_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        layer.save_adapters(path)
    with pytest.raises(DependencyMissingError):
        layer.load_adapters(path)


def test_inject_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify inject_lora replaces matching nn.Linear modules in nested MLX models and lists."""

    class SubBlock(nn.Module):
        """Inner MLX module with q_proj and k_proj linear layers."""

        def __init__(self) -> None:
            """Initialize SubBlock."""
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.k_proj = nn.Linear(4, 4)

    class ToyMLXModel(nn.Module):
        """Toy MLX model with nested blocks and a direct q_proj layer."""

        def __init__(self) -> None:
            """Initialize ToyMLXModel."""
            super().__init__()
            self.layers = [SubBlock()]
            self.q_proj = nn.Linear(4, 4)

    model: Any = ToyMLXModel()
    adapted, count = peft_module.inject_lora(model, ["q_proj"], lora_r=2)
    assert count == 2
    assert isinstance(adapted.q_proj, peft_module.MLXLoRALinear)
    assert isinstance(adapted.layers[0].q_proj, peft_module.MLXLoRALinear)
    assert isinstance(adapted.layers[0].k_proj, nn.Linear)

    # List container with top-level numeric index
    class ListModel(list[Any]):
        """List model exposing named_modules with numeric root index."""

        def named_modules(self) -> list[tuple[str, Any]]:
            """Yield indexed modules."""
            return [("0.q_proj", self[0].q_proj), ("1", self[1])]

    holder = SubBlock()
    model_list: Any = ListModel([holder, nn.Linear(4, 4)])
    _, count_list = peft_module.inject_lora(model_list, ["q_proj", "1"], lora_r=2)
    assert count_list == 2
    assert isinstance(model_list[1], peft_module.MLXLoRALinear)

    # Empty target_modules and missing named_modules
    _, count_empty = peft_module.inject_lora(model, [])
    assert count_empty == 0

    class NoNamedModules:
        """Object without named_modules."""

    no_nm: Any = NoNamedModules()
    _, count_no_nm = peft_module.inject_lora(no_nm, ["q_proj"])
    assert count_no_nm == 0

    monkeypatch.setattr(peft_module, "nn", None)
    with pytest.raises(DependencyMissingError):
        peft_module.inject_lora(model, ["q_proj"])


def test_save_load_adapter_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify save_adapter_weights and load_adapter_weights persist and restore safetensors."""

    class SimpleMLP(nn.Module):
        """Single-layer MLX module."""

        def __init__(self) -> None:
            """Initialize SimpleMLP."""
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    model: Any = SimpleMLP()
    peft_module.inject_lora(model, ["q_proj"], lora_r=2)
    model.q_proj.lora_b = mx.ones((2, 4))

    peft_module.save_adapter_weights(model, tmp_path)
    assert (tmp_path / "adapter.safetensors").exists()

    explicit_file = tmp_path / "custom.safetensors"
    peft_module.save_adapter_weights(model, explicit_file)
    assert explicit_file.exists()

    txt_file = tmp_path / "custom.txt"
    peft_module.save_adapter_weights(model, txt_file)
    assert (tmp_path / "custom.txt" / "adapter.safetensors").exists()

    model2: Any = SimpleMLP()
    peft_module.inject_lora(model2, ["q_proj"], lora_r=2)
    peft_module.load_adapter_weights(model2, explicit_file)
    assert np.allclose(np.array(model2.q_proj.lora_b), 1.0)

    class DictLoadModel:
        """Model without load_weights but with update()."""

        def __init__(self) -> None:
            """Initialize DictLoadModel."""
            self.updated = False

        def update(self, _weights: Any) -> None:
            """Mark updated."""
            self.updated = True

    dm: Any = DictLoadModel()
    peft_module.load_adapter_weights(dm, explicit_file)
    assert dm.updated is True

    with pytest.raises(FileNotFoundError):
        peft_module.load_adapter_weights(model2, tmp_path / "not_found.safetensors")

    monkeypatch.setattr(peft_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        peft_module.save_adapter_weights(model, tmp_path)
    with pytest.raises(DependencyMissingError):
        peft_module.load_adapter_weights(model, explicit_file)


def test_apply_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_lora end-to-end execution, output_dir saving, and error handling."""
    import gemma_4_sql.backends.mlx as mlx_pkg
    from gemma_4_sql.backends.mlx.adapter import MLXBackend

    assert mlx_pkg.get_trainer() == "mlx_trainer"
    assert isinstance(MLXBackend(), MLXBackend)

    class SimpleMLP(nn.Module):
        """Single-layer MLX module."""

        def __init__(self) -> None:
            """Initialize SimpleMLP."""
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    monkeypatch.setattr(peft_module, "load", lambda _name: (SimpleMLP(), None))
    res = peft_module.apply_lora("model", ["q_proj"], lora_r=2, output_dir=str(tmp_path))
    assert res["status"] == "completed"
    assert res["injected_modules"] == 1
    assert (tmp_path / "adapter.safetensors").exists()

    res2 = peft_module.apply_lora("model", ["q_proj"], lora_r=2, model=SimpleMLP())
    assert res2["status"] == "completed"

    # Model without freeze() method
    class NoFreezeContainer:
        """Container with named_modules but no freeze method."""

        def __init__(self) -> None:
            """Initialize NoFreezeContainer."""
            self.q_proj = nn.Linear(4, 4)

        def named_modules(self) -> list[tuple[str, Any]]:
            """Yield q_proj."""
            return [("q_proj", self.q_proj)]

    nf_model: Any = NoFreezeContainer()
    _, count_nf = peft_module.inject_lora(nf_model, ["q_proj"], lora_r=2)
    assert count_nf == 1

    # Model without load_weights or update in load_adapter_weights
    peft_module.load_adapter_weights(nf_model, tmp_path / "adapter.safetensors")

    monkeypatch.setattr(peft_module, "inject_lora", MagicMock(side_effect=RuntimeError("error")))
    res3 = peft_module.apply_lora("model", ["q_proj"])
    assert "failed: error" in res3["status"]
    with pytest.raises(TrainingError, match="error"):
        peft_module.apply_lora("model", ["q_proj"], raise_on_error=True)

    monkeypatch.setattr(peft_module, "nn", None)
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("model", ["q_proj"])
