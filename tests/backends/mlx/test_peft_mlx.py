"""Unit and behavioral tests for MLX PEFT / LoRA."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_mlx_lora_linear_forward_and_validation() -> None:
    """Test MLXLoRALinear initialization, forward pass with/without bias and dropout, and rank validation."""
    import mlx.core as mx

    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        MLXLoRALinear(in_features=4, out_features=4, r=0)

    layer = MLXLoRALinear(in_features=4, out_features=3, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=True)
    assert layer.scale == 2.0
    assert layer.W.shape == (3, 4)
    assert layer.A.shape == (4, 2)
    assert layer.B.shape == (2, 3)
    assert layer.bias is not None

    x = mx.ones((2, 4))
    out_zero = layer(x)
    assert out_zero.shape == (2, 3)
    assert bool(mx.allclose(out_zero, mx.zeros((2, 3)), atol=1e-6).item())

    layer.lora_a = mx.ones((4, 2)) * 0.5
    layer.lora_b = mx.ones((2, 3)) * 0.5
    out_nonzero = layer(x)
    # x @ A = (2, 4) @ (4, 2) -> all 2.0; (x @ A) @ B = (2, 2) @ (2, 3) -> all 2.0; scaled by 2.0 -> all 4.0
    assert bool(mx.allclose(out_nonzero, mx.full((2, 3), 4.0), atol=1e-5).item())

    layer_drop = MLXLoRALinear(in_features=4, out_features=3, r=2, lora_alpha=4.0, lora_dropout=0.1, bias=False)
    assert layer_drop.bias is None
    out_drop = layer_drop(x)
    assert out_drop.shape == (2, 3)


def test_mlx_lora_linear_from_linear_and_safetensors(tmp_path: Path) -> None:
    """Test MLXLoRALinear.from_linear conversion and adapter safetensors save/load."""
    import mlx.core as mx
    from mlx import nn

    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear

    base_lin = nn.Linear(4, 3, bias=True)
    x = mx.ones((1, 4))
    expected_base = base_lin(x)

    lora_lin = MLXLoRALinear.from_linear(base_lin, r=2, lora_alpha=4.0, lora_dropout=0.0)
    assert bool(mx.allclose(lora_lin(x), expected_base, rtol=1e-5, atol=1e-5).item())

    lora_lin.lora_b = mx.ones((2, 3)) * 0.25
    expected_adapted = lora_lin(x)

    adapter_file = tmp_path / "sub" / "layer_adapter.safetensors"
    lora_lin.save_adapters(adapter_file)
    assert adapter_file.exists()

    lora_lin2 = MLXLoRALinear.from_linear(base_lin, r=2, lora_alpha=4.0, lora_dropout=0.0)
    lora_lin2.load_adapters(adapter_file)
    assert bool(mx.allclose(lora_lin2(x), expected_adapted, rtol=1e-5, atol=1e-5).item())

    # Test missing keys error on load_adapters
    bad_file = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(bad_file), {"wrong_key": mx.zeros((1,))})
    with pytest.raises(KeyError, match="missing 'lora_a' or 'lora_b'"):
        lora_lin2.load_adapters(bad_file)


def test_inject_lora_and_model_adapter_persistence(tmp_path: Path) -> None:
    """Test inject_lora on an MLX model hierarchy, forward-pass behavior, and save/load_adapter_weights."""
    import mlx.core as mx
    from mlx import nn

    from gemma_4_sql.backends.mlx.peft import (
        MLXLoRALinear,
        apply_lora,
        inject_lora,
        load_adapter_weights,
        save_adapter_weights,
    )

    class ToyAttention(nn.Module):
        """Toy attention block with q_proj, k_proj, and v_proj."""

        def __init__(self) -> None:
            """Initialize ToyAttention."""
            super().__init__()
            self.q_proj: nn.Linear | MLXLoRALinear = nn.Linear(4, 4, bias=False)
            self.k_proj: nn.Linear | MLXLoRALinear = nn.Linear(4, 4, bias=False)
            self.v_proj: nn.Linear | MLXLoRALinear = nn.Linear(4, 4, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            """Compute sum of q, k, and v projections."""
            return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

    class ToyMLXModel(nn.Module):
        """Toy MLX model with nested attention block and list of blocks."""

        def __init__(self) -> None:
            """Initialize ToyMLXModel."""
            super().__init__()
            self.attn = ToyAttention()
            self.layers = [nn.Linear(4, 4, bias=False)]

        def __call__(self, x: mx.array) -> mx.array:
            """Run forward pass through attention and first layer."""
            return self.attn(x) + self.layers[0](x)

    model = ToyMLXModel()
    x = mx.ones((2, 4))
    out_before = model(x)

    _, zero_count = inject_lora(model, target_modules=[])
    assert zero_count == 0

    model, injected_count = inject_lora(model, target_modules=["q_proj", "v_proj", "layers.0"], lora_r=2, lora_alpha=4.0, lora_dropout=0.0)
    assert injected_count == 3
    assert isinstance(model.attn.q_proj, MLXLoRALinear)
    assert isinstance(model.attn.v_proj, MLXLoRALinear)
    assert isinstance(model.layers[0], MLXLoRALinear)
    assert not isinstance(model.attn.k_proj, MLXLoRALinear)

    # Verify forward-pass output matches base model exactly when lora_b is initialized to zero
    out_zero_b = model(x)
    assert bool(mx.allclose(out_zero_b, out_before, rtol=1e-5, atol=1e-5).item())

    # Modify lora_b to non-zero and verify forward-pass output changes
    model.attn.q_proj.lora_b = mx.ones((2, 4)) * 0.5
    out_nonzero_b = model(x)
    assert not bool(mx.allclose(out_nonzero_b, out_before).item())

    # Save and load model-level adapter weights
    save_dir = tmp_path / "adapters_out"
    save_adapter_weights(model, save_dir)
    expected_file = save_dir / "adapter.safetensors"
    assert expected_file.exists()

    model.attn.q_proj.lora_b = mx.zeros((2, 4))
    load_adapter_weights(model, expected_file)
    assert bool(mx.allclose(model.attn.q_proj.lora_b, mx.ones((2, 4)) * 0.5).item())

    with pytest.raises(FileNotFoundError):
        load_adapter_weights(model, tmp_path / "nonexistent.safetensors")

    # Test apply_lora with pre-instantiated model and output_dir
    model2 = ToyMLXModel()
    out_dir2 = tmp_path / "apply_out"
    with patch("gemma_4_sql.backends.mlx.peft.load", return_value=(model2, None)):
        res = apply_lora("test-mlx", ["q_proj"], lora_r=2, model=model2, output_dir=str(out_dir2))
        assert res["status"] == "completed"
        assert res["injected_modules"] == 1
        assert (out_dir2 / "adapter.safetensors").exists()


def test_mlx_peft_edge_cases_and_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test fallback paths in inject_lora, load_adapter_weights, and missing dependency errors."""
    import mlx.core as mx
    from mlx import nn

    import gemma_4_sql.backends.mlx.peft as peft_module

    # Test load_adapter_weights fallback when load_weights raises AttributeError
    class FallbackLoadModel:
        """Model stub without load_weights method."""

        def __init__(self) -> None:
            """Initialize FallbackLoadModel."""
            self.updated_weights: dict[str, mx.array] | None = None

        def named_modules(self) -> list[tuple[str, nn.Linear]]:
            """Return empty modules list."""
            return []

        def update(self, weights: dict[str, mx.array]) -> None:
            """Record updated weights."""
            self.updated_weights = weights

    fb_file = tmp_path / "fb.safetensors"
    mx.save_safetensors(str(fb_file), {"k": mx.ones((1,))})
    fb_model = FallbackLoadModel()
    peft_module.load_adapter_weights(fb_model, fb_file)
    assert fb_model.updated_weights is not None

    # Test list container replacement in inject_lora
    lin = nn.Linear(4, 4, bias=False)

    class ListModel(list[nn.Module]):
        """List-based model container supporting named_modules."""

        def named_modules(self) -> list[tuple[str, nn.Linear]]:
            """Return top-level and dotted module names."""
            return [("0", lin), ("1.child", lin)]

    sub_container = MagicMock()
    list_model = ListModel([lin, sub_container])
    _, cnt = peft_module.inject_lora(list_model, ["0", "child"], lora_r=2)
    assert cnt == 2
    sub_container.update_modules.assert_called_once()

    # Test apply_lora loading from mlx_lm.load and error handling
    toy_mod = nn.Linear(4, 4)
    monkeypatch.setattr(peft_module, "load", lambda _name: (toy_mod, None))  # pyright: ignore[reportUnknownLambdaType]
    res = peft_module.apply_lora("model", ["weight"], lora_r=2)
    assert res["status"] == "completed"

    monkeypatch.setattr(peft_module, "load", MagicMock(side_effect=RuntimeError("load error")))
    res_err = peft_module.apply_lora("model", ["q_proj"])
    assert "failed: load error" in res_err["status"]
    with pytest.raises(TrainingError, match="load error"):
        peft_module.apply_lora("model", ["q_proj"], raise_on_error=True)

    lora_inst = peft_module.MLXLoRALinear(2, 2, r=1)
    monkeypatch.setattr(peft_module, "mx", None)
    with pytest.raises(DependencyMissingError):
        lora_inst.save_adapters(tmp_path / "x.safetensors")
    with pytest.raises(DependencyMissingError):
        lora_inst.load_adapters(tmp_path / "x.safetensors")
    with pytest.raises(DependencyMissingError):
        peft_module.save_adapter_weights(fb_model, tmp_path / "x.safetensors")
    with pytest.raises(DependencyMissingError):
        peft_module.load_adapter_weights(fb_model, tmp_path / "x.safetensors")

    monkeypatch.setattr(peft_module, "nn", None)
    with pytest.raises(DependencyMissingError):
        peft_module.MLXLoRALinear(2, 2, r=1)
    with pytest.raises(DependencyMissingError):
        peft_module.inject_lora(toy_mod, ["q_proj"])
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("model", ["q_proj"])
