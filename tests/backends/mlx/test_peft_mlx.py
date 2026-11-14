"""Unit and behavioral tests for MLX PEFT / LoRA (`MLXLoRALinear(mlx.nn.Module)`) without MagicMock."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Never, Self
from unittest.mock import patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_mlx
from tests.utils.tensor_helpers import assert_tensors_close

pytestmark = requires_mlx


def test_mlx_peft_fallback_base_when_mlx_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `MLXLoRALinear` falls back cleanly when `mlx` is absent on non-Apple hosts."""
    import mlx.core as mx
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.peft as peft_module

    with patch.dict(sys.modules, {"mlx": None, "mlx.nn": None, "mlx.core": None}):
        importlib.reload(peft_module)
        assert peft_module._get_native_mlx_lora_cls() is None
    importlib.reload(peft_module)

    class _MinimalNNNamespace:
        """Namespace exposing `Dropout` and `Identity` without `Module` so fallback `MLXLoRALinear` is constructed."""

        Dropout = staticmethod(mlx_nn.Dropout)
        Identity = staticmethod(mlx_nn.Identity)

    monkeypatch.setattr(peft_module, "nn", _MinimalNNNamespace())
    monkeypatch.setattr(peft_module, "mx", mx)
    fb_biased = peft_module.MLXLoRALinear(4, 3, r=2, bias=True)
    fb_unbiased = peft_module.MLXLoRALinear(4, 3, r=2, bias=False)
    fb_biased._freeze_base_weights(has_bias=True)
    assert set(fb_biased.parameters().keys()) == {"weight", "bias", "lora_a", "lora_b"}
    assert set(fb_unbiased.parameters().keys()) == {"weight", "lora_a", "lora_b"}
    assert set(fb_biased.trainable_parameters().keys()) == {"lora_a", "lora_b"}
    peft_module._set_mlx_submodule({}, "missing_leaf", fb_biased)
    peft_module._set_mlx_submodule((fb_biased,), "0", fb_unbiased)
    peft_module._set_mlx_submodule({"t": (fb_biased,)}, "t.0.leaf", fb_unbiased)


def test_mlx_lora_linear_nn_module_parameters_and_value_and_grad() -> None:
    """Test MLXLoRALinear subclasses mlx.nn.Module, freezes base weights, and trains under mlx.nn.value_and_grad."""
    from collections.abc import Callable

    import mlx.core as mx
    from mlx import nn

    import gemma_4_sql.backends.mlx.train as mlx_train_mod
    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear
    from tests.utils.tensor_helpers import mx_eval

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        MLXLoRALinear(in_features=4, out_features=4, r=0)

    layer = MLXLoRALinear(in_features=4, out_features=3, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=True)
    assert isinstance(layer, nn.Module)
    assert layer.scale == 2.0
    assert layer.W.shape == (3, 4)
    assert layer.A.shape == (4, 2)
    assert layer.B.shape == (2, 3)
    assert layer.bias is not None
    assert set(layer.parameters().keys()) == {"weight", "bias", "lora_a", "lora_b"}
    assert set(layer.trainable_parameters().keys()) == {"lora_a", "lora_b"}

    x = mx.ones((2, 4))
    out_zero = layer(x)
    assert out_zero.shape == (2, 3)
    assert bool(mx.allclose(out_zero, mx.zeros((2, 3)), atol=1e-6).item())

    # Real MLX training step via mlx.nn.value_and_grad + AdamW: loss decreases while frozen weight stays identical!
    base_lin = nn.Linear(4, 3, bias=True)
    lora_train = MLXLoRALinear.from_linear(base_lin, r=2, lora_alpha=4.0, lora_dropout=0.0)
    assert isinstance(lora_train, nn.Module)
    mx_eval(lora_train.parameters())
    assert isinstance(lora_train.weight, mx.array)
    frozen_w_before = mx.array(lora_train.weight)
    mx_eval(frozen_w_before)

    target = mx.ones((2, 3)) * 2.0
    assert mlx_train_mod.optim is not None
    opt = mlx_train_mod.optim.AdamW(learning_rate=5e-2)

    def _mse_loss(mod: MLXLoRALinear, inp: mx.array, tgt: mx.array) -> mx.array:
        """Compute mean squared error loss.

        Returns:
            Scalar MSE loss array.
        """
        diff = mod(inp) - tgt
        return mx.mean(diff * diff)

    vg_fn: Callable[
        [MLXLoRALinear, Callable[[MLXLoRALinear, mx.array, mx.array], mx.array]],
        Callable[[MLXLoRALinear, mx.array, mx.array], tuple[mx.array, dict[str, mx.array]]],
    ] = vars(nn)["value_and_grad"]
    loss_and_grad = vg_fn(lora_train, _mse_loss)
    loss0, grads0 = loss_and_grad(lora_train, x, target)
    assert set(grads0.keys()) == {"lora_a", "lora_b"}
    opt.update(lora_train, grads0)
    mx_eval(lora_train.parameters(), opt.state)

    _loss1, grads1 = loss_and_grad(lora_train, x, target)
    opt.update(lora_train, grads1)
    mx_eval(lora_train.parameters(), opt.state)
    loss2 = _mse_loss(lora_train, x, target)
    mx_eval(loss2)

    assert float(loss2) < float(loss0)
    assert isinstance(lora_train.weight, mx.array)
    assert bool(mx.array_equal(lora_train.weight, frozen_w_before).item())

    layer_drop = MLXLoRALinear(in_features=4, out_features=3, r=2, lora_alpha=4.0, lora_dropout=0.1, bias=False)
    assert isinstance(layer_drop, nn.Module)
    assert layer_drop.bias is None
    assert "lora_a" in layer_drop.trainable_parameters()
    assert "weight" not in layer_drop.trainable_parameters()
    out_drop = layer_drop(x)
    assert out_drop.shape == (2, 3)

    # Also cover custom non-Mapping _MlxBiasedLinearProtocol and unbiased _MlxLinearProtocol inputs to from_linear
    class _AttrBiasedLinear:
        """Non-dict linear stub exposing `.weight` and `.bias` attributes."""

        def __init__(self) -> None:
            """Initialize weight and bias arrays."""
            self.weight = mx.ones((3, 4))
            self.bias = mx.zeros((3,))

    class _AttrUnbiasedLinear:
        """Non-dict linear stub exposing only `.weight` without `.bias`."""

        def __init__(self) -> None:
            """Initialize weight array."""
            self.weight = mx.ones((3, 4))

    attr_lora = MLXLoRALinear.from_linear(_AttrBiasedLinear(), r=2)
    assert attr_lora.bias is not None
    attr_unbiased_lora = MLXLoRALinear.from_linear(_AttrUnbiasedLinear(), r=2)
    assert attr_unbiased_lora.bias is None


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

    from tests.utils.tensor_helpers import mx_save_safetensors

    bad_file = tmp_path / "bad.safetensors"
    mx_save_safetensors(bad_file, {"wrong_key": mx.zeros((1,))})
    with pytest.raises(KeyError, match="missing 'lora_a' or 'lora_b'"):
        lora_lin2.load_adapters(bad_file)


def test_inject_lora_and_model_adapter_persistence(tmp_path: Path) -> None:
    """Test inject_lora on an MLX model hierarchy, trainable_parameters traversal, and save/load_adapter_weights."""
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
            """Compute sum of q, k, and v projections.

            Returns:
                Summed projection output array.
            """
            return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

    class ToyMLXModel(nn.Module):
        """Toy MLX model with nested attention block and list of blocks."""

        def __init__(self) -> None:
            """Initialize ToyMLXModel."""
            super().__init__()
            self.attn = ToyAttention()
            self.layers: list[nn.Linear | MLXLoRALinear] = [nn.Linear(4, 4, bias=False)]

        def __call__(self, x: mx.array) -> mx.array:
            """Run forward pass through attention and first layer.

            Returns:
                Combined output array.
            """
            return self.attn(x) + self.layers[0](x)

    model = ToyMLXModel()
    x = mx.ones((2, 4))
    out_before = model(x)

    _, zero_count = inject_lora(model, target_modules=[])
    assert zero_count == 0

    model, injected_count = inject_lora(
        model,
        target_modules=["q_proj", "v_proj", "layers.0"],
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    assert injected_count == 3
    assert isinstance(model.attn.q_proj, MLXLoRALinear)
    assert isinstance(model.attn.v_proj, MLXLoRALinear)
    assert isinstance(model.layers[0], MLXLoRALinear)
    assert not isinstance(model.attn.k_proj, MLXLoRALinear)

    # Verify model.trainable_parameters() traverses into MLXLoRALinear and excludes frozen k_proj & base weights!
    from collections.abc import Callable

    import mlx.utils as mlx_utils

    flatten_fn: Callable[..., list[tuple[str, mx.array]]] = vars(mlx_utils)["tree_flatten"]
    flat_trainable = dict(flatten_fn(model.trainable_parameters()))
    assert set(flat_trainable.keys()) == {
        "attn.q_proj.lora_a",
        "attn.q_proj.lora_b",
        "attn.v_proj.lora_a",
        "attn.v_proj.lora_b",
        "layers.0.lora_a",
        "layers.0.lora_b",
    }

    out_zero_b = model(x)
    assert bool(mx.allclose(out_zero_b, out_before, rtol=1e-5, atol=1e-5).item())

    model.attn.q_proj.lora_b = mx.ones((2, 4)) * 0.5
    out_nonzero_b = model(x)
    assert not bool(mx.allclose(out_nonzero_b, out_before).item())

    save_dir = tmp_path / "adapters_out"
    save_adapter_weights(model, save_dir)
    expected_file = save_dir / "adapter.safetensors"
    assert expected_file.exists()

    model.attn.q_proj.lora_b = mx.zeros((2, 4))
    load_adapter_weights(model, expected_file)
    assert isinstance(model.attn.q_proj.lora_b, mx.array)
    assert_tensors_close(model.attn.q_proj.lora_b, mx.ones((2, 4)) * 0.5, backend="mlx")

    with pytest.raises(FileNotFoundError):
        load_adapter_weights(model, tmp_path / "nonexistent.safetensors")

    model2 = ToyMLXModel()
    out_dir2 = tmp_path / "apply_out"
    with patch("gemma_4_sql.backends.mlx.peft.load", return_value=(model2, None)):
        res = apply_lora("test-mlx", ["q_proj"], lora_r=2, model=model2, output_dir=str(out_dir2))
        assert res["status"] == "completed"
        assert res.get("injected_modules") == 1
        assert (out_dir2 / "adapter.safetensors").exists()


def test_mlx_peft_edge_cases_and_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test fallback paths in inject_lora, load_adapter_weights, and missing dependency errors without MagicMock."""
    import mlx.core as mx
    from mlx import nn

    import gemma_4_sql.backends.mlx.peft as peft_module

    class FallbackLoadModel:
        """Model stub without LoRA children that records `.update(weights)`."""

        def __init__(self) -> None:
            """Initialize FallbackLoadModel."""
            self.updated_weights: Mapping[str, peft_module.MlxArrayProtocol] | None = None

        def named_modules(self) -> list[tuple[str, nn.Linear]]:
            """Return empty modules list."""
            return []

        def update(self, weights: Mapping[str, peft_module.MlxArrayProtocol]) -> Self:
            """Record updated weights.

            Returns:
                Self after recording weights.
            """
            self.updated_weights = weights
            return self

    from tests.utils.tensor_helpers import mx_save_safetensors

    fb_file = tmp_path / "fb.safetensors"
    mx_save_safetensors(fb_file, {"k": mx.ones((1,))})
    fb_model = FallbackLoadModel()
    assert fb_model.named_modules() == []
    peft_module.load_adapter_weights(fb_model, fb_file)
    assert fb_model.updated_weights is not None

    # Also test load_adapter_weights when model has neither LoRA layers nor .update()
    peft_module.load_adapter_weights({}, fb_file)

    class _UpdateModulesSlot:
        """Non-dict container implementing `_MlxUpdateModulesProtocol`."""

        def __init__(self) -> None:
            """Initialize recorded modules."""
            self.updated: Mapping[str, peft_module.MLXLoRALinear] | None = None

        def update_modules(self, modules: Mapping[str, peft_module.MLXLoRALinear]) -> Self:
            """Record `modules`.

            Returns:
                Self after recording modules.
            """
            self.updated = modules
            return self

    lin = nn.Linear(4, 4, bias=False)
    sub_container = _UpdateModulesSlot()

    class ListModel(list[peft_module.MlxPEFTTarget]):
        """List-based model container supporting named_modules."""

        def named_modules(self) -> list[tuple[str, nn.Linear]]:
            """Return top-level and dotted module names."""
            return [("0", lin), ("1.child", lin)]

    list_model = ListModel([lin, sub_container])
    _, cnt = peft_module.inject_lora(list_model, ["0", "child"], lora_r=2)
    assert cnt == 2
    assert sub_container.updated is not None and "child" in sub_container.updated

    toy_mod = nn.Linear(4, 4)

    def _load_toy(_name: str) -> tuple[nn.Linear, None]:
        """Return `(toy_mod, None)`."""
        return (toy_mod, None)

    monkeypatch.setattr(peft_module, "load", _load_toy)
    res = peft_module.apply_lora("model", ["weight"], lora_r=2)
    assert res["status"] == "completed"

    def _raise_load(_name: str) -> Never:
        """Raise `RuntimeError`.

        Raises:
            RuntimeError: Always raised to simulate load failure.
        """
        raise RuntimeError("load error")

    monkeypatch.setattr(peft_module, "load", _raise_load)
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


def test_mlx_peft_sdk_integration_and_standalone_save(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify SDK apply_peft with real MLX nn.Module and standalone MLXLoRALinear adapter persistence."""
    import mlx.core as mx
    from mlx import nn

    import gemma_4_sql.backends.mlx.peft as peft_module
    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear, load_adapter_weights, save_adapter_weights
    from gemma_4_sql.sdk.peft import apply_peft

    class _RealMlxBlock(nn.Module):
        """Real MLX module block with `q_proj` and `v_proj` Linear layers."""

        def __init__(self) -> None:
            """Initialize `q_proj` and `v_proj`."""
            super().__init__()
            self.q_proj: nn.Linear | MLXLoRALinear = nn.Linear(8, 8)
            self.v_proj: nn.Linear | MLXLoRALinear = nn.Linear(8, 8)

        def __call__(self, x: mx.array) -> mx.array:
            """Compute sum of `q_proj` and `v_proj`.

            Returns:
                Summed `q_proj` and `v_proj` output array.
            """
            return self.q_proj(x) + self.v_proj(x)

    block = _RealMlxBlock()

    def _load_block(_name: str) -> tuple[_RealMlxBlock, None]:
        """Return `(block, None)`."""
        return (block, None)

    monkeypatch.setattr(peft_module, "load", _load_block)
    res = apply_peft(
        model_name="gemma-4-mlx",
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        backend="mlx",
    )
    assert res["backend"] == "mlx"
    assert res.get("injected_modules") == 2
    assert isinstance(block.q_proj, MLXLoRALinear)
    assert isinstance(block.v_proj, MLXLoRALinear)
    out = block(mx.ones((1, 3, 8)))
    assert tuple(out.shape) == (1, 3, 8)

    standalone_file = tmp_path / "standalone_lora.safetensors"
    block.q_proj.lora_b = mx.ones((4, 8)) * 0.75
    save_adapter_weights(block.q_proj, standalone_file)
    assert standalone_file.exists()
    block.q_proj.lora_b = mx.zeros((4, 8))
    load_adapter_weights(block.q_proj, standalone_file)
    assert isinstance(block.q_proj.lora_b, mx.array)
    assert_tensors_close(block.q_proj.lora_b, mx.ones((4, 8)) * 0.75, backend="mlx")
