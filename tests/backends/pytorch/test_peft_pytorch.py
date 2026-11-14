"""Tests for PyTorch PEFT / LoRA implementation, gradient isolation, and forward-pass equivalence."""

from __future__ import annotations

from pathlib import Path
from typing import Never

import pytest
import torch
from torch import nn

import gemma_4_sql.backends.pytorch.peft as peft_module
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.pytorch.inference import _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_torch

pytestmark = requires_torch


class _FakePeftWrappedModel:
    """Typed PEFT-wrapped model double supporting `print_trainable_parameters` and `save_pretrained`."""

    def __init__(self) -> None:
        """Initialize stub flags."""
        self.printed = False
        self.saved_dir: str | None = None

    def print_trainable_parameters(self) -> None:
        """Record call to `print_trainable_parameters`."""
        self.printed = True

    def save_pretrained(self, save_dir: str) -> str:
        """Record saved directory path.

        Returns:
            Saved directory path `save_dir`.
        """
        self.saved_dir = save_dir
        return save_dir


class _MinimalPeftModel:
    """Minimal PEFT model double without `print_trainable_parameters` or `save_pretrained`."""


def test_apply_lora_real_tiny_model_sgd_and_linear_independence(tmp_path: Path) -> None:
    """Verify NativeLoRALinear freezes base weights, initializes full-rank lora_a, and lowers loss under AdamW."""
    from collections.abc import Callable

    from tests.utils.tensor_helpers import torch_manual_seed

    torch_manual_seed(42)
    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    tiny_model = Gemma4ForCausalLM(cfg)
    tiny_model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        logits_base = tiny_model(tokens)[0].clone()

    injected = peft_module.inject_native_pytorch_lora(
        tiny_model,
        ["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    assert injected >= 2
    from collections.abc import Iterable

    named_mods_fn: Callable[[nn.Module], Iterable[tuple[str, nn.Module]]] = vars(nn.Module)["named_modules"]
    named_mods = dict(named_mods_fn(tiny_model))
    lora_q = named_mods["layers.0.self_attn.q_proj"]
    assert isinstance(lora_q, peft_module.NativeLoRALinearProtocol)
    assert isinstance(lora_q, nn.Module)

    # 1. Verify lora_a columns are linearly independent (full rank 4, non-constant!)
    lora_a_tensor, lora_b_tensor = lora_q.lora_weights()
    assert isinstance(lora_a_tensor, torch.Tensor)
    assert isinstance(lora_b_tensor, torch.Tensor)
    svdvals_fn: Callable[[torch.Tensor], torch.Tensor] = vars(torch.linalg)["svdvals"]
    s_vals = svdvals_fn(lora_a_tensor)
    assert int((s_vals > 1e-5).sum().item()) == 4
    assert not torch.allclose(lora_a_tensor[:, 0], lora_a_tensor[:, 1])

    # 2. Verify base weights are frozen (requires_grad is False) and bit-for-bit preserved
    base_submod = lora_q.get_submodule("base")
    assert isinstance(base_submod, nn.Linear)
    assert base_submod.weight.requires_grad is False
    base_weight_snapshot = base_submod.weight.detach().clone()

    # Zero-initialized lora_b preserves exact initial forward output
    with torch.no_grad():
        logits_zero_b = tiny_model(tokens)[0].clone()
    assert torch.allclose(logits_base, logits_zero_b, atol=1e-6)

    # 3. Run 2 steps of AdamW on trainable LoRA parameters only: loss decreases and base.weight is bit-for-bit unchanged
    trainable_params = [p for p in tiny_model.parameters() if p.requires_grad]
    assert len(trainable_params) == injected * 2
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-2)
    backward_fn: Callable[[torch.Tensor], None] = vars(torch.Tensor)["backward"]
    step_fn: Callable[[torch.optim.AdamW], None] = vars(torch.optim.AdamW)["step"]

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        logits = tiny_model(tokens)[0]
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        losses.append(float(loss.item()))
        backward_fn(loss)
        step_fn(optimizer)

    assert losses[-1] < losses[0]
    assert torch.equal(base_submod.weight, base_weight_snapshot)

    # 4. Also test top-level linear module injection and invalid lora_r <= 0
    top_lin_container = nn.Module()
    top_lin_container.add_module("q_proj", nn.Linear(4, 4, bias=False))
    assert (
        peft_module.inject_native_pytorch_lora(top_lin_container, ["q_proj"], lora_r=2, lora_alpha=4, lora_dropout=0.1)
        == 1
    )
    with pytest.raises(ValueError, match="lora_r must be positive"):
        peft_module.inject_native_pytorch_lora(top_lin_container, ["q_proj"], lora_r=0)

    # 5. Test pytorch_native apply_lora caching (cold, direct cache hit, keyed cache hit) and output_dir persistence
    out_dir = tmp_path / "native_lora_saved"
    _MODEL_CACHE.clear()
    res_native = peft_module.apply_lora(
        "gemma-4-tiny",
        ["q_proj"],
        lora_r=2,
        lora_alpha=4,
        backend_alias="pytorch_native",
        output_dir=str(out_dir),
    )
    assert res_native["status"] == "completed"
    assert (out_dir / "model.safetensors").is_file()
    assert _MODEL_CACHE.get_model("gemma-4-tiny") is not None

    # Direct cache hit branch
    res_cached_direct = peft_module.apply_lora(
        "gemma-4-tiny",
        ["v_proj"],
        lora_r=2,
        lora_alpha=4,
        backend_alias="pytorch_native",
    )
    assert res_cached_direct["status"] == "completed"

    # Keyed cache hit branch (remove direct key while keeping keyed entry)
    _MODEL_CACHE.pop("gemma-4-tiny", None)
    res_cached_keyed = peft_module.apply_lora(
        "gemma-4-tiny",
        ["o_proj"],
        lora_r=2,
        lora_alpha=4,
        backend_alias="pytorch_native",
    )
    assert res_cached_keyed["status"] == "completed"

    class BadNamedMods:
        """Stub raising RuntimeError from named_modules."""

        def named_modules(self) -> Never:
            """Raise RuntimeError.

            Raises:
                RuntimeError: Always raised to test fallback module counting.
            """
            raise RuntimeError("boom")

    assert peft_module._count_injected_peft_modules(BadNamedMods(), ["q_proj"]) == 1
    assert peft_module._count_injected_peft_modules("not_a_model", ["q_proj", "v_proj"]) == 2

    class UnmatchedNamedMods:
        """Stub yielding non-matching module names to exercise fallback len(target_modules)."""

        def named_modules(self) -> list[tuple[str, nn.Module]]:
            """Return single unmatched module."""
            return [("embed_tokens", nn.Identity())]

    assert peft_module._count_injected_peft_modules(UnmatchedNamedMods(), ["q_proj", "v_proj"]) == 2


def test_apply_lora_hf_peft_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HuggingFace PEFT apply_lora branches using typed protocol doubles without MagicMock."""
    fake_wrapped = _FakePeftWrappedModel()
    raise_on_load = False

    class _FakeAutoModel:
        """Typed `AutoModelForCausalLM` double."""

        @staticmethod
        def from_pretrained(_name: str) -> nn.Module:
            """Return a dummy `nn.Module` or raise `ValueError`.

            Raises:
                ValueError: If `raise_on_load` is True.
            """
            if raise_on_load:
                raise ValueError("error")
            return nn.Linear(4, 4)

    class _FakePeftPkg:
        """Non-None marker object representing imported `peft` package."""

    def _fake_lora_cfg(**kw: float | str | list[str]) -> dict[str, float | str | list[str]]:
        """Return LoRA config dictionary."""
        return kw

    current_model: _FakePeftWrappedModel | _MinimalPeftModel = fake_wrapped

    def _fake_get_peft(
        _base: nn.Module, _cfg: dict[str, float | str | list[str]]
    ) -> _FakePeftWrappedModel | _MinimalPeftModel:
        """Return `current_model`."""
        return current_model

    monkeypatch.setattr(peft_module, "peft", _FakePeftPkg())
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(peft_module, "LoraConfig", _fake_lora_cfg)
    monkeypatch.setattr(peft_module, "get_peft_model", _fake_get_peft)

    res = peft_module.apply_lora("m", ["l1", "l2"], output_dir="out")
    assert res["status"] == "completed"
    assert res.get("injected_modules") == 2
    assert fake_wrapped.printed is True
    assert fake_wrapped.saved_dir == "out"

    current_model = _MinimalPeftModel()
    res2 = peft_module.apply_lora("m", ["l1", "l2"], output_dir="out")
    assert res2["status"] == "completed"

    raise_on_load = True
    res_err = peft_module.apply_lora("m", ["l"])
    assert "failed: error" in res_err["status"]

    with pytest.raises(TrainingError, match="error"):
        peft_module.apply_lora("m", ["l"], raise_on_error=True)

    monkeypatch.setattr(peft_module, "peft", None)
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("m", ["l"])

    monkeypatch.setattr(peft_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("m", ["l"])
    with pytest.raises(DependencyMissingError):
        peft_module.inject_native_pytorch_lora(nn.Module(), ["l"])
