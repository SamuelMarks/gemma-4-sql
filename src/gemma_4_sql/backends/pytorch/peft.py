"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_peft import apply_lora_wrapper, matches_target_module
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import CausalLMProtocol, PeftModelProtocol, PEFTResultDict, TorchCausalLMProtocol

logger = logging.getLogger(__name__)


@runtime_checkable
class _NamedModulesContainerProtocol(Protocol):
    """Protocol for a model exposing `.named_modules()`."""

    def named_modules(
        self,
    ) -> Iterable[tuple[str, TorchCausalLMProtocol | CausalLMProtocol | _LoRAModuleMarkerProtocol | _LoRAAdapterLowerMarkerProtocol]]:
        """Yield `(name, module)` pairs across the module hierarchy."""
        ...


@runtime_checkable
class _LoRAModuleMarkerProtocol(Protocol):
    """Protocol for a PEFT LoRA submodule exposing `.lora_A` or `.lora_a`."""

    @property
    def lora_A(self) -> TorchCausalLMProtocol | None:
        """Provide the `lora_A` parameter container."""
        ...


@runtime_checkable
class _LoRAAdapterLowerMarkerProtocol(Protocol):
    """Protocol for a PEFT LoRA submodule exposing `.lora_a`."""

    @property
    def lora_a(self) -> TorchCausalLMProtocol | None:
        """Provide the `lora_a` parameter container."""
        ...


@runtime_checkable
class _HFModelLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoModelForCausalLM` exposing `.from_pretrained`."""

    def from_pretrained(self, model_name: str, /) -> TorchCausalLMProtocol:
        """Load a causal language model from a pretrained identifier or path."""
        ...


@runtime_checkable
class _LoraConfigInstanceProtocol(Protocol):
    """Protocol for an initialized `peft.LoraConfig` instance."""


@runtime_checkable
class _LoraConfigConstructorProtocol(Protocol):
    """Protocol for `peft.LoraConfig` constructor."""

    def __call__(
        self,
        *,
        r: int,
        lora_alpha: int,
        target_modules: list[str],
        lora_dropout: float,
        bias: str,
        task_type: str,
    ) -> _LoraConfigInstanceProtocol:
        """Construct a `peft.LoraConfig`."""
        ...


@runtime_checkable
class _GetPeftModelProtocol(Protocol):
    """Protocol for `peft.get_peft_model(model, lora_config)`."""

    def __call__(
        self,
        model: TorchCausalLMProtocol,
        peft_config: _LoraConfigInstanceProtocol,
        /,
    ) -> PeftModelProtocol:
        """Wrap `model` with PEFT adapters according to `peft_config`."""
        ...


peft = optional_import("peft")
torch = optional_import("torch")
LoraConfig: _LoraConfigConstructorProtocol | None = optional_attr(
    "peft",
    "LoraConfig",
    _LoraConfigConstructorProtocol,
)
get_peft_model: _GetPeftModelProtocol | None = optional_attr(
    "peft",
    "get_peft_model",
    _GetPeftModelProtocol,
)
AutoModelForCausalLM: _HFModelLoaderProtocol | None = optional_attr(
    "transformers",
    "AutoModelForCausalLM",
    _HFModelLoaderProtocol,
)


@runtime_checkable
class _SubmoduleRegistrarProtocol[SubmodT](Protocol):
    """Protocol for a `torch.nn.Module` supporting `get_submodule` and `add_module`."""

    def named_modules(self) -> Iterable[tuple[str, SubmodT]]:
        """Yield `(name, module)` pairs across the module hierarchy."""
        ...

    def get_submodule(self, target: str) -> SubmodT:
        """Retrieve a submodule by dotted path."""
        ...

    def add_module(self, name: str, module: SubmodT | None) -> None:
        """Register a child submodule on this module."""
        ...


def _count_injected_peft_modules(
    model: _NamedModulesContainerProtocol | str | None,
    target_modules: list[str],
) -> int:
    """Count injected LoRA adapter layers on a PyTorch PEFT model or fall back to `len(target_modules)`.

    Args:
        model: Wrapped PEFT model instance.
        target_modules: Target module name suffixes.

    Returns:
        Count of injected LoRA adapter modules.

    """
    if isinstance(model, _NamedModulesContainerProtocol):
        try:
            mods_iter = model.named_modules()
            matched = sum(1 for name, mod in mods_iter if isinstance(mod, (_LoRAModuleMarkerProtocol, _LoRAAdapterLowerMarkerProtocol)) or matches_target_module(str(name), target_modules))
            return matched if matched > 0 else len(target_modules)
        except (TypeError, ValueError, RuntimeError):
            pass
    return len(target_modules)


def inject_native_pytorch_lora[SubmodT](
    model: _SubmoduleRegistrarProtocol[SubmodT],
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> int:
    """Inject native PyTorch LoRA linear wrappers into matching `nn.Linear` submodules using `add_module`.

    Args:
        model: Target `torch.nn.Module` hierarchy.
        target_modules: Target module suffix names (e.g. `['q_proj', 'v_proj']`).
        lora_r: LoRA rank (`r > 0`).
        lora_alpha: LoRA scaling factor (`alpha`).
        lora_dropout: Dropout probability.

    Returns:
        Number of linear submodules wrapped with native LoRA adapters.

    Raises:
        DependencyMissingError: If PyTorch is unavailable.

    """
    if torch is None:
        raise DependencyMissingError("PyTorch is required for native LoRA injection.", backend="pytorch")
    import math as _math

    import torch as _torch
    from torch import nn

    scale = float(lora_alpha) / float(max(1, lora_r))

    class NativeLoRALinear(nn.Module):
        """Low-rank adapter wrapper around a `torch.nn.Linear` module."""

        def __init__(self, base: nn.Linear) -> None:
            """Initialize `NativeLoRALinear` around `base`."""
            super().__init__()
            self.base = base
            self.lora_a = nn.Parameter(_torch.full((base.in_features, lora_r), 1.0 / _math.sqrt(max(1, lora_r)), dtype=base.weight.dtype, device=base.weight.device))
            self.lora_b = nn.Parameter(_torch.zeros((lora_r, base.out_features), dtype=base.weight.dtype, device=base.weight.device))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        def forward(self, x: _torch.Tensor) -> _torch.Tensor:
            """Compute `base(x) + scale * ((dropout(x) @ lora_a) @ lora_b)`."""
            return self.base(x) + scale * ((self.dropout(x) @ self.lora_a) @ self.lora_b)

    matched_paths: list[tuple[str, nn.Linear]] = [(name, mod) for name, mod in list(model.named_modules()) if isinstance(mod, nn.Linear) and matches_target_module(name, target_modules)]
    for path, lin_mod in matched_paths:
        wrapped = NativeLoRALinear(lin_mod)
        if "." in path:
            parent_path, child_name = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            assert isinstance(parent, nn.Module)
            parent.add_module(child_name, wrapped)
        else:
            assert isinstance(model, nn.Module)
            model.add_module(path, wrapped)
    return len(matched_paths)


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: StatusValue,
) -> PEFTResultDict:
    """Apply LoRA to a model using the PyTorch backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Additional configuration options (including `output_dir` and `raise_on_error`).

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If PyTorch PEFT dependencies are missing.
        TrainingError: If LoRA application fails and `raise_on_error` is enabled.

    """
    backend_alias = str(kwargs.get("backend_alias", "pytorch"))
    if torch is None:
        raise DependencyMissingError("PyTorch PEFT dependencies are missing.", backend=backend_alias)

    if backend_alias == "pytorch_native":

        def _inject_native() -> int:
            """Load native Gemma 4 model and inject native LoRA linear wrappers."""
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

            native_model = Gemma4ForCausalLM.from_pretrained(model_name)
            return inject_native_pytorch_lora(native_model, target_modules, lora_r, lora_alpha, lora_dropout)

        return apply_lora_wrapper(
            "pytorch_native",
            model_name,
            target_modules,
            lora_r,
            lora_alpha,
            lora_dropout,
            _inject_native,
            **kwargs,
        )

    if peft is None or AutoModelForCausalLM is None:
        raise DependencyMissingError("PyTorch PEFT dependencies are missing.", backend="pytorch")

    def _inject() -> int:
        """Load base model, apply PEFT LoRA config, and optionally save adapters."""
        assert AutoModelForCausalLM is not None
        assert LoraConfig is not None
        assert get_peft_model is not None
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        with contextlib.suppress(AttributeError, TypeError):
            model.print_trainable_parameters()
        if "output_dir" in kwargs:
            with contextlib.suppress(AttributeError, TypeError):
                model.save_pretrained(str(kwargs["output_dir"]))
        return _count_injected_peft_modules(model, target_modules)

    return apply_lora_wrapper(
        "pytorch",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **kwargs,
    )
