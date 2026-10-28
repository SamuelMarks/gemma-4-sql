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
    if peft is None or torch is None or AutoModelForCausalLM is None:
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
