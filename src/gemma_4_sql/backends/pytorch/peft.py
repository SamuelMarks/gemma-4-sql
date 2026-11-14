"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_inference import make_model_cache_key
from gemma_4_sql.backends.common_peft import apply_lora_wrapper, matches_target_module
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    CausalLMProtocol,
    PeftModelProtocol,
    PEFTResultDict,
    TorchCausalLMProtocol,
    TorchTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class NativeLoRALinearProtocol(Protocol):
    """Runtime-checkable protocol for a `NativeLoRALinear` adapter module."""

    def lora_weights(self) -> tuple[TorchTensorProtocol, TorchTensorProtocol]:
        """Return `(lora_a, lora_b)` trainable adapter parameter tensors.

        Returns:
            Tuple of `(lora_a, lora_b)` tensors.

        """
        ...


@runtime_checkable
class _NamedModulesContainerProtocol(Protocol):
    """Protocol for a model exposing `.named_modules()`."""

    def named_modules(
        self,
    ) -> Iterable[
        tuple[
            str, TorchCausalLMProtocol | CausalLMProtocol | _LoRAModuleMarkerProtocol | _LoRAAdapterLowerMarkerProtocol
        ]
    ]:
        """Yield `(name, module)` pairs across the module hierarchy.

        Returns:
            Iterable of `(module_path, module)` tuples.

        """
        ...


@runtime_checkable
class _LoRAModuleMarkerProtocol(Protocol):
    """Protocol for a PEFT LoRA submodule exposing `.lora_A`."""

    @property
    def lora_A(self) -> TorchCausalLMProtocol | None:
        """Provide the `lora_A` parameter container.

        Returns:
            `lora_A` container or `None`.

        """
        ...


@runtime_checkable
class _LoRAAdapterLowerMarkerProtocol(Protocol):
    """Protocol for a PEFT LoRA submodule exposing `.lora_a`."""

    @property
    def lora_a(self) -> TorchCausalLMProtocol | TorchTensorProtocol | None:
        """Provide the `lora_a` parameter container.

        Returns:
            `lora_a` parameter or container.

        """
        ...


@runtime_checkable
class _PrintableTrainableProtocol(Protocol):
    """Protocol for PEFT models exposing `.print_trainable_parameters()`."""

    def print_trainable_parameters(self) -> None:
        """Log trainable parameter summary."""
        ...


@runtime_checkable
class _SavePretrainedAdapterProtocol(Protocol):
    """Protocol for PEFT models exposing `.save_pretrained(output_dir)`."""

    def save_pretrained(self, save_directory: str, /) -> str | None:
        """Persist PEFT adapter weights to `save_directory`.

        Args:
            save_directory: Target directory path.

        Returns:
            Saved file path or `None`.

        """
        ...


@runtime_checkable
class _HFModelLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoModelForCausalLM` exposing `.from_pretrained`."""

    def from_pretrained(self, model_name: str, /) -> TorchCausalLMProtocol:
        """Load a causal language model from a pretrained identifier or path.

        Args:
            model_name: Pretrained model identifier or local path.

        Returns:
            Loaded `TorchCausalLMProtocol` instance.

        """
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
        """Construct a `peft.LoraConfig`.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA scaling alpha.
            target_modules: Target projection suffixes.
            lora_dropout: Dropout probability.
            bias: Bias training mode (`'none'`).
            task_type: PEFT task identifier (`'CAUSAL_LM'`).

        Returns:
            Initialized `LoraConfig` instance.

        """
        ...


@runtime_checkable
class _GetPeftModelProtocol(Protocol):
    """Protocol for `peft.get_peft_model(model, lora_config)`."""

    def __call__(
        self,
        model: TorchCausalLMProtocol,
        peft_config: _LoraConfigInstanceProtocol,
        /,
    ) -> (
        PeftModelProtocol
        | _PrintableTrainableProtocol
        | _SavePretrainedAdapterProtocol
        | _NamedModulesContainerProtocol
        | None
    ):
        """Wrap `model` with PEFT adapters according to `peft_config`.

        Args:
            model: Base PyTorch causal LM.
            peft_config: Configured `LoraConfig`.

        Returns:
            PEFT-wrapped model.

        """
        ...


peft = optional_import("peft")
torch = optional_import("torch")
LoraConfig: _LoraConfigConstructorProtocol | None = import_optional_symbol(
    "peft",
    "LoraConfig",
    _LoraConfigConstructorProtocol,
)
get_peft_model: _GetPeftModelProtocol | None = import_optional_symbol(
    "peft",
    "get_peft_model",
    _GetPeftModelProtocol,
)
AutoModelForCausalLM: _HFModelLoaderProtocol | None = import_optional_symbol(
    "transformers",
    "AutoModelForCausalLM",
    _HFModelLoaderProtocol,
)


@runtime_checkable
class _SubmoduleRegistrarProtocol[SubmodT](Protocol):
    """Protocol for a `torch.nn.Module` supporting `get_submodule` and `add_module`."""

    def named_modules(self) -> Iterable[tuple[str, SubmodT]]:
        """Yield `(name, module)` pairs across the module hierarchy.

        Returns:
            Iterable of `(path, submodule)` pairs.

        """
        ...

    def get_submodule(self, target: str) -> SubmodT:
        """Retrieve a submodule by dotted path.

        Args:
            target: Dotted submodule path.

        Returns:
            Target submodule.

        """
        ...

    def add_module(self, name: str, module: SubmodT | None) -> None:
        """Register a child submodule on this module.

        Args:
            name: Child attribute name.
            module: Replacement submodule.

        """
        ...


def _count_injected_peft_modules(
    model: PeftModelProtocol
    | _PrintableTrainableProtocol
    | _SavePretrainedAdapterProtocol
    | _NamedModulesContainerProtocol
    | str
    | None,
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
            matched = sum(
                1
                for name, mod in mods_iter
                if isinstance(mod, (_LoRAModuleMarkerProtocol, _LoRAAdapterLowerMarkerProtocol))
                or matches_target_module(str(name), target_modules)
            )
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
    """Freeze base weights and inject Kaiming-initialized `NativeLoRALinear` wrappers into `nn.Linear` submodules.

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
        ValueError: If `lora_r <= 0`.

    """
    if torch is None:
        raise DependencyMissingError("PyTorch is required for native LoRA injection.", backend="pytorch")
    if lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {lora_r}")

    import torch as _torch
    from torch import nn

    assert isinstance(model, nn.Module)
    for param in model.parameters():
        param.requires_grad_(False)

    scale = float(lora_alpha) / float(lora_r)

    class NativeLoRALinear(nn.Module):
        """Low-rank adapter wrapper around a frozen `torch.nn.Linear` module.

        Attributes:
            base: Frozen base `nn.Linear` module.
            lora_a: Kaiming-uniform initialized rank-projection parameter `(in_features, lora_r)`.
            lora_b: Zero-initialized output-projection parameter `(lora_r, out_features)`.
            dropout: Dropout or identity module applied to inputs on the LoRA branch.

        """

        base: nn.Linear
        lora_a: nn.Parameter
        lora_b: nn.Parameter
        dropout: nn.Module

        def __init__(self, base: nn.Linear) -> None:
            """Initialize `NativeLoRALinear` around frozen `base` with Kaiming-uniform `lora_a` and zero `lora_b`.

            Args:
                base: Target `nn.Linear` projection to wrap.

            """
            super().__init__()
            for base_param in base.parameters():
                base_param.requires_grad_(False)
            self.base = base
            self.lora_a = nn.Parameter(
                _torch.empty(
                    (base.in_features, lora_r),
                    dtype=base.weight.dtype,
                    device=base.weight.device,
                )
            )
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.lora_b = nn.Parameter(
                _torch.zeros(
                    (lora_r, base.out_features),
                    dtype=base.weight.dtype,
                    device=base.weight.device,
                )
            )
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        def lora_weights(self) -> tuple[_torch.Tensor, _torch.Tensor]:
            """Return `(lora_a, lora_b)` trainable adapter parameter tensors.

            Returns:
                Tuple of `(lora_a, lora_b)` parameter tensors.

            """
            return (self.lora_a, self.lora_b)

        def forward(self, x: _torch.Tensor) -> _torch.Tensor:
            """Compute `base(x) + scale * ((dropout(x) @ lora_a) @ lora_b)`.

            Args:
                x: Input activation tensor `(..., in_features)`.

            Returns:
                Adapted output tensor `(..., out_features)`.

            """
            return self.base(x) + scale * ((self.dropout(x) @ self.lora_a) @ self.lora_b)

    named_modules_fn: Callable[[nn.Module], Iterable[tuple[str, nn.Module]]] = vars(nn.Module)["named_modules"]
    matched_paths: list[tuple[str, nn.Linear]] = [
        (name, mod)
        for name, mod in named_modules_fn(model)
        if isinstance(mod, nn.Linear) and matches_target_module(name, target_modules)
    ]
    for path, lin_mod in matched_paths:
        wrapped = NativeLoRALinear(lin_mod)
        if "." in path:
            parent_path, child_name = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            assert isinstance(parent, nn.Module)
            parent.add_module(child_name, wrapped)
        else:
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
            """Load or retrieve cached native Gemma 4 model, inject LoRA, cache, and optionally persist.

            Returns:
                Number of linear submodules wrapped with native LoRA adapters.

            """
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM
            from gemma_4_sql.backends.pytorch.inference import MODEL_CACHE

            direct_cached = MODEL_CACHE.get_model(model_name)
            native_cache_key = make_model_cache_key(
                model_name,
                backend_name="native",
                factories=(Gemma4ForCausalLM.from_pretrained,),
            )
            keyed_cached = MODEL_CACHE.get_model(native_cache_key)
            if direct_cached is not None and isinstance(direct_cached[0], Gemma4ForCausalLM):
                native_model = direct_cached[0]
                tok_ref = direct_cached[1]
            elif keyed_cached is not None and isinstance(keyed_cached[0], Gemma4ForCausalLM):
                native_model = keyed_cached[0]
                tok_ref = keyed_cached[1]
            else:
                native_model = Gemma4ForCausalLM.from_pretrained(model_name)
                tok_ref = None

            count = inject_native_pytorch_lora(native_model, target_modules, lora_r, lora_alpha, lora_dropout)
            MODEL_CACHE.put_model(model_name, (native_model, tok_ref))
            MODEL_CACHE.put_model(native_cache_key, (native_model, tok_ref))
            if "output_dir" in kwargs and kwargs["output_dir"] is not None:
                out_dir = Path(str(kwargs["output_dir"]))
                out_dir.mkdir(parents=True, exist_ok=True)
                native_model.save_pretrained(str(out_dir))
            return count

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

    if peft is None or AutoModelForCausalLM is None or LoraConfig is None or get_peft_model is None:
        raise DependencyMissingError("PyTorch PEFT dependencies are missing.", backend="pytorch")

    auto_cls = AutoModelForCausalLM
    lora_cfg_cls = LoraConfig
    get_peft_fn = get_peft_model

    def _inject() -> int:
        """Load base model, apply PEFT LoRA config, and optionally save adapters.

        Returns:
            Count of injected PEFT LoRA adapter modules.

        """
        base_model = auto_cls.from_pretrained(model_name)
        lora_config = lora_cfg_cls(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_fn(base_model, lora_config)
        if isinstance(model, _PrintableTrainableProtocol) and callable(model.print_trainable_parameters):
            model.print_trainable_parameters()
        if (
            "output_dir" in kwargs
            and isinstance(model, _SavePretrainedAdapterProtocol)
            and callable(model.save_pretrained)
        ):
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


__all__ = [
    "AutoModelForCausalLM",
    "LoraConfig",
    "NativeLoRALinearProtocol",
    "apply_lora",
    "get_peft_model",
    "inject_native_pytorch_lora",
]
