"""Common Parameter-Efficient Fine-Tuning (PEFT / LoRA) utilities across backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, TrainingError, resolve_raise_on_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import PEFTResultDict

logger = logging.getLogger(__name__)


def build_peft_result(
    backend_name: str,
    model_name: str,
    target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    status: str,
    injected_count: int,
) -> PEFTResultDict:
    """Construct a standardized `PEFTResultDict`.

    Args:
        backend_name: Target backend identifier.
        model_name: Target model name.
        target_modules: List of target module patterns.
        lora_r: LoRA rank `r`.
        lora_alpha: LoRA scaling factor `alpha`.
        lora_dropout: LoRA dropout probability.
        status: Execution status string.
        injected_count: Number of injected LoRA layers/modules.

    Returns:
        Populated `PEFTResultDict`.

    """
    return {
        "backend": backend_name,
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
        "injected_modules": injected_count,
    }


def apply_lora_wrapper(
    backend_name: str,
    model_name: str,
    target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    inject_fn: Callable[[], int],
    **kwargs: object,
) -> PEFTResultDict:
    """Execute a backend LoRA injection pass with standardized error handling and result formatting.

    Args:
        backend_name: Target backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        model_name: Target model identifier.
        target_modules: List of target module names.
        lora_r: LoRA rank `r`.
        lora_alpha: LoRA scaling factor `alpha`.
        lora_dropout: LoRA dropout probability.
        inject_fn: Zero-argument callable executing the backend LoRA injection and returning `injected_count`.
        **kwargs: Runtime keyword options (including `raise_on_error`).

    Returns:
        Populated `PEFTResultDict`.

    Raises:
        DependencyMissingError: Re-raised if raised by `inject_fn`.
        ConfigurationError: Re-raised if raised by `inject_fn`.
        TrainingError: If `inject_fn` fails and `raise_on_error` is enabled.

    """
    status = "completed"
    injected_count = 0
    try:
        injected_count = inject_fn()
    except (DependencyMissingError, ConfigurationError):
        raise
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as e:
        logger.exception("%s PEFT failed: ", backend_name)
        if resolve_raise_on_error(kwargs=dict(kwargs)):
            raise TrainingError(
                f"{backend_name} PEFT failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="apply_lora",
                cause=e,
            ) from e
        status = f"failed: {e!s}"

    return build_peft_result(
        backend_name=backend_name,
        model_name=model_name,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        status=status,
        injected_count=injected_count,
    )


__all__ = ["apply_lora_wrapper", "build_peft_result"]
