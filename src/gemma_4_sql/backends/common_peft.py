"""Common Parameter-Efficient Fine-Tuning (PEFT / LoRA) utilities across backends."""

from __future__ import annotations

import logging
from collections.abc import Callable

from gemma_4_sql.exceptions import (
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    GemmaSQLError,
    PEFTError,
    resolve_raise_on_error,
)
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
    *,
    status_code: str | None = None,
    error_message: str | None = None,
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
        status_code: Optional canonical `ExecutionStatus` value.
        error_message: Optional structured error message when `status_code` is `'failed'`.

    Returns:
        Populated `PEFTResultDict`.

    """
    resolved_code = status_code or (ExecutionStatus.FAILED.value if status.startswith("failed") else ExecutionStatus.COMPLETED.value)
    res: PEFTResultDict = {
        "backend": backend_name,
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
        "status_code": resolved_code,
        "injected_modules": injected_count,
    }
    if error_message is not None:
        res["error_message"] = error_message
    return res


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
        PEFTError: If `inject_fn` fails and `raise_on_error` is enabled.

    """
    status = ExecutionStatus.COMPLETED.value
    status_code = ExecutionStatus.COMPLETED.value
    error_message: str | None = None
    injected_count = 0
    try:
        injected_count = inject_fn()
    except (DependencyMissingError, ConfigurationError):
        raise
    except (GemmaSQLError, RuntimeError, ValueError, TypeError, OSError, LookupError, ArithmeticError, AttributeError, ImportError) as e:
        logger.exception("%s PEFT failed: ", backend_name)
        if resolve_raise_on_error(kwargs=dict(kwargs)):
            raise PEFTError(
                f"{backend_name} PEFT failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="apply_lora",
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value

    return build_peft_result(
        backend_name=backend_name,
        model_name=model_name,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        status=status,
        injected_count=injected_count,
        status_code=status_code,
        error_message=error_message,
    )


__all__ = ["apply_lora_wrapper", "build_peft_result"]
