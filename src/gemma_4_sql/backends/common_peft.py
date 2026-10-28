"""Common Parameter-Efficient Fine-Tuning (PEFT / LoRA) utilities across backends."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    PEFTError,
    StatusValue,
    is_failed_status,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import PEFTResultDict

logger = logging.getLogger(__name__)


def matches_target_module(name: str, target_modules: list[str]) -> bool:
    """Return whether `name` matches any target module pattern by suffix or substring.

    Args:
        name: Module or parameter path name.
        target_modules: List of target module patterns.

    Returns:
        `True` if `name` matches any entry in `target_modules`.

    """
    return any(name == target or name.endswith(f".{target}") or target in name for target in target_modules)


def compute_lora_scale(lora_alpha: float, lora_r: int) -> float:
    """Compute the LoRA scaling factor `lora_alpha / lora_r`.

    Args:
        lora_alpha: LoRA scaling parameter `alpha`.
        lora_r: LoRA rank `r` (must be positive).

    Returns:
        Floating-point scale factor `float(lora_alpha) / float(lora_r)`.

    Raises:
        ValueError: If `lora_r <= 0`.

    """
    if lora_r <= 0:
        raise ValueError(f"LoRA rank r must be positive, got {lora_r}")
    return float(lora_alpha) / float(lora_r)


def compute_lora_matrices(
    in_features: int,
    out_features: int,
    lora_r: int,
    seed: int = 0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute Kaiming-initialized `lora_a` `(in_features, lora_r)` and zero-initialized `lora_b` `(lora_r, out_features)`.

    Mathematical Formulation (Hu et al., 2021):
        - `A ~ Uniform(-bound, bound)` where `bound = 1 / sqrt(in_features)`
        - `B = 0` so `Delta W = B @ A = 0` at initialization (`t = 0`).

    Args:
        in_features: Input dimension of the linear projection.
        out_features: Output dimension of the linear projection.
        lora_r: Rank `r` of the low-rank decomposition.
        seed: Deterministic integer seed offset for reproducible initialization.

    Returns:
        A 2-tuple of `(lora_a, lora_b)` nested float lists of shapes `(in_features, lora_r)` and `(lora_r, out_features)`.

    Raises:
        ValueError: If `in_features <= 0`, `out_features <= 0`, or `lora_r <= 0`.

    """
    if in_features <= 0 or out_features <= 0 or lora_r <= 0:
        msg = f"Invalid LoRA dimensions: in_features={in_features}, out_features={out_features}, lora_r={lora_r}"
        raise ValueError(msg)
    bound = 1.0 / math.sqrt(float(in_features))
    lora_a: list[list[float]] = []
    for i in range(in_features):
        row: list[float] = []
        for j in range(lora_r):
            phase = float((i * 131 + j * 17 + seed * 31) % 1009) / 1009.0
            val = (2.0 * phase - 1.0) * bound
            row.append(val)
        lora_a.append(row)
    lora_b: list[list[float]] = [[0.0 for _ in range(out_features)] for _ in range(lora_r)]
    return (lora_a, lora_b)


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
    resolved_code = status_code or (ExecutionStatus.FAILED.value if is_failed_status(status) else ExecutionStatus.COMPLETED.value)
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
    **kwargs: StatusValue,
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
        ConfigurationError: If `lora_r <= 0` or `lora_dropout` is outside `[0.0, 1.0)`, or if raised by `inject_fn`.
        PEFTError: If `inject_fn` fails and `raise_on_error` is enabled.

    """
    if lora_r <= 0:
        raise ConfigurationError(f"lora_r must be > 0, got {lora_r}", backend=backend_name, model_name=model_name, operation="apply_lora")
    if not (0.0 <= lora_dropout < 1.0):
        raise ConfigurationError(
            f"lora_dropout must be in [0.0, 1.0), got {lora_dropout}",
            backend=backend_name,
            model_name=model_name,
            operation="apply_lora",
        )
    status = ExecutionStatus.COMPLETED.value
    status_code = ExecutionStatus.COMPLETED.value
    error_message: str | None = None
    injected_count = 0
    try:
        injected_count = inject_fn()
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
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


__all__ = [
    "apply_lora_wrapper",
    "build_peft_result",
    "compute_lora_matrices",
    "compute_lora_scale",
    "matches_target_module",
]
