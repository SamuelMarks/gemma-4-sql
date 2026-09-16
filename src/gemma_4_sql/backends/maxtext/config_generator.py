"""MaxText Gin configuration generator and bridge.

Translates high-level TrainingConfig and distributed hyperparameters into
MaxText-compatible Gin configuration files and CLI argument lists.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import TrainingConfig


@dataclass
class MaxTextHyperparameters:
    """Hyperparameters and distributed topology settings for MaxText training.

    Attributes:
        model_architecture: Target model architecture identifier (e.g., 'gemma4_2b', 'gemma4_7b').
        per_device_batch_size: Batch size allocated per individual accelerator device.
        global_batch_size: Global effective batch size across all participating devices.
        learning_rate: Base peak learning rate for optimizer.
        learning_rate_schedule_steps: Total number of steps over which the schedule is computed.
        warmup_steps_fraction: Proportion of total steps dedicated to linear learning rate warmup.
        opt_type: Name of the optimizer ('adamw', 'adam', 'sgd', etc.).
        mesh_axes: List of axis names defining the multi-dimensional device mesh topology.
        base_output_directory: Base filesystem or storage path for checkpoints and artifacts.
        run_name: Unique identifier for this specific training execution run.
        steps: Total training steps to execute.
        dataset_name: Dataset identifier used for loading and tokenization.
        checkpoint_period: Frequency in steps at which checkpoints are persisted.
    """

    model_architecture: str = "gemma4_2b"
    per_device_batch_size: float = 2.0
    global_batch_size: int = 8
    learning_rate: float = 1e-4
    learning_rate_schedule_steps: int = 1000
    warmup_steps_fraction: float = 0.1
    opt_type: str = "adamw"
    mesh_axes: list[str] = field(default_factory=lambda: ["data", "fsdp", "tensor"])
    base_output_directory: str = "./maxtext_output"
    run_name: str = "gemma4_sql_run"
    steps: int = 1000
    dataset_name: str = "dummy"
    checkpoint_period: int = 500


def _to_float(val: object, default: float) -> float:
    """Safely convert an object value to float.

    Args:
        val: Input object value.
        default: Fallback float value.

    Returns:
        Converted float value.
    """
    if val is None:
        return default
    if isinstance(val, (int, float, str)):
        return float(val)
    return default


def _to_int(val: object, default: int) -> int:
    """Safely convert an object value to int.

    Args:
        val: Input object value.
        default: Fallback int value.

    Returns:
        Converted int value.
    """
    if val is None:
        return default
    if isinstance(val, (int, float, str)):
        return int(val)
    return default


def normalize_model_architecture(model_name: str) -> str:
    """Normalize user model identifier into a supported MaxText model architecture.

    Args:
        model_name: Raw model identifier (e.g., 'gemma-4', 'gemma-4-7b', 'gemma4_2b').

    Returns:
        Canonical MaxText model architecture string ('gemma4_2b' or 'gemma4_7b').

    Raises:
        ValueError: If model_name is empty or cannot be parsed.
    """
    if not model_name or not isinstance(model_name, str):
        msg = "model_name must be a non-empty string."
        raise ValueError(msg)

    cleaned = model_name.strip().lower()
    if "7b" in cleaned:
        return "gemma4_7b"
    return "gemma4_2b"


def build_maxtext_hyperparameters(config: TrainingConfig, **kwargs: object) -> MaxTextHyperparameters:
    """Build and validate MaxTextHyperparameters from TrainingConfig and optional overrides.

    Args:
        config: High-level TrainingConfig instance containing training metadata.
        **kwargs: Explicit keyword overrides for MaxText hyperparameters.

    Returns:
        Validated MaxTextHyperparameters instance.

    Raises:
        ValueError: If any hyperparameter violates numerical or structural constraints.
    """
    extra = dict(getattr(config, "extra_kwargs", {}) or {})
    merged: dict[str, object] = {**extra, **kwargs}

    raw_arch = str(merged.get("model_architecture") or getattr(config, "model_name", "gemma4_2b"))
    model_architecture = normalize_model_architecture(raw_arch)

    per_device = _to_float(merged.get("per_device_batch_size"), float(getattr(config, "batch_size", 2.0)))
    if per_device <= 0:
        msg = f"per_device_batch_size must be positive, got {per_device}"
        raise ValueError(msg)

    global_batch = _to_int(merged.get("global_batch_size"), int(per_device * 4))
    if global_batch <= 0:
        msg = f"global_batch_size must be positive, got {global_batch}"
        raise ValueError(msg)

    learning_rate = _to_float(merged.get("learning_rate"), float(getattr(config, "learning_rate", 1e-4)))
    if learning_rate <= 0:
        msg = f"learning_rate must be positive, got {learning_rate}"
        raise ValueError(msg)

    epochs = int(getattr(config, "epochs", 1))
    default_steps = max(1, epochs * 1000)
    steps = _to_int(merged.get("steps"), default_steps)
    if steps <= 0:
        msg = f"steps must be positive, got {steps}"
        raise ValueError(msg)

    lr_schedule_steps = _to_int(merged.get("learning_rate_schedule_steps"), steps)
    if lr_schedule_steps <= 0:
        msg = f"learning_rate_schedule_steps must be positive, got {lr_schedule_steps}"
        raise ValueError(msg)

    warmup_frac = _to_float(merged.get("warmup_steps_fraction"), 0.1)
    if not (0.0 <= warmup_frac <= 1.0):
        msg = f"warmup_steps_fraction must be between 0.0 and 1.0, got {warmup_frac}"
        raise ValueError(msg)

    opt_type = str(merged.get("opt_type", "adamw")).strip().lower()
    if not opt_type:
        msg = "opt_type cannot be empty."
        raise ValueError(msg)

    raw_mesh = merged.get("mesh_axes", ["data", "fsdp", "tensor"])
    if not isinstance(raw_mesh, list) or not all(isinstance(x, str) for x in raw_mesh):
        msg = "mesh_axes must be a list of strings."
        raise ValueError(msg)
    mesh_axes = list(raw_mesh)

    base_output_dir = str(merged.get("base_output_directory", "./maxtext_output"))
    dataset_name = str(merged.get("dataset_name", getattr(config, "dataset", "dummy")))
    run_name = str(merged.get("run_name", f"{model_architecture}_{dataset_name}"))
    checkpoint_period = _to_int(merged.get("checkpoint_period"), 500)
    if checkpoint_period <= 0:
        msg = f"checkpoint_period must be positive, got {checkpoint_period}"
        raise ValueError(msg)

    return MaxTextHyperparameters(
        model_architecture=model_architecture,
        per_device_batch_size=per_device,
        global_batch_size=global_batch,
        learning_rate=learning_rate,
        learning_rate_schedule_steps=lr_schedule_steps,
        warmup_steps_fraction=warmup_frac,
        opt_type=opt_type,
        mesh_axes=mesh_axes,
        base_output_directory=base_output_dir,
        run_name=run_name,
        steps=steps,
        dataset_name=dataset_name,
        checkpoint_period=checkpoint_period,
    )


def generate_maxtext_gin_config(config: TrainingConfig, **kwargs: object) -> str:
    """Generate a MaxText-compliant Gin configuration file string from TrainingConfig.

    Args:
        config: High-level TrainingConfig instance.
        **kwargs: Optional hyperparameter overrides.

    Returns:
        Formatted Gin configuration string.

    Raises:
        ValueError: If configuration validation fails.
    """
    params = build_maxtext_hyperparameters(config, **kwargs)
    mesh_axes_str = "[" + ", ".join(f"'{axis}'" for axis in params.mesh_axes) + "]"

    return f"""# MaxText Configuration for Gemma-4 SQL
# Auto-generated by gemma_4_sql.backends.maxtext.config_generator

# Global run settings
base_output_directory = "{params.base_output_directory}"
run_name = "{params.run_name}"

# Model architecture
model_name = "{params.model_architecture}"

# Batch size and distributed topology
per_device_batch_size = {params.per_device_batch_size}
global_batch_size = {params.global_batch_size}
mesh_axes = {mesh_axes_str}

# Optimizer & learning rate schedule
learning_rate = {params.learning_rate}
opt_type = "{params.opt_type}"
steps = {params.steps}
learning_rate_schedule_steps = {params.learning_rate_schedule_steps}
warmup_steps_fraction = {params.warmup_steps_fraction}

# Dataset & checkpointing
dataset_name = "{params.dataset_name}"
checkpoint_period = {params.checkpoint_period}
"""


def save_maxtext_gin_config(gin_content: str, output_path: str | Path | None = None) -> Path:
    """Save Gin configuration text to a file.

    Args:
        gin_content: String containing formatted Gin configuration lines.
        output_path: Optional explicit filesystem path. If None, a temporary file is created.

    Returns:
        Path pointing to the written Gin configuration file.

    Raises:
        ValueError: If gin_content is empty.
        OSError: If writing to the filesystem fails.
    """
    if not gin_content or not gin_content.strip():
        msg = "gin_content cannot be empty."
        raise ValueError(msg)

    if output_path is not None:
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(gin_content, encoding="utf-8")
        return target_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".gin", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(gin_content)
        return Path(temp_file.name).resolve()


def build_maxtext_cli_args(config: TrainingConfig, gin_config_path: str | Path | None = None, **kwargs: object) -> list[str]:
    """Construct CLI argument list for invoking maxtext.train.main.

    Args:
        config: High-level TrainingConfig instance.
        gin_config_path: Optional path to an already saved .gin configuration file.
        **kwargs: Optional hyperparameter overrides.

    Returns:
        List of command-line argument strings passed to MaxText.

    Raises:
        ValueError: If configuration generation fails.
    """
    if gin_config_path is None:
        gin_str = generate_maxtext_gin_config(config, **kwargs)
        path = save_maxtext_gin_config(gin_str)
    else:
        path = Path(gin_config_path).resolve()

    return ["train.py", str(path)]
