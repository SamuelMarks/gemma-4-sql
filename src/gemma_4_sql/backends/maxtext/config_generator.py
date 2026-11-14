"""MaxText Gin configuration generator and bridge.

Translates high-level TrainingConfig and distributed hyperparameters into
MaxText-compatible Gin configuration files and CLI argument lists.
"""

from __future__ import annotations

import atexit
import itertools
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from gemma_4_sql.type_hints import JSONValue, TrainingConfig

_MANAGED_GIN_TEMP_DIRS: list[Path] = []
_GIN_FILE_COUNTER = itertools.count(1)


def _slice_preset(dp: int, fsdp: int, tp: int, dcn_dp: int = 1) -> Mapping[str, int]:
    """Build an immutable ICI/DCN parallelism dictionary for a TPU slice SKU.

    Args:
        dp: ICI data parallelism degree.
        fsdp: ICI FSDP parallelism degree.
        tp: ICI tensor parallelism degree.
        dcn_dp: DCN multi-slice data parallelism degree.

    Returns:
        Read-only mapping of parallelism axis degrees.

    """
    return MappingProxyType({
        "ici_data_parallelism": dp,
        "ici_fsdp_parallelism": fsdp,
        "ici_tensor_parallelism": tp,
        "dcn_data_parallelism": dcn_dp,
    })


TPU_SLICE_TOPOLOGY_PRESETS: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "v4-8": _slice_preset(1, 4, 1),
    "v4-16": _slice_preset(1, 8, 1),
    "v4-32": _slice_preset(1, 8, 2),
    "v5litepod-4": _slice_preset(1, 4, 1),
    "v5litepod-8": _slice_preset(1, 8, 1),
    "v5litepod-16": _slice_preset(1, 8, 2),
    "v5e-8": _slice_preset(1, 8, 1),
    "v5p-8": _slice_preset(1, 4, 1),
    "v6e-4": _slice_preset(1, 4, 1),
    "v6e-8": _slice_preset(1, 8, 1),
    "v6e-16": _slice_preset(1, 8, 2),
})


def resolve_tpu_slice_parallelism(tpu_type: str | None) -> Mapping[str, int] | None:
    """Resolve optimal ICI/DCN parallelism settings for a Cloud TPU SKU (`v4-8`, `v5litepod-8`, `v6e-8`, etc.).

    Args:
        tpu_type: Raw TPU SKU or hardware string (e.g., `'v6e-8'`, `'tpu-v5litepod-8'`, `'v4-32'`).

    Returns:
        Read-only mapping of ICI/DCN parallelism integers, or `None` when unrecognized.

    """
    if not tpu_type:
        return None
    norm = tpu_type.strip().lower().removeprefix("tpu-")
    if norm in TPU_SLICE_TOPOLOGY_PRESETS:
        return TPU_SLICE_TOPOLOGY_PRESETS[norm]
    if norm.startswith("v6e"):
        return TPU_SLICE_TOPOLOGY_PRESETS["v6e-8"]
    if norm.startswith(("v5e", "v5litepod")):
        return TPU_SLICE_TOPOLOGY_PRESETS["v5litepod-8"]
    if norm.startswith("v5p"):
        return TPU_SLICE_TOPOLOGY_PRESETS["v5p-8"]
    if norm.startswith("v4"):
        return TPU_SLICE_TOPOLOGY_PRESETS["v4-8"]
    return None


def cleanup_temp_gin_configs() -> None:
    """Remove all managed temporary Gin configuration directories created during this process."""
    while _MANAGED_GIN_TEMP_DIRS:
        temp_dir = _MANAGED_GIN_TEMP_DIRS.pop()
        shutil.rmtree(temp_dir, ignore_errors=True)


atexit.register(cleanup_temp_gin_configs)


@dataclass
class MaxTextHyperparameters:
    """Hyperparameters and distributed topology settings for MaxText training.

    Attributes:
        model_architecture: Target Gemma 4 model architecture identifier
            (`'gemma4_e2b'`, `'gemma4_e4b'`, `'gemma4_26b_a4b'`, or `'gemma4_31b'`).
        per_device_batch_size: Batch size allocated per individual accelerator device.
        global_batch_size: Global effective batch size across all participating devices.
        learning_rate: Base peak learning rate for optimizer.
        learning_rate_schedule_steps: Total number of steps over which the schedule is computed.
        warmup_steps_fraction: Proportion of total steps dedicated to linear learning rate warmup.
        opt_type: Name of the optimizer (`'adamw'`, `'adam'`, `'sgd'`, etc.).
        mesh_axes: List of axis names defining the multi-dimensional device mesh topology.
        base_output_directory: Base filesystem or storage path for checkpoints and artifacts.
        run_name: Unique identifier for this specific training execution run.
        steps: Total training steps to execute.
        dataset_name: Dataset identifier used for loading and tokenization.
        checkpoint_period: Frequency in steps at which checkpoints are persisted.
        ici_fsdp_parallelism: Optional ICI FSDP parallelism degree for TPU slice topology.
        ici_tensor_parallelism: Optional ICI tensor parallelism degree for TPU slice topology.
        dcn_data_parallelism: Optional DCN multi-slice data parallelism degree.

    """

    model_architecture: str = "gemma4_e2b"
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
    ici_fsdp_parallelism: int | None = None
    ici_tensor_parallelism: int | None = None
    dcn_data_parallelism: int | None = None


def _to_float(val: JSONValue, default: float) -> float:
    """Safely convert a `JSONValue` to `float`.

    Args:
        val: Input `JSONValue`.
        default: Fallback float value.

    Returns:
        Converted float value.

    """
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float, str)):
        return float(val)
    return default


def _to_int(val: JSONValue, default: int) -> int:
    """Safely convert a `JSONValue` to `int`.

    Args:
        val: Input `JSONValue`.
        default: Fallback int value.

    Returns:
        Converted int value.

    """
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float, str)):
        return int(val)
    return default


def normalize_model_architecture(model_name: str) -> str:
    """Normalize user model identifier into a supported Gemma 4 MaxText model architecture.

    Args:
        model_name: Raw model identifier (e.g., `'gemma-4-e2b'`, `'gemma-4-e4b'`,
            `'gemma-4-26b-a4b'`, `'gemma-4-31b'`).

    Returns:
        Canonical MaxText Gemma 4 architecture string (`'gemma4_e2b'`, `'gemma4_e4b'`,
        `'gemma4_26b_a4b'`, or `'gemma4_31b'`).

    Raises:
        ValueError: If `model_name` is empty or whitespace-only.

    """
    if not model_name or not model_name.strip():
        msg = "model_name must be a non-empty string."
        raise ValueError(msg)

    cleaned = model_name.strip().lower().replace("-", "_")
    if "26b" in cleaned or "a4b" in cleaned or "moe" in cleaned:
        return "gemma4_26b_a4b"
    if "31b" in cleaned:
        return "gemma4_31b"
    if "e4b" in cleaned or "_4b" in cleaned or cleaned.endswith("4b"):
        return "gemma4_e4b"
    return "gemma4_e2b"


def build_maxtext_hyperparameters(config: TrainingConfig, **kwargs: JSONValue) -> MaxTextHyperparameters:
    """Build and validate `MaxTextHyperparameters` from `TrainingConfig` and optional overrides.

    Args:
        config: High-level `TrainingConfig` instance containing training metadata.
        **kwargs: Explicit keyword overrides for MaxText hyperparameters.

    Returns:
        Validated `MaxTextHyperparameters` instance.

    Raises:
        ValueError: If any hyperparameter violates numerical or structural constraints.

    """
    extra: dict[str, JSONValue] = {
        k: v for k, v in (config.extra_kwargs or {}).items() if isinstance(v, (str, int, float, bool)) or v is None
    }
    merged: dict[str, JSONValue] = {**extra, **kwargs}

    raw_arch = str(merged.get("model_architecture") or config.model_name or "gemma4_e2b")
    model_architecture = normalize_model_architecture(raw_arch)

    per_device = _to_float(merged.get("per_device_batch_size"), float(config.batch_size))
    if per_device <= 0:
        msg = f"per_device_batch_size must be positive, got {per_device}"
        raise ValueError(msg)

    global_batch = _to_int(merged.get("global_batch_size"), int(per_device * 4))
    if global_batch <= 0:
        msg = f"global_batch_size must be positive, got {global_batch}"
        raise ValueError(msg)

    learning_rate = _to_float(merged.get("learning_rate"), float(config.learning_rate))
    if learning_rate <= 0:
        msg = f"learning_rate must be positive, got {learning_rate}"
        raise ValueError(msg)

    epochs = int(config.epochs)
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
    mesh_axes: list[str] = [str(x) for x in raw_mesh]

    base_output_dir = str(merged.get("base_output_directory", "./maxtext_output"))
    dataset_name = str(merged.get("dataset_name", config.dataset))
    run_name = str(merged.get("run_name", f"{model_architecture}_{dataset_name}"))
    checkpoint_period = _to_int(merged.get("checkpoint_period"), 500)
    if checkpoint_period <= 0:
        msg = f"checkpoint_period must be positive, got {checkpoint_period}"
        raise ValueError(msg)

    raw_tpu_sku = merged.get("tpu_type") or merged.get("hardware")
    slice_preset = resolve_tpu_slice_parallelism(str(raw_tpu_sku)) if isinstance(raw_tpu_sku, str) else None
    ici_fsdp = (
        _to_int(merged["ici_fsdp_parallelism"], 1)
        if "ici_fsdp_parallelism" in merged and merged["ici_fsdp_parallelism"] is not None
        else (slice_preset["ici_fsdp_parallelism"] if slice_preset is not None else None)
    )
    ici_tp = (
        _to_int(merged["ici_tensor_parallelism"], 1)
        if "ici_tensor_parallelism" in merged and merged["ici_tensor_parallelism"] is not None
        else (slice_preset["ici_tensor_parallelism"] if slice_preset is not None else None)
    )
    dcn_dp = (
        _to_int(merged["dcn_data_parallelism"], 1)
        if "dcn_data_parallelism" in merged and merged["dcn_data_parallelism"] is not None
        else (slice_preset["dcn_data_parallelism"] if slice_preset is not None else None)
    )

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
        ici_fsdp_parallelism=ici_fsdp,
        ici_tensor_parallelism=ici_tp,
        dcn_data_parallelism=dcn_dp,
    )


def generate_maxtext_gin_config(config: TrainingConfig, **kwargs: JSONValue) -> str:
    """Generate a MaxText-compliant Gin configuration file string from `TrainingConfig`.

    Args:
        config: High-level `TrainingConfig` instance.
        **kwargs: Optional hyperparameter overrides.

    Returns:
        Formatted Gin configuration string.

    Raises:
        ValueError: If configuration validation fails.

    """
    params = build_maxtext_hyperparameters(config, **kwargs)
    mesh_axes_str = "[" + ", ".join(f"'{axis}'" for axis in params.mesh_axes) + "]"
    topology_lines = ""
    if params.ici_fsdp_parallelism is not None:
        topology_lines += f"ici_fsdp_parallelism = {params.ici_fsdp_parallelism}\n"
    if params.ici_tensor_parallelism is not None:
        topology_lines += f"ici_tensor_parallelism = {params.ici_tensor_parallelism}\n"
    if params.dcn_data_parallelism is not None:
        topology_lines += f"dcn_data_parallelism = {params.dcn_data_parallelism}\n"

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
{topology_lines}
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
        output_path: Optional explicit filesystem path. If `None`, a temporary file is created.

    Returns:
        Path pointing to the written Gin configuration file.

    Raises:
        ValueError: If `gin_content` is empty.
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

    if not _MANAGED_GIN_TEMP_DIRS or not _MANAGED_GIN_TEMP_DIRS[-1].exists():
        managed_dir = Path(tempfile.mkdtemp(prefix="gemma4_maxtext_gin_")).resolve()
        _MANAGED_GIN_TEMP_DIRS.append(managed_dir)
    else:
        managed_dir = _MANAGED_GIN_TEMP_DIRS[-1]
    target_path = (managed_dir / f"maxtext_config_{next(_GIN_FILE_COUNTER)}.gin").resolve()
    target_path.write_text(gin_content, encoding="utf-8")
    return target_path


def build_maxtext_cli_args(
    config: TrainingConfig,
    gin_config_path: str | Path | None = None,
    **kwargs: JSONValue,
) -> list[str]:
    """Construct CLI argument list for invoking `maxtext.train.main`.

    Args:
        config: High-level `TrainingConfig` instance.
        gin_config_path: Optional path to an already saved `.gin` configuration file.
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
