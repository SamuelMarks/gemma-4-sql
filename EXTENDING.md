# Extending Gemma-4-SQL

Gemma-4-SQL uses a flexible, plugin-based architecture powered by Python's `entry-points` and structural `Protocol` typing. This means you can add support for entirely new ML frameworks (like Apple MLX, HuggingFace Accelerate, or ONNX) without modifying a single line of the core SDK code.

Backends can either live inside this repository (under `src/gemma_4_sql/backends/`) or in completely separate, third-party Python packages (e.g., `pip install gemma-4-sql-mlx`).

## How to Add a New Backend (Example: MLX)

Follow these steps to create a new backend plugin.

### 1. Create the Backend Package
Create a new Python module for your backend. If you are adding it to this repository, create a folder like `src/gemma_4_sql/backends/mlx/`.

### 2. Implement the Backend Submodules & `BackendProtocol`
Your backend must implement the strongly-typed interface defined in `src/gemma_4_sql/sdk/protocols.py` (`BackendProtocol`, which combines `TrainingProtocol` and `InferenceProtocol`).

Create the submodules (`train.py`, `inference.py`, `etl.py`, `dpo.py`, `peft.py`, `quantize.py`, `serve.py`, `benchmark.py`, `export.py`, `logging.py`) and implement the required functions using the shared wrappers in `src/gemma_4_sql/backends/common_*.py` and strict type hints from `src/gemma_4_sql/type_hints.py` (never use `typing.Any`, `object`, `typing.cast`, or `getattr`).

**CRITICAL RULE: Lazy Import Heavy Dependencies via `optional_import` / `optional_attr`!**
To prevent dependency conflicts and keep the CLI fast, use `optional_import` or `optional_attr` from `gemma_4_sql.backends.lazy_loader` and raise `DependencyMissingError` when optional backend libraries are absent:

```python
# src/gemma_4_sql/backends/mlx/train.py
from __future__ import annotations

from gemma_4_sql.backends.common_train import run_train_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import TrainingConfig, TrainResultDict

mlx = optional_import("mlx.core")
nn = optional_import("mlx.nn")


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
    """Train a Text-to-SQL model using MLX.

    Args:
        config: Training configuration dataclass.
        **kwargs: Extra runtime options such as `'distributed_strategy'` or `'raise_on_error'`.

    Returns:
        Structured `TrainResultDict` containing training status and final metrics.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    if mlx is None or nn is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")

    def _execute() -> tuple[str, float]:
        # Execute native backend training loop using config.model_name, config.dataset, etc.
        return ("completed", 0.0)

    return run_train_wrapper("mlx", config, _execute, **kwargs)
```

### 3. Create `adapter.py` with `ModuleDelegatedBackend`
Each backend exposes an `adapter.py` module containing a `ModuleDelegatedBackend` subclass and an explicit `ADAPTER_CLASS: type[BackendProtocol]` binding (matching `AdapterModuleProtocol` in `src/gemma_4_sql/sdk/registry.py`):

```python
# src/gemma_4_sql/backends/mlx/adapter.py
from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend
from gemma_4_sql.sdk.protocols import BackendProtocol

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class MLXBackend(ModuleDelegatedBackend):
    """Unified MLX backend implementation."""

    def __init__(self) -> None:
        """Initialize the MLXBackend adapter with its submodule bindings."""
        super().__init__(
            trainer_name="mlx_trainer",
            train_mod=train,
            dpo_mod=dpo,
            etl_mod=etl,
            export_mod=export,
            logging_mod=logging,
            peft_mod=peft,
            quantize_mod=quantize,
            inference_mod=inference,
            serve_mod=serve,
            benchmark_mod=benchmark,
        )


ADAPTER_CLASS: type[BackendProtocol] = MLXBackend
```

### 4. Expose the API in `__init__.py`
Expose all protocol functions in your backend's root `__init__.py`:

```python
# src/gemma_4_sql/backends/mlx/__init__.py
from __future__ import annotations

from .benchmark import benchmark_model
from .dpo import run_dpo
from .etl import build_dataloader
from .export import export_model
from .inference import generate_sql
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .serve import serve_model
from .train import train_model

__all__ = [
    "apply_lora",
    "benchmark_model",
    "build_dataloader",
    "export_model",
    "generate_sql",
    "get_trainer",
    "log_metrics",
    "quantize_model",
    "run_dpo",
    "serve_model",
    "train_model",
]


def get_trainer() -> str:
    """Return the MLX trainer identifier.

    Returns:
        The resulting string.

    """
    return "mlx_trainer"
```

### 5. Register the Entry Point
The SDK discovers backends by querying the Python package metadata via `importlib.metadata.entry_points(group="gemma_4_sql.backends")`.

**If adding to the core repo:**
Edit the `pyproject.toml` file at the root of the Gemma-4-SQL repository:
```toml
[project.entry-points."gemma_4_sql.backends"]
jax = "gemma_4_sql.backends.jax"
keras = "gemma_4_sql.backends.keras"
maxtext = "gemma_4_sql.backends.maxtext"
pytorch = "gemma_4_sql.backends.pytorch"
pytorch_hf = "gemma_4_sql.backends.pytorch"
pytorch_native = "gemma_4_sql.backends.pytorch"
mlx = "gemma_4_sql.backends.mlx"  # <-- Add your new backend here
```

**If creating a third-party plugin package:**
In your standalone project's `pyproject.toml` (e.g., a package named `gemma-4-sql-mlx`), add:
```toml
[project.entry-points."gemma_4_sql.backends"]
mlx = "my_custom_mlx_package.main"
```

### 6. Install and Test
Reinstall the project in editable mode so Python registers the new entry point:

```bash
pip install -e .
```

You can now immediately use your new backend through the SDK or CLI:

```python
from gemma_4_sql.sdk.models import train_from_scratch
from gemma_4_sql.type_hints import TrainingConfig

# The SDK dynamically loads your MLX implementation!
config = TrainingConfig(model_name="gemma-2b", backend="mlx")
train_from_scratch(config)
```
