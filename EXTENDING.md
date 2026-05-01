# Extending Gemma-4-SQL

Gemma-4-SQL uses a flexible, plugin-based architecture powered by Python's `entry-points`. This means you can add support for entirely new ML frameworks (like Apple MLX, HuggingFace Accelerate, or ONNX) without modifying a single line of the core SDK code.

Backends can either live inside this repository (under `src/gemma_4_sql/backends/`) or in completely separate, third-party Python packages (e.g., `pip install gemma-4-sql-mlx`).

## How to Add a New Backend (Example: MLX)

Follow these steps to create a new backend plugin.

### 1. Create the Backend Package
Create a new Python module for your backend. If you are adding it to this repository, create a folder like `src/gemma_4_sql/backends/mlx/`.

### 2. Implement the `BackendProtocol`
Your backend must implement the functional interface defined in `src/gemma_4_sql/sdk/protocols.py`. You don't need to inherit from any base classes; Python's structural typing (duck typing) will ensure compatibility.

Create the necessary files (e.g., `train.py`, `inference.py`, `etl.py`) and implement the required functions.

**CRITICAL RULE: Lazy Import Heavy Dependencies!**
To prevent dependency conflicts and keep the CLI fast, **never** import heavy ML libraries at the top level of your files. Import them *inside* the functions instead:

```python
# src/gemma_4_sql/backends/mlx/train.py
from typing import Any

def train_model(
    action: str,
    model_name: str,
    dataset: str,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    # ✅ IMPORT HEAVY LIBRARIES HERE
    import mlx.core as mx
    import mlx.nn as nn
    
    # Implementation details...
    return {"status": "success", "backend": "mlx"}
```

### 3. Expose the API in `__init__.py`
The SDK will load your backend module and look for specific functions. Expose all the required protocol functions in your backend's root `__init__.py`:

```python
# src/gemma_4_sql/backends/mlx/__init__.py
from __future__ import annotations

def get_trainer() -> str:
    return "mlx_trainer"

from .train import train_model
from .inference import generate_sql
from .agent import run_agentic_loop
from .dpo import run_dpo
from .evaluate import evaluate_model
from .etl import build_dataloader
from .export import export_model
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .chat import chat_turn
from .few_shot import build_few_shot_prompt
from .serve import serve_model
from .benchmark import benchmark_model
```

### 4. Register the Entry Point
The SDK discovers backends by querying the Python package metadata. You need to register your module under the `gemma_4_sql.backends` group.

**If adding to the core repo:**
Edit the `pyproject.toml` file at the root of the Gemma-4-SQL repository:
```toml
[project.entry-points."gemma_4_sql.backends"]
jax = "gemma_4_sql.backends.jax"
keras = "gemma_4_sql.backends.keras"
maxtext = "gemma_4_sql.backends.maxtext"
pytorch = "gemma_4_sql.backends.pytorch"
mlx = "gemma_4_sql.backends.mlx"  # <-- Add your new backend here
```

**If creating a third-party plugin package:**
In your standalone project's `pyproject.toml` (e.g., a package named `gemma-4-sql-mlx`), add:
```toml
[project.entry-points."gemma_4_sql.backends"]
mlx = "my_custom_mlx_package.main"
```

### 5. Install and Test
Reinstall the project in editable mode so Python registers the new entry point:

```bash
pip install -e .
```

You can now immediately use your new backend through the SDK or CLI:

```python
from gemma_4_sql.sdk.models import train_from_scratch

# The SDK dynamically loads your MLX implementation!
train_from_scratch(model_name="gemma-2b", backend="mlx")
```
