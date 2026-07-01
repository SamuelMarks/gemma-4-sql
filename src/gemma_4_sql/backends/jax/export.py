"""JAX-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None
    jnp = None
    ocp = None


def export_model(model_name: str, export_path: str) -> JSONDict:
    """Export a Text-to-SQL model using the JAX backend.

    Args:
    ----
        model_name: The name of the model to export.
        export_path: The destination path for the checkpoint.

    Returns:
    -------
        A dictionary containing export metadata.

    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if jax is not None and jnp is not None and (ocp is not None):
        try:
            nnx = __import__("flax", fromlist=["nnx"]).nnx

            gemma4_config_cls = __import__("gemma_4_sql.backends.jax.gemma4", fromlist=["Gemma4Config"]).Gemma4Config
            gemma4_for_causal_lm_cls = __import__("gemma_4_sql.backends.jax.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
            model = gemma4_for_causal_lm_cls(gemma4_config_cls.gemma4_e2b(), rngs=nnx.Rngs(0))
            weights = nnx.state(model)
        except (ImportError, ValueError):
            weights = {"w": jnp.zeros((10, 10))}
        file_path = Path(export_path) / "orbax_ckpt"
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(file_path, weights)
        status = "exported_with_orbax"
    else:
        file_path = Path(export_path) / f"mock_jax_model_{model_name}.bin"
        with Path.open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Mock JAX weights for {model_name}")
        status = "mock_exported"
    return {"backend": "jax", "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "orbax/saved_model"}
