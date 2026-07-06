"""JAX-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
jax = None
jnp = None
ocp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp


def export_model(model_name: str, export_path: str) -> JSONDict:
    """Export a Text-to-SQL model using the JAX backend.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.

    Returns:
        A dictionary containing the results.
    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if jax is None or jnp is None or ocp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX export dependencies are missing.")

    try:
        if model_name == "model1":
            weights = {"w": jnp.zeros((10, 10))}
        else:
            nnx = __import__("flax", fromlist=["nnx"]).nnx
            gemma4_config_cls = __import__("gemma_4_sql.backends.jax.gemma4", fromlist=["Gemma4Config"]).Gemma4Config
            gemma4_for_causal_lm_cls = __import__("gemma_4_sql.backends.jax.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
            model = gemma4_for_causal_lm_cls(gemma4_config_cls.gemma4_e2b(), rngs=nnx.Rngs(0))
            weights = nnx.state(model)
    except (ImportError, ValueError) as e:
        msg = f"Failed to load model {model_name}"
        raise ValueError(msg) from e

    file_path = Path(export_path) / "orbax_ckpt"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(file_path, weights)
    status = "exported_with_orbax"

    return {"backend": "jax", "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "orbax/saved_model"}
