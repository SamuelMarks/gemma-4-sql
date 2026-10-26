"""JAX-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.type_hints import ExportResultDict

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
ocp = optional_import("orbax.checkpoint")


def export_model(model_name: str, export_path: str, **kwargs: object) -> ExportResultDict:
    """Export a Text-to-SQL model using the JAX backend.

    Args:
        model_name: The name or identifier of the target model.
        export_path: Destination directory where the Orbax checkpoint will be saved.
        **kwargs: Optional keyword arguments, including 'config', 'model', or 'weights'.

    Returns:
        Dictionary containing backend, model name, export path, file path, status, and format.

    Raises:
        DependencyMissingError: If JAX export dependencies (jax, orbax.checkpoint, flax) are missing.
        ExportError: If model state extraction or checkpoint saving fails.

    """
    from gemma_4_sql.exceptions import DependencyMissingError, ExportError

    if jax is None or jnp is None or ocp is None:
        raise DependencyMissingError("JAX export dependencies are missing.")

    def _execute() -> tuple[Path, str]:
        """Extract or resolve JAX model weights and persist them via Orbax PyTreeCheckpointer."""
        weights: Any = kwargs.get("weights", kwargs.get("params"))
        if weights is None:
            try:
                from flax import nnx

                from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
                from gemma_4_sql.backends.jax.gemma4.config import resolve_gemma4_config
                from gemma_4_sql.backends.jax.inference import _MODEL_CACHE

                cached = _MODEL_CACHE.get_model(model_name) if kwargs.get("config") is None else None
                cfg: Any = kwargs.get("config") or resolve_gemma4_config(Gemma4Config, model_name)
                model = (cached[0] if isinstance(cached, tuple) else cached) if cached is not None else Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
                state = nnx.state(model)
                weights = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, "value") else x, state)
            except ImportError as e:
                raise DependencyMissingError(f"Flax NNX dependency missing: {e}") from e
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                raise ExportError(f"Failed to extract state for model '{model_name}': {e}") from e

        file_path = Path(export_path) / "orbax_ckpt"
        try:
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(file_path, weights)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            raise ExportError(f"Failed to save Orbax checkpoint at '{file_path}': {e}") from e

        return (file_path, "exported_with_orbax")

    return run_export_wrapper("jax", model_name, export_path, "orbax/saved_model", _execute)
