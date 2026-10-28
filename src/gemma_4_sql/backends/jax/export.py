"""JAX-specific model export pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax import Array

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    ExportResultDict,
    JaxArrayProtocol,
    JaxCausalLMProtocol,
    OrbaxCheckpointerProtocol,
)

type ExportWeightLeaf = Array | JaxArrayProtocol | float | int
type ExportWeightsTree = Mapping[str, ExportWeightLeaf | Mapping[str, ExportWeightLeaf]]


def export_model(
    model_name: str,
    export_path: str,
    **kwargs: BackendKwargValue | ModelConfig | JaxCausalLMProtocol | ExportWeightsTree,
) -> ExportResultDict:
    """Export a Text-to-SQL model using the JAX backend.

    Args:
        model_name: The name or identifier of the target model.
        export_path: Destination directory where the Orbax checkpoint will be saved.
        **kwargs: Optional keyword arguments, including `'config'`, `'model'`, or `'weights'`.

    Returns:
        Dictionary containing backend, model name, export path, file path, status, and format.

    Raises:
        DependencyMissingError: If JAX export dependencies (`jax`, `orbax.checkpoint`, `flax`) are missing.
        ExportError: If model state extraction or checkpoint saving fails.

    """
    if jax is None or jnp is None or ocp is None:
        raise DependencyMissingError("JAX export dependencies are missing.")

    def _execute() -> tuple[Path, str]:
        """Extract or resolve JAX model weights and persist them via Orbax `PyTreeCheckpointer`."""
        raw_weights = kwargs.get("weights", kwargs.get("params"))
        if isinstance(raw_weights, Mapping):
            weights: ExportWeightsTree = raw_weights  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
        else:
            try:
                from flax import nnx

                from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
                from gemma_4_sql.backends.jax.gemma4.config import resolve_gemma4_config
                from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE

                raw_cfg = kwargs.get("config")
                cached = _MODEL_CACHE.get_model(model_name) if raw_cfg is None else None
                cfg = raw_cfg if isinstance(raw_cfg, ModelConfig) else resolve_gemma4_config(Gemma4Config, model_name)
                model = cached if cached is not None else Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
                state = nnx.state(model)

                def _unwrap_leaf(x: nnx.Variable[Array] | Array) -> Array:
                    """Extract array value from NNX `Variable` leaves if present."""
                    return x[...] if isinstance(x, nnx.Variable) else x

                weights = jax.tree_util.tree_map(_unwrap_leaf, state)  # pyright: ignore[reportAssignmentType]
            except ImportError as e:
                raise DependencyMissingError(f"Flax NNX dependency missing: {e}") from e
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                raise ExportError(f"Failed to extract state for model '{model_name}': {e}") from e

        file_path = Path(export_path) / "orbax_ckpt"
        try:
            checkpointer: OrbaxCheckpointerProtocol[ExportWeightsTree] = ocp.PyTreeCheckpointer()
            checkpointer.save(file_path, weights)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            raise ExportError(f"Failed to save Orbax checkpoint at '{file_path}': {e}") from e

        return (file_path, "exported_with_orbax")

    return run_export_wrapper("jax", model_name, export_path, "orbax/saved_model", _execute)
