"""JAX-specific model export pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.jax._protocols import is_jnp_missing
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    ExportResultDict,
    JaxArrayProtocol,
    JaxCausalLMProtocol,
)

type ExportWeightLeaf = Array | JaxArrayProtocol | float | int | bool | None
type ExportWeightsTree = Mapping[str, "ExportWeightLeaf | ExportWeightsTree"]


@runtime_checkable
class _NNXVariableValueProtocol(Protocol):
    """Protocol for an NNX `Variable` exposing `.raw_value`."""

    @property
    def raw_value(self) -> ExportWeightLeaf | ExportWeightsTree:
        """Provide the underlying array or scalar value."""
        ...


@runtime_checkable
class _NNXStateDictProtocol(Protocol):
    """Protocol for an NNX `State` exposing `.to_pure_dict()`."""

    def to_pure_dict(self) -> Mapping[str, ExportWeightLeaf | ExportWeightsTree | _NNXVariableValueProtocol]:
        """Convert state to a pure nested dictionary."""
        ...


@runtime_checkable
class _OrbaxSaveCheckpointerProtocol(Protocol):
    """Protocol for an Orbax `PyTreeCheckpointer` supporting `save` and `close`."""

    def save(self, directory: Path | str, item: ExportWeightsTree, /) -> None:
        """Persist `item` to `directory`."""
        ...

    def close(self) -> None:
        """Close background save resources."""
        ...


@runtime_checkable
class _ItemsMappingProtocol(Protocol):
    """Non-generic structural protocol for mappings exposing `.items()`."""

    def items(
        self,
    ) -> Iterable[
        tuple[str | int, BackendKwargValue | ExportWeightLeaf | ExportWeightsTree | _NNXVariableValueProtocol]
    ]:
        """Yield `(key, value)` pairs.

        Returns:
            Iterable of `(key, value)` tuples.

        """
        ...


def _normalize_export_tree(
    node: (
        BackendKwargValue
        | ModelConfig
        | JaxCausalLMProtocol
        | ExportWeightsTree
        | Mapping[str | int, ExportWeightLeaf | str | ExportWeightsTree | _NNXVariableValueProtocol]
        | Mapping[str, ExportWeightLeaf | ExportWeightsTree | _NNXVariableValueProtocol]
        | _NNXVariableValueProtocol
        | None
    ),
) -> dict[str, ExportWeightLeaf | ExportWeightsTree] | None:
    """Convert an NNX `State` or nested mapping into a string-keyed `ExportWeightsTree` dictionary.

    Args:
        node: Mapping or NNX `State` whose leaves are JAX arrays, NNX variables, or scalars.

    Returns:
        Recursively normalized string-keyed dictionary, or `None` when `node` is not a `Mapping`.

    """
    if isinstance(node, _NNXVariableValueProtocol) or not isinstance(node, Mapping):
        return None
    assert isinstance(node, _ItemsMappingProtocol)
    out: dict[str, ExportWeightLeaf | ExportWeightsTree] = {}
    for key, val in node.items():
        str_key = str(key)
        raw_val: BackendKwargValue | ExportWeightLeaf | ExportWeightsTree = (
            val.raw_value if isinstance(val, _NNXVariableValueProtocol) else val
        )
        sub = _normalize_export_tree(raw_val)
        if sub is not None:
            out[str_key] = sub
        elif isinstance(raw_val, (Array, JaxArrayProtocol, float, int, bool)) or raw_val is None:
            out[str_key] = raw_val
    return out


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
    if jax is None or is_jnp_missing(jnp) or ocp is None:
        raise DependencyMissingError("JAX export dependencies are missing.")

    def _execute() -> tuple[Path, str]:
        """Extract or resolve JAX model weights and persist them via Orbax `PyTreeCheckpointer`.

        Returns:
            Tuple of `(checkpoint_path, status_string)`.

        Raises:
            DependencyMissingError: If Flax NNX is unavailable during model extraction.
            ExportError: If state extraction or Orbax checkpoint persistence fails.

        """
        raw_weights = kwargs.get("weights", kwargs.get("params"))
        normalized_arg = _normalize_export_tree(raw_weights)
        if normalized_arg is not None:
            weights: ExportWeightsTree = normalized_arg
        else:
            try:
                from flax import nnx as local_nnx

                from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
                from gemma_4_sql.backends.jax.gemma4.config import resolve_gemma4_config
                from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE

                raw_cfg = kwargs.get("config")
                cached = _MODEL_CACHE.get_model(model_name) if raw_cfg is None else None
                cfg = raw_cfg if isinstance(raw_cfg, ModelConfig) else resolve_gemma4_config(Gemma4Config, model_name)
                model = (
                    cached if isinstance(cached, local_nnx.Module) else Gemma4ForCausalLM(cfg, rngs=local_nnx.Rngs(0))
                )
                state_fn: Callable[[nnx.Module], _NNXStateDictProtocol] = vars(local_nnx)["state"]
                state = state_fn(model)
                weights = _normalize_export_tree(state.to_pure_dict()) or {}
            except ImportError as e:
                raise DependencyMissingError(f"Flax NNX dependency missing: {e}") from e
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                raise ExportError(f"Failed to extract state for model '{model_name}': {e}") from e

        file_path = Path(export_path) / "orbax_ckpt"
        try:
            ckpt_factory: Callable[[], _OrbaxSaveCheckpointerProtocol] = vars(ocp)["PyTreeCheckpointer"]
            checkpointer = ckpt_factory()
            try:
                checkpointer.save(file_path, weights)
            finally:
                checkpointer.close()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            raise ExportError(f"Failed to save Orbax checkpoint at '{file_path}': {e}") from e

        return (file_path, "exported_with_orbax")

    return run_export_wrapper("jax", model_name, export_path, "orbax/saved_model", _execute)
