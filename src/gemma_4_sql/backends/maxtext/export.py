"""MaxText-specific model export pipeline with cached `(model, params)` resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import (
    JaxPRNGHolderProtocol,
    JnpZerosInt32Protocol,
    OrbaxCheckpointModuleProtocol,
)
from gemma_4_sql.backends.maxtext.inference import cache_maxtext_params, get_cached_maxtext_params
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import (
    ExportResultDict,
    JSONValue,
    PyTreeNode,
    TensorType,
)

jax: JaxPRNGHolderProtocol | None = optional_import("jax", JaxPRNGHolderProtocol)
jnp: JnpZerosInt32Protocol | None = optional_import("jax.numpy", JnpZerosInt32Protocol)
ocp: OrbaxCheckpointModuleProtocol | None = optional_import("orbax.checkpoint", OrbaxCheckpointModuleProtocol)
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)


def export_model(
    model_name: str,
    export_path: str,
    **kwargs: StatusValue | JSONValue | Mapping[str, TensorType | PyTreeNode],
) -> ExportResultDict:
    """Export a Text-to-SQL model using the MaxText backend.

    Resolves weights from explicit `params` / `weights` kwargs, from cached MaxText
    `(model, params)` state (`get_cached_maxtext_params(model_name)`), or by initializing
    `Gemma4Model(model_name)` once and caching the initialized weights.

    Args:
        model_name: The name or identifier of the target model.
        export_path: Destination directory where the Orbax checkpoint will be saved.
        **kwargs: Optional keyword arguments, including `'params'` or `'weights'`.

    Returns:
        Dictionary containing backend, model name, export path, file path, status, and format.

    Raises:
        DependencyMissingError: If required export dependencies (JAX, Orbax, or MaxText) are missing.
        ExportError: If model weight initialization or Orbax checkpoint persistence fails.

    """
    if jax is None or jnp is None or ocp is None:
        raise DependencyMissingError("MaxText export dependencies (jax, orbax.checkpoint) are missing.")

    def _execute() -> tuple[Path, str]:
        """Resolve or initialize MaxText model weights and save an Orbax checkpoint.

        Returns:
            Tuple of `(orbax_checkpoint_dir_path, export_status_message)`.

        Raises:
            DependencyMissingError: If `Gemma4Model` is unavailable when initializing weights.
            ExportError: If MaxText model initialization or Orbax checkpoint saving fails.

        """
        assert jax is not None
        assert jnp is not None
        assert ocp is not None
        weights = kwargs.get("params", kwargs.get("weights"))
        if weights is None:
            weights = get_cached_maxtext_params(model_name)
        if weights is None:
            if Gemma4Model is None:
                raise DependencyMissingError("MaxText dependency (maxtext.models.gemma4.Gemma4Model) is missing.")
            try:
                model = Gemma4Model(model_name)
                rng = jax.random.PRNGKey(0)
                dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
                init_weights = model.init(rng, dummy_input)
                weights = init_weights
                cache_maxtext_params(
                    model_name,
                    {str(k): v for k, v in init_weights.items()},
                    model=model,
                )
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                raise ExportError(
                    f"Failed to initialize MaxText model '{model_name}': {e}",
                    backend="maxtext",
                    model_name=model_name,
                    cause=e,
                ) from e

        file_path = Path(export_path) / "maxtext_orbax_ckpt"
        try:
            options = ocp.CheckpointManagerOptions(max_to_keep=1)
            with ocp.CheckpointManager(file_path, ocp.PyTreeCheckpointer(), options) as mngr:
                mngr.save(0, weights)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            raise ExportError(
                f"Failed to save MaxText Orbax checkpoint at '{file_path}': {e}",
                backend="maxtext",
                model_name=model_name,
                cause=e,
            ) from e

        return (file_path, "exported_with_maxtext_orbax")

    return run_export_wrapper("maxtext", model_name, export_path, "maxtext/checkpoint", _execute)
