"""MaxText-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    import jax as _jax
    import jax.numpy as _jnp
    import orbax.checkpoint as _ocp

    jax: Any = _jax
    jnp: Any = _jnp
    ocp: Any = _ocp
except (ImportError, AttributeError):
    jax = None
    jnp = None
    ocp = None

try:
    from maxtext.models.gemma4 import Gemma4Model as _Gemma4Model

    Gemma4Model: Any = _Gemma4Model
except (ImportError, AttributeError):
    Gemma4Model = None


def export_model(model_name: str, export_path: str, **kwargs: object) -> JSONDict:
    """Export a Text-to-SQL model using the MaxText backend.

    Args:
        model_name: The name or identifier of the target model.
        export_path: Destination directory where the Orbax checkpoint will be saved.
        **kwargs: Optional keyword arguments, including 'params' or 'weights'.

    Returns:
        Dictionary containing backend, model name, export path, file path, status, and format.

    Raises:
        DependencyMissingError: If required export dependencies (JAX, Orbax, or MaxText) are missing.
        ExportError: If model weight initialization or Orbax checkpoint persistence fails.
    """
    from gemma_4_sql.exceptions import DependencyMissingError, ExportError

    if jax is None or jnp is None or ocp is None:
        raise DependencyMissingError("MaxText export dependencies (jax, orbax.checkpoint) are missing.")

    Path(export_path).mkdir(parents=True, exist_ok=True)

    weights: Any = kwargs.get("params", kwargs.get("weights"))
    if weights is None:
        if Gemma4Model is None:
            raise DependencyMissingError("MaxText dependency (maxtext.models.gemma4.Gemma4Model) is missing.")
        try:
            model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            weights = model.init(rng, dummy_input)
        except Exception as e:
            raise ExportError(f"Failed to initialize MaxText model '{model_name}': {e}") from e

    file_path = Path(export_path) / "maxtext_orbax_ckpt"
    try:
        options = ocp.CheckpointManagerOptions(max_to_keep=1)
        with ocp.CheckpointManager(file_path, ocp.PyTreeCheckpointer(), options) as mngr:
            mngr.save(0, weights)
    except Exception as e:
        raise ExportError(f"Failed to save MaxText Orbax checkpoint at '{file_path}': {e}") from e

    status = "exported_with_maxtext_orbax"
    return {
        "backend": "maxtext",
        "model": model_name,
        "export_path": export_path,
        "file_path": str(file_path),
        "status": status,
        "format": "maxtext/checkpoint",
    }
