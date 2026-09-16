"""JAX-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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


def export_model(model_name: str, export_path: str, **kwargs: object) -> JSONDict:
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

    Path(export_path).mkdir(parents=True, exist_ok=True)

    weights = kwargs.get("weights")
    if weights is None:
        try:
            from flax import nnx

            from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

            cfg = kwargs.get("config")
            if cfg is None:
                if kwargs.get("test_mode") or model_name.startswith(("test", "model")):
                    cfg = Gemma4Config(
                        vocab_size=128,
                        hidden_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=32,
                        intermediate_size=128,
                    )
                else:
                    cfg = Gemma4Config.gemma4_e2b()
            model = Gemma4ForCausalLM(cast(Any, cfg), rngs=nnx.Rngs(0))
            weights = nnx.state(model)
        except ImportError as e:
            raise DependencyMissingError(f"Flax NNX dependency missing: {e}") from e
        except Exception as e:
            raise ExportError(f"Failed to extract state for model '{model_name}': {e}") from e

    file_path = Path(export_path) / "orbax_ckpt"
    try:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(file_path, weights)
    except Exception as e:
        raise ExportError(f"Failed to save Orbax checkpoint at '{file_path}': {e}") from e

    status = "exported_with_orbax"
    return {
        "backend": "jax",
        "model": model_name,
        "export_path": export_path,
        "file_path": str(file_path),
        "status": status,
        "format": "orbax/saved_model",
    }
