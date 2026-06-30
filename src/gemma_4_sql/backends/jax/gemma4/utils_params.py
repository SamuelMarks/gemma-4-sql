"""Module docstring."""

from __future__ import annotations

import logging
import re

import jax
import jax.numpy as jnp

TransformValueType = tuple[tuple[int, ...], tuple[int, ...] | None, bool] | None
TransformType = object
KeyMapType = tuple[str, TransformType]

logger = logging.getLogger(__name__)


def map_to_jax_key(mapping: dict[str, KeyMapType], source_key: str) -> KeyMapType | tuple[None, None]:
    """Map a safetensors key to exactly one JAX key & transform, else warn/error."""
    subs = [(re.sub(pat, repl, source_key), transform) for (pat, (repl, transform)) in mapping.items() if re.match(pat, source_key)]
    if not subs:
        logger.warning("No mapping found for key: %s", source_key)
        return (None, None)
    if len(subs) > 1:
        keys = [s for (s, _) in subs]
        msg = f"Multiple mappings found for {source_key!r}: {keys}"
        raise ValueError(msg)
    return subs[0]


def stoi(s: str) -> int | str:
    """Convert a string to an int if possible, otherwise return the string."""
    try:
        return int(s)
    except ValueError:
        return s


def assign_weights(keys: list[str], tensor: jnp.ndarray, state_dict: dict, st_key: str, transform: TransformType, **kwargs: object) -> object:  # type: ignore[return, type-arg]
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.

    Assumes that the state_dict values are of type Array.
    """
    sharding_dict = kwargs.get("sharding_dict")
    (key, *rest) = keys
    if not rest:
        if transform is not None:
            (permute, reshape, reshape_first) = transform  # type: ignore[misc]
            if reshape_first and reshape is not None:  # type: ignore[has-type]
                tensor = tensor.reshape(reshape)  # type: ignore[has-type]
            if permute:  # type: ignore[has-type]
                tensor = tensor.transpose(permute)  # type: ignore[has-type]
            if not reshape_first and reshape is not None:  # type: ignore[has-type]
                tensor = tensor.reshape(reshape)  # type: ignore[has-type]
        if tensor.shape != (state_dict[key].value.shape if hasattr(state_dict[key], "value") else getattr(state_dict[key], "shape", ())):
            msg = f"Shape mismatch for {st_key}: {tensor.shape} vs {(state_dict[key].value.shape if hasattr(state_dict[key], 'value') else getattr(state_dict[key], 'shape', ()))}"
            raise ValueError(msg)
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])  # type: ignore[index]
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None  # type: ignore[index]
        assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)  # type: ignore[call-arg]


def assign_weights_from_eval_shape(keys: list[str], tensor: jnp.ndarray, state_dict: dict, st_key: str, transform: TransformType) -> object:  # type: ignore[return, type-arg]
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.

    Assumes that the state_dict values are of type ShapeDtypeStruct.
    """
    (key, *rest) = keys
    if not rest:
        if transform is not None:
            (permute, reshape, reshape_first) = transform  # type: ignore[misc]
            if reshape_first and reshape is not None:  # type: ignore[has-type]
                tensor = tensor.reshape(reshape)  # type: ignore[has-type]
            if permute:  # type: ignore[has-type]
                tensor = tensor.transpose(permute)  # type: ignore[has-type]
            if not reshape_first and reshape is not None:  # type: ignore[has-type]
                tensor = tensor.reshape(reshape)  # type: ignore[has-type]
        if tensor.shape != (state_dict[key].value.shape if hasattr(state_dict[key], "value") else getattr(state_dict[key], "shape", ())):
            msg = f"Shape mismatch for {st_key}: {tensor.shape} vs {(state_dict[key].value.shape if hasattr(state_dict[key], 'value') else getattr(state_dict[key], 'shape', ()))}"
            raise ValueError(msg)
        tensor = tensor.astype(state_dict[key].value.dtype if hasattr(state_dict[key], "value") else getattr(state_dict[key], "dtype", None))
        if hasattr(getattr(state_dict[key], "value", state_dict[key]), "sharding") and getattr(state_dict[key], "value", state_dict[key]).sharding is not None:
            tensor = jax.device_put(tensor, getattr(state_dict[key], "value", state_dict[key]).sharding.spec)
        state_dict[key] = type(state_dict[key])(state_dict[key].type, tensor) if hasattr(state_dict[key], "type") else tensor
    else:
        assign_weights_from_eval_shape(rest, tensor, state_dict[key], st_key, transform)


def create_model_from_safe_tensors(_file_dir: str, _model_cls: object, _cfg: object, _key_mapping: dict, _mesh: jax.sharding.Mesh | None = None) -> object:  # type: ignore[type-arg]
    """Load tensors from the safetensors file and create a model (memory-optimized).

    We currently define this separately for each model, but this may be a useful tool later
    NOTE: This is not yet implemented.
    """
    msg = "This is in progress."
    raise NotImplementedError(msg)
