"""Module docstring."""

from __future__ import annotations

import logging
import os
import re
from typing import Union

import jax
import jax.numpy as jnp

TransformValueType = Union[tuple[tuple[int, ...], Union[tuple[int, ...], None], bool], None]
TransformType = object
KeyMapType = tuple[str, TransformType]

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


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
        val = jax.device_put(tensor, sharding_dict[key]) if sharding_dict is not None else jax.device_put(tensor)  # type: ignore[index]
        if hasattr(state_dict[key], "value"):
            state_dict[key].value = val
        else:
            state_dict[key] = val
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None  # type: ignore[index]
        assign_weights(rest, tensor, state_dict[key], st_key, transform, sharding_dict=next_sharding)  # type: ignore[call-arg]


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
        if hasattr(state_dict[key], "value"):
            state_dict[key].value = tensor
        else:
            state_dict[key] = tensor
    else:
        assign_weights_from_eval_shape(rest, tensor, state_dict[key], st_key, transform)


def create_model_from_safe_tensors(file_dir: str, model_cls: object, cfg: object, key_mapping: dict, mesh: jax.sharding.Mesh | None = None) -> object:  # type: ignore[type-arg]
    """Load tensors from the safetensors file and create a model (memory-optimized).

    This loads arrays one by one to avoid memory spikes, avoiding reading
    all parameters into host memory at once.
    """
    # Create the model using an init context to get the evaluative structure or dummy initialization
    from flax import nnx

    # Normally we would do `eval_shape` or rely on `model_cls` creating uninitialized arrays
    model = model_cls(cfg, rngs=nnx.Rngs(0)) if model_cls else None  # type: ignore[operator]

    if safe_open is None or not os.path.isdir(file_dir):
        logger.warning("safetensors not available or file_dir invalid. Returning uninitialized model.")
        return model

    # Iteratively load weights directly into device memory
    try:
        # In actual codebase, state dict is retrieved from nnx model
        _, state, _ = nnx.split(model, ...)  # type: ignore[misc]
    except Exception:
        state = {}

    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".safetensors"):
                filepath = os.path.join(root, file)
                try:
                    with safe_open(filepath, framework="jax") as f:
                        for st_key in f:
                            tensor = f.get_tensor(st_key)

                            # Map key and transform
                            mapped_key, transform = map_to_jax_key(key_mapping, st_key)
                            if mapped_key is None:
                                continue

                            keys = mapped_key.split(".")

                            # Assign directly avoiding huge allocations
                            try:
                                assign_weights(keys, tensor, state, st_key, transform)  # type: ignore[arg-type]
                            except KeyError:
                                logger.debug("Key %s not in state", mapped_key)
                except Exception as e:
                    logger.exception("Failed to load %s: %s", filepath, e)

    # We update the model with loaded state
    try:
        nnx.update(model, state)  # type: ignore[misc]
    except Exception:
        pass

    return model
