"""Core functionality for the utils_params module."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import jax

TransformValueType = Optional[tuple[tuple[int, ...], Optional[tuple[int, ...]], bool]]
TransformType = Any
KeyMapType = tuple[str, TransformType]
logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open as _safe_open

    safe_open: Any = _safe_open
except ImportError:
    safe_open = None


def map_to_jax_key(mapping: dict[str, KeyMapType], source_key: str) -> KeyMapType | tuple[None, None]:
    """Map a safetensors key to exactly one JAX key & transform, else warn/error.

    Args:
        mapping: A mapping representing mapping.
        source_key: The string representing the source key.

    Returns:
        A tuple containing the results.

    Raises:
        ValueError: If multiple mappings are found for source_key.
    """
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
    """Convert a string to an int if possible, otherwise return the string.

    Args:
        s: The string representing the s.

    Returns:
        The execution result.
    """
    try:
        return int(s)
    except ValueError:
        return s


def _apply_transform(tensor: Any, transform: Any) -> Any:
    """Apply transformation to tensor.

    Args:
        tensor: The input tensor.
        transform: The transform.

    Returns:
        The resulting tensor array.
    """
    if transform is None:
        return tensor
    (permute, reshape, reshape_first) = transform
    if reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)
    if permute:  # pragma: no cover
        tensor = tensor.transpose(permute)
    if not reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)
    return tensor


def assign_weights(keys: list[str], tensor: Any, state_dict: Any, st_key: str, transform: Any, **kwargs: Any) -> None:
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.

    Args:
        keys: List of string keys.
        tensor: Tensor to assign.
        state_dict: State dict to assign into.
        st_key: Original safetensors key name.
        transform: Transform specification.
        **kwargs: Optional keyword arguments for advanced configuration.

    Raises:
        ValueError: If the operation encounters an unexpected ValueError.
    """
    sharding_dict = kwargs.get("sharding_dict")
    (key, *rest) = keys
    resolved_key: Any = key
    if hasattr(state_dict, "__contains__") and resolved_key not in state_dict:
        if isinstance(resolved_key, str) and resolved_key.isdigit() and int(resolved_key) in state_dict:
            resolved_key = int(resolved_key)
        elif isinstance(resolved_key, int) and str(resolved_key) in state_dict:
            resolved_key = str(resolved_key)
    if not rest:
        tensor = _apply_transform(tensor, transform)
        if tensor.shape != (state_dict[resolved_key].value.shape if hasattr(state_dict[resolved_key], "value") else getattr(state_dict[resolved_key], "shape", ())):
            msg = f"Shape mismatch for {st_key}: {tensor.shape} vs {(state_dict[resolved_key].value.shape if hasattr(state_dict[resolved_key], 'value') else getattr(state_dict[resolved_key], 'shape', ()))}"
            raise ValueError(msg)
        val = jax.device_put(tensor, sharding_dict[resolved_key]) if sharding_dict is not None else jax.device_put(tensor)
        if hasattr(state_dict[resolved_key], "value"):
            state_dict[resolved_key].value = val
        else:
            state_dict[resolved_key] = val
    else:
        next_sharding = sharding_dict[resolved_key] if sharding_dict is not None else None
        assign_weights(rest, tensor, state_dict[resolved_key], st_key, transform, sharding_dict=next_sharding)


def assign_weights_from_eval_shape(keys: list[str], tensor: Any, state_dict: Any, st_key: str, transform: Any) -> None:
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.

    Args:
        keys: List of string keys.
        tensor: Tensor to assign.
        state_dict: State dict to assign into.
        st_key: Original safetensors key name.
        transform: Transform specification.

    Raises:
        ValueError: If the operation encounters an unexpected ValueError.
    """
    (key, *rest) = keys
    resolved_key: Any = key
    if hasattr(state_dict, "__contains__") and resolved_key not in state_dict:
        if isinstance(resolved_key, str) and resolved_key.isdigit() and int(resolved_key) in state_dict:
            resolved_key = int(resolved_key)
        elif isinstance(resolved_key, int) and str(resolved_key) in state_dict:
            resolved_key = str(resolved_key)
    if not rest:
        tensor = _apply_transform(tensor, transform)
        val_obj = state_dict[resolved_key]
        expected_shape = val_obj.value.shape if hasattr(val_obj, "value") else getattr(val_obj, "shape", ())
        if tensor.shape != expected_shape:
            msg = f"Shape mismatch for {st_key}: {tensor.shape} vs {expected_shape}"
            raise ValueError(msg)
        expected_dtype = val_obj.value.dtype if hasattr(val_obj, "value") else getattr(val_obj, "dtype", None)
        tensor = tensor.astype(expected_dtype)
        target = getattr(val_obj, "value", val_obj)
        if hasattr(target, "sharding") and target.sharding is not None:
            shd_spec = getattr(target.sharding, "spec", target.sharding)
            tensor = jax.device_put(tensor, shd_spec)
        if hasattr(val_obj, "value"):
            val_obj.value = tensor  # pragma: no cover
        else:
            state_dict[resolved_key] = tensor
    else:
        assign_weights_from_eval_shape(rest, tensor, state_dict[resolved_key], st_key, transform)


def _load_weights_from_safetensors_file(filepath: str, state: dict[str, object], key_mapping: dict[str, KeyMapType]) -> None:
    """Load weights from a single safetensors file."""
    try:
        with safe_open(filepath, framework="jax") as f:
            for st_key in f:
                tensor = f.get_tensor(st_key)
                (mapped_key, transform) = map_to_jax_key(key_mapping, st_key)
                if mapped_key is None:
                    continue
                keys = mapped_key.split(".")
                try:
                    assign_weights(keys, tensor, state, st_key, transform)
                except KeyError:
                    logger.debug("Key %s not in state", mapped_key)
    except (OSError, ValueError, TypeError):
        logger.exception("Failed to load %s", filepath)


def _get_model_and_state(model_cls: Any, cfg: object) -> tuple[object, dict[str, Any]]:
    """Instantiate the model and extract its state.

    Args:
        model_cls: Model class constructor.
        cfg: Model configuration object.

    Returns:
        A tuple of (model_instance, state_dict).
    """
    nnx = __import__("flax", fromlist=["nnx"]).nnx
    model = model_cls(cfg, rngs=nnx.Rngs(0)) if model_cls else None
    state: dict[str, Any] = {}
    with contextlib.suppress(RuntimeError, ValueError, TypeError, AttributeError):
        (_, state, _) = nnx.split(model, ...)
    return model, state


def _populate_state_from_files(file_dir: str, state: dict[str, Any], key_mapping: dict[str, Any]) -> None:
    """Iterate directory files and populate state dictionary from safetensors.

    Args:
        file_dir: Directory containing safetensors weight files.
        state: State dictionary to populate.
        key_mapping: Name mapping from HF to JAX keys.
    """
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".safetensors"):  # pragma: no cover
                filepath = str(Path(root) / file)
                _load_weights_from_safetensors_file(filepath, state, key_mapping)


def create_model_from_safe_tensors(file_dir: str, model_cls: Any, cfg: object, key_mapping: dict[str, Any]) -> object:
    """Load tensors from the safetensors file and create a model (memory-optimized).

    This loads arrays one by one to avoid memory spikes, avoiding reading
    all parameters into host memory at once.

    Returns:
        object: The resulting output from the operation.

    """
    if safe_open is None or not Path(file_dir).is_dir():
        logger.warning("safetensors not available or file_dir invalid. Returning uninitialized model.")
        nnx = __import__("flax", fromlist=["nnx"]).nnx
        return model_cls(cfg, rngs=nnx.Rngs(0)) if model_cls else None

    model, state = _get_model_and_state(model_cls, cfg)
    _populate_state_from_files(file_dir, state, key_mapping)

    try:
        nnx = __import__("flax", fromlist=["nnx"]).nnx
        nnx.update(model, state)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        logger.exception("Failed to update model with loaded state")
    return model
