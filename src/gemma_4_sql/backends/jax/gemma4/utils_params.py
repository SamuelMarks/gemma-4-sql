"""Safetensors weight loading, key mapping, and shape transformation utilities for JAX Gemma 4."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import jax
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.backends.lazy_loader import optional_attr
from gemma_4_sql.type_hints import DTypeProtocol, DTypeSpec, ModelConfigProtocol, NDArrayProtocol, TensorProtocol, ToListArrayProtocol

type TransformSpec = tuple[tuple[int, ...] | None, tuple[int, ...] | None, bool] | None


class TransformKind(Enum):
    """Enumeration of standard parameter tensor transformations."""

    DEFAULT = "default"
    BIAS = "bias"
    EMBED = "embed"
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    LINEAR_3D = ((0, 2, 1), None, False)


type TransformType = TransformSpec | TransformKind
type KeyMapType = tuple[str, TransformType]


@runtime_checkable
class TransformableArrayProtocol(Protocol):
    """Protocol for arrays supporting shape, dtype, reshape, transpose, and astype."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the array shape."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec | jax.numpy.dtype:
        """Provide the array dtype."""
        ...

    def reshape(self, *shape: int | tuple[int, ...]) -> Self:
        """Reshape the array."""
        ...

    def transpose(self, *axes: int | tuple[int, ...]) -> Self:
        """Permute array dimensions."""
        ...

    def astype(self, dtype: DTypeProtocol | DTypeSpec | jax.numpy.dtype | None) -> Self:
        """Cast array to `dtype`."""
        ...


type ArrayLikeTensor = jax.Array | TensorProtocol | NDArrayProtocol | TransformableArrayProtocol | ToListArrayProtocol


@runtime_checkable
class ShardingWithSpecProtocol(Protocol):
    """Protocol for a JAX sharding object exposing a `.spec` attribute."""

    @property
    def spec(self) -> jax.sharding.PartitionSpec | None:
        """Provide the underlying `PartitionSpec`."""
        ...


@runtime_checkable
class ShardedTargetProtocol(Protocol):
    """Protocol for an array or shape descriptor carrying a `.sharding` attribute."""

    @property
    def sharding(self) -> ShardingWithSpecProtocol | jax.sharding.Sharding | None:
        """Provide the sharding descriptor."""
        ...


@runtime_checkable
class ShapeDtypeLeafProtocol(Protocol):
    """Protocol for a state leaf exposing `.shape` and `.dtype`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the tensor shape."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec | jax.numpy.dtype:
        """Provide the tensor dtype."""
        ...


@runtime_checkable
class ParamValueHolderProtocol(Protocol):
    """Protocol for an NNX Variable or holder exposing a mutable `.value` attribute."""

    value: ShapeDtypeLeafProtocol | ArrayLikeTensor


@runtime_checkable
class SafeOpenFileProtocol(Protocol):
    """Protocol for an opened safetensors reader."""

    def keys(self) -> Iterable[str]:
        """Return an iterable of tensor names in the archive."""
        ...

    def get_tensor(self, key: str, /) -> ArrayLikeTensor:
        """Load the tensor for `key`."""
        ...


@runtime_checkable
class SafeOpenContextProtocol(Protocol):
    """Protocol for the `safetensors.safe_open` context manager."""

    def __enter__(self) -> SafeOpenFileProtocol:
        """Enter the safetensors context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the safetensors context."""
        ...


@runtime_checkable
class SafeOpenCallableProtocol(Protocol):
    """Protocol for the `safetensors.safe_open` callable."""

    def __call__(self, filepath: str, *, framework: str = "jax") -> SafeOpenContextProtocol:
        """Open a safetensors file and return a context manager."""
        ...


type StateKey = str | int
type StateLeafValue = "StateNode | ParamValueHolderProtocol | ShapeDtypeLeafProtocol | ArrayLikeTensor | float | int | str | None"
type StateNode = MutableMapping[StateKey, StateLeafValue]
type ShardingNode = Mapping[StateKey, "ShardingNode | jax.sharding.Sharding | jax.sharding.PartitionSpec | None"]

logger = logging.getLogger(__name__)

safe_open: SafeOpenCallableProtocol | None = optional_attr("safetensors", "safe_open", SafeOpenCallableProtocol)


def map_to_jax_key(mapping: Mapping[str, KeyMapType], source_key: str) -> KeyMapType | tuple[None, None]:
    """Map a safetensors key to a single JAX parameter path and tensor transform.

    Args:
        mapping: Dictionary mapping regex patterns to `(replacement_key, transform)` tuples.
        source_key: Original safetensors parameter name.

    Returns:
        Tuple of `(mapped_key, transform)` or `(None, None)` if unmapped.

    Raises:
        ValueError: If multiple regex patterns match `source_key`.

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
    """Convert a string to an integer if numeric, otherwise return the original string.

    Args:
        s: Input string token.

    Returns:
        Parsed integer or original string.

    """
    try:
        return int(s)
    except ValueError:
        return s


def _resolve_transform_spec(transform: TransformType) -> TransformSpec:
    """Resolve a `TransformType` (enum or tuple) to a concrete 3-tuple or `None`.

    Args:
        transform: `TransformKind` enum member, 3-tuple specification, or `None`.

    Returns:
        Normalized `(permute, reshape, reshape_first)` 3-tuple or `None`.

    """
    if transform is None:
        return None
    if isinstance(transform, Enum):
        val = transform.value
        return val if isinstance(val, tuple) else None
    return transform


def _apply_transform(tensor: ArrayLikeTensor, transform: TransformType) -> ArrayLikeTensor:
    """Apply permutation and reshape transformations to a weight tensor.

    Args:
        tensor: Input weight tensor.
        transform: Optional 3-tuple of `(permute_axes, target_shape, reshape_first)` or `TransformKind`.

    Returns:
        Transformed tensor array.

    """
    spec = _resolve_transform_spec(transform)
    if spec is None:
        return tensor
    (permute, reshape, reshape_first) = spec
    res: ArrayLikeTensor = tensor
    if reshape_first and reshape is not None:
        res = res.reshape(reshape)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute, invalid-argument-type]
    if permute:
        res = res.transpose(permute)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue,reportArgumentType]  # ty: ignore[missing-argument, unresolved-attribute, invalid-argument-type]
    if not reshape_first and reshape is not None:
        res = res.reshape(reshape)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute, invalid-argument-type]
    return res


def _resolve_state_key(state_dict: MutableMapping[StateKey, StateLeafValue], key: StateKey) -> StateKey:
    """Resolve `key` against `state_dict`, coercing between `int` and numeric `str` when needed."""
    if key not in state_dict:
        if isinstance(key, str) and key.isdigit() and int(key) in state_dict:
            return int(key)
        if isinstance(key, int) and str(key) in state_dict:
            return str(key)
    return key


def _extract_leaf_shape_and_dtype(
    leaf: StateLeafValue,
) -> tuple[tuple[int, ...], DTypeProtocol | DTypeSpec | jax.numpy.dtype | None]:
    """Return `(shape, dtype)` for a parameter holder or raw tensor leaf."""
    inner = leaf.value if isinstance(leaf, ParamValueHolderProtocol) else leaf
    return (inner.shape, inner.dtype) if isinstance(inner, ShapeDtypeLeafProtocol) else ((), None)


def assign_weights(
    keys: Sequence[StateKey],
    tensor: ArrayLikeTensor,
    state_dict: StateNode,
    st_key: str,
    transform: TransformType,
    *,
    sharding_dict: ShardingNode | None = None,
) -> None:
    """Recursively descend into `state_dict` and assign the transformed weight tensor.

    Args:
        keys: Hierarchical path segments within `state_dict`.
        tensor: Source tensor to assign.
        state_dict: Nested Flax NNX state dictionary.
        st_key: Original safetensors key name for diagnostics.
        transform: Optional shape/permutation transform specification.
        sharding_dict: Optional sharding specification tree for device placement.

    Raises:
        ValueError: If the transformed tensor shape does not match the target parameter shape.

    """
    (key, *rest) = keys
    resolved_key = _resolve_state_key(state_dict, key)
    if not rest:
        transformed = _apply_transform(tensor, transform)
        leaf = state_dict[resolved_key]
        target_shape, _ = _extract_leaf_shape_and_dtype(leaf)
        if transformed.shape != target_shape:
            msg = f"Shape mismatch for {st_key}: {transformed.shape} vs {target_shape}"
            raise ValueError(msg)
        shd_leaf = sharding_dict[resolved_key] if isinstance(sharding_dict, Mapping) and resolved_key in sharding_dict else None
        if isinstance(shd_leaf, (jax.sharding.Sharding, jax.sharding.PartitionSpec)):
            val: ArrayLikeTensor = jax.device_put(transformed, shd_leaf)
        else:
            val = jax.device_put(transformed)
        if isinstance(leaf, ParamValueHolderProtocol):
            leaf.value = val
        else:
            state_dict[resolved_key] = val
    else:
        next_sharding = sharding_dict[resolved_key] if isinstance(sharding_dict, Mapping) and resolved_key in sharding_dict else None
        child_node = state_dict[resolved_key]
        assert isinstance(child_node, MutableMapping)
        assign_weights(
            rest,
            tensor,
            child_node,
            st_key,
            transform,
            sharding_dict=next_sharding if isinstance(next_sharding, Mapping) else None,
        )


def assign_weights_from_eval_shape(
    keys: Sequence[StateKey],
    tensor: ArrayLikeTensor,
    state_dict: StateNode,
    st_key: str,
    transform: TransformType,
) -> None:
    """Recursively descend into an abstract `eval_shape` state dict and assign the tensor.

    Args:
        keys: Hierarchical path segments within `state_dict`.
        tensor: Source tensor to assign.
        state_dict: Abstract state dictionary produced via `nnx.eval_shape`.
        st_key: Original safetensors key name for diagnostics.
        transform: Optional shape/permutation transform specification.

    Raises:
        ValueError: If the transformed tensor shape does not match the target parameter shape.

    """
    (key, *rest) = keys
    resolved_key = _resolve_state_key(state_dict, key)
    if not rest:
        transformed = _apply_transform(tensor, transform)
        val_obj = state_dict[resolved_key]
        expected_shape, expected_dtype = _extract_leaf_shape_and_dtype(val_obj)
        if transformed.shape != expected_shape:
            msg = f"Shape mismatch for {st_key}: {transformed.shape} vs {expected_shape}"
            raise ValueError(msg)
        resolved_dtype: DTypeSpec | jax.numpy.dtype = str(expected_dtype) if isinstance(expected_dtype, DTypeProtocol) and not isinstance(expected_dtype, jax.numpy.dtype) else expected_dtype
        casted: ArrayLikeTensor = transformed.astype(resolved_dtype)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        target = val_obj.value if isinstance(val_obj, ParamValueHolderProtocol) else val_obj
        if isinstance(target, ShardedTargetProtocol) and bool(target.sharding):
            shd = target.sharding
            shd_spec = shd.spec if isinstance(shd, ShardingWithSpecProtocol) else shd
            casted = jax.device_put(casted, shd_spec)
        if isinstance(val_obj, ParamValueHolderProtocol):
            val_obj.value = casted
        else:
            state_dict[resolved_key] = casted
    else:
        child_node = state_dict[resolved_key]
        assert isinstance(child_node, MutableMapping)
        assign_weights_from_eval_shape(rest, tensor, child_node, st_key, transform)


def _iter_safe_open_keys(f: SafeOpenFileProtocol | Iterable[str]) -> Iterator[str]:
    """Yield tensor keys from an opened safetensors reader."""
    yield from (f if isinstance(f, Iterable) else f.keys())


def _load_weights_from_safetensors_file(
    filepath: str,
    state: StateNode,
    key_mapping: Mapping[str, KeyMapType],
) -> None:
    """Load and assign weights from a single `.safetensors` file into `state`.

    Args:
        filepath: Path to the `.safetensors` file.
        state: Mutable model state dictionary.
        key_mapping: Mapping from Hugging Face parameter patterns to JAX paths.

    """
    try:
        assert safe_open is not None
        with safe_open(filepath, framework="jax") as f:
            for st_key in _iter_safe_open_keys(f):
                tensor = f.get_tensor(st_key)
                (mapped_key, transform) = map_to_jax_key(key_mapping, st_key)
                if mapped_key is None:
                    continue
                keys = mapped_key.split(".")
                try:
                    assign_weights(keys, tensor, state, st_key, transform)
                except KeyError:
                    logger.debug("Key %s not in state", mapped_key)
    except (OSError, ValueError, TypeError, AssertionError):
        logger.exception("Failed to load %s", filepath)


@runtime_checkable
class PureDictStateProtocol(Protocol):
    """Protocol for an NNX State supporting `to_pure_dict()`."""

    def to_pure_dict(self) -> StateNode:
        """Convert NNX state to a pure nested dictionary."""
        ...


@runtime_checkable
class FlatDictStateProtocol(Protocol):
    """Protocol for an NNX State supporting `to_flat_dict()`."""

    def to_flat_dict(self) -> StateNode:
        """Convert NNX state to a flat dictionary."""
        ...


def state_to_mutable_dict(abs_state: PureDictStateProtocol | FlatDictStateProtocol | Mapping[StateKey, StateLeafValue] | Iterable[tuple[StateKey, StateLeafValue]]) -> StateNode:
    """Convert an NNX `State` or mapping into a mutable `StateNode` dictionary.

    Args:
        abs_state: Abstract or concrete NNX state returned by `nnx.split`.

    Returns:
        Mutable dictionary representation of the state.

    """
    if isinstance(abs_state, PureDictStateProtocol):
        return dict(abs_state.to_pure_dict())
    if isinstance(abs_state, FlatDictStateProtocol):
        return dict(abs_state.to_flat_dict())
    return dict(abs_state)  # pyright: ignore[reportArgumentType]


def _get_model_and_state[ModelT](
    model_cls: Callable[..., ModelT],
    cfg: ModelConfig | ModelConfigProtocol | Mapping[str, int | str] | str | None,
) -> tuple[ModelT, StateNode]:
    """Instantiate `model_cls(cfg, rngs=nnx.Rngs(0))` and extract its mutable state dictionary.

    Args:
        model_cls: Model class to instantiate.
        cfg: Model configuration instance.

    Returns:
        Tuple of `(model, state_dict)`.

    """
    model = model_cls(cfg, rngs=nnx.Rngs(0))
    state: StateNode = {}
    with contextlib.suppress(RuntimeError, ValueError, TypeError, AttributeError):
        split_res = nnx.split(model)  # pyright: ignore[reportArgumentType]
        raw_state = split_res[1]
        state = state_to_mutable_dict(raw_state) if not isinstance(raw_state, dict) else raw_state
    return model, state


def _populate_state_from_files(
    file_dir: str,
    state: StateNode,
    key_mapping: Mapping[str, KeyMapType],
) -> None:
    """Walk `file_dir` and populate `state` from all discovered `.safetensors` files.

    Args:
        file_dir: Directory containing `.safetensors` weight shards.
        state: State dictionary to populate in-place.
        key_mapping: Regex mapping from Hugging Face keys to JAX state keys.

    """
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".safetensors"):
                filepath = str(Path(root) / file)
                _load_weights_from_safetensors_file(filepath, state, key_mapping)


def create_model_from_safe_tensors[ModelT](
    file_dir: str,
    model_cls: Callable[..., ModelT],
    cfg: ModelConfig | ModelConfigProtocol | Mapping[str, int | str] | str | None,
    key_mapping: Mapping[str, KeyMapType],
    *,
    strict: bool = False,
) -> ModelT | None:
    """Stream tensors from `.safetensors` shards in `file_dir` into a newly initialized model.

    Args:
        file_dir: Directory path containing `.safetensors` files.
        model_cls: Model class to instantiate.
        cfg: Model configuration instance.
        key_mapping: Regex mapping from source keys to target JAX state keys.
        strict: If True, raise `ExportError` when `nnx.update(model, state)` fails.

    Returns:
        Instantiated model updated with loaded weights.

    Raises:
        ExportError: If `strict=True` and updating the model state fails.

    """
    if safe_open is None or not Path(file_dir).is_dir():
        logger.warning("safetensors not available or file_dir invalid. Returning uninitialized model.")
        return model_cls(cfg, rngs=nnx.Rngs(0)) if model_cls is not None else None

    model, state = _get_model_and_state(model_cls, cfg)
    _populate_state_from_files(file_dir, state, key_mapping)

    try:
        nnx.update(model, state)  # pyright: ignore[reportArgumentType]
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        logger.exception("Failed to update model with loaded state")
        if strict:
            from gemma_4_sql.exceptions import ExportError

            raise ExportError(f"Failed to update JAX model state from {file_dir}: {e!s}", backend="jax", cause=e) from e
    return model
