"""MaxText-specific PEFT / LoRA parameter transformation implementation."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, overload, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import JnpZerosAndArrayProtocol
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    DTypeProtocol,
    JaxArrayProtocol,
    MaxTextCausalLMProtocol,
    PEFTResultDict,
    PyTreeNode,
    TensorType,
    extract_status_kwargs,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _KeyPathEntryProtocol(Protocol):
    """Protocol for a JAX PyTree path entry exposing `.key`."""

    @property
    def key(self) -> str:
        """Provide the dictionary key name.

        Returns:
            Dictionary key string.

        """
        ...


@runtime_checkable
class _OptaxStateProtocol(Protocol):
    """Protocol satisfied by any Optax optimizer state tuple or dictionary."""

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            Formatted state representation.

        """
        ...


@runtime_checkable
class _OptaxTransformProtocol(Protocol):
    """Protocol for an Optax `GradientTransformation`."""

    @property
    def init(self) -> Callable[..., _OptaxStateProtocol]:
        """Provide the optimizer state initialization callable.

        Returns:
            Initializer callable mapping parameter PyTrees to optimizer states.

        """
        ...

    @property
    def update(self) -> Callable[..., tuple[_OptaxStateProtocol, _OptaxStateProtocol]]:
        """Provide the optimizer step update callable.

        Returns:
            Update callable mapping `(grads, state, params)` to `(updates, new_state)`.

        """
        ...


@runtime_checkable
class _MaxTextApplyProtocol[OutT](Protocol):
    """Protocol for a stateless MaxText model exposing `.apply(params, inputs)`."""

    def apply(
        self,
        params: Mapping[str, PyTreeNode],
        inputs: OutT,
        /,
    ) -> OutT:
        """Execute forward pass with explicit parameter PyTree.

        Args:
            params: Parameter PyTree dictionary.
            inputs: Input token ID or activation array.

        Returns:
            Model forward output array.

        """
        ...


type MaxTextParamLeaf = JaxArrayProtocol | TensorType | SupportsFloat | str
type MaxTextParamDict = dict[str, "MaxTextParamDict | MaxTextParamLeaf"]


@runtime_checkable
class _ParamItemsMappingProtocol(Protocol):
    """Non-generic protocol for iterating `(key, value)` pairs of MaxText parameter mappings."""

    def items(self) -> Iterable[tuple[str, BackendKwargValue | MaxTextParamDict | PyTreeNode | None]]:
        """Yield `(key, value)` pairs."""
        ...


def _coerce_maxtext_param_dict(
    raw: BackendKwargValue | MaxTextParamDict | PyTreeNode | None,
) -> MaxTextParamDict | None:
    """Recursively coerce a mapping into a typed `MaxTextParamDict`.

    Args:
        raw: Candidate parameter mapping or scalar.

    Returns:
        Coerced `MaxTextParamDict`, or `None` when `raw` is not a `Mapping`.

    """
    if not isinstance(raw, Mapping):
        return None
    assert isinstance(raw, _ParamItemsMappingProtocol)
    out: MaxTextParamDict = {}
    for k, v in raw.items():
        key_s = str(k)
        sub = _coerce_maxtext_param_dict(v)
        if sub is not None:
            out[key_s] = sub
        elif isinstance(v, (JaxArrayProtocol, SupportsFloat, str, int, float)):
            out[key_s] = v
    return out


@runtime_checkable
class _JaxPEFTRandomProtocol(Protocol):
    """Protocol for `jax.random` exposing `PRNGKey`, `split`, and `uniform`."""

    def PRNGKey(self, seed: int, /) -> JaxArrayProtocol:
        """Create a deterministic JAX PRNG key.

        Args:
            seed: Integer PRNG seed.

        Returns:
            PRNG key `JaxArrayProtocol`.

        """
        ...

    def split(
        self,
        key: JaxArrayProtocol | SupportsFloat,
        /,
    ) -> tuple[JaxArrayProtocol, JaxArrayProtocol]:
        """Split a PRNG key into two independent subkeys.

        Args:
            key: Parent PRNG key.

        Returns:
            2-tuple of `(next_key, subkey)`.

        """
        ...

    def uniform(
        self,
        key: JaxArrayProtocol,
        shape: tuple[int, int],
        *,
        dtype: DTypeProtocol | str | None = None,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> JaxArrayProtocol:
        """Sample uniform random float values in `[minval, maxval)`.

        Args:
            key: JAX PRNG subkey.
            shape: 2D matrix shape tuple.
            dtype: Target floating-point dtype.
            minval: Lower bound.
            maxval: Upper bound.

        Returns:
            Sampled `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class _JaxTreeUtilProtocol(Protocol):
    """Protocol for `jax.tree_util` exposing `tree_map_with_path`."""

    def tree_map_with_path[LeafT, OutLeafT](
        self,
        f: Callable[[Sequence[_KeyPathEntryProtocol], LeafT], OutLeafT],
        tree: Mapping[str, PyTreeNode],
        /,
    ) -> Mapping[str, PyTreeNode]:
        """Map `f(path, leaf)` across every leaf in `tree`.

        Args:
            f: Path-aware leaf mapping function.
            tree: Input parameter PyTree dictionary.

        Returns:
            Mapped PyTree of partition labels.

        """
        ...


@runtime_checkable
class _JaxPEFTModuleProtocol(Protocol):
    """Protocol for `jax` exposing `.random` and `.tree_util` in MaxText PEFT."""

    @property
    def random(self) -> _JaxPEFTRandomProtocol:
        """Provide `jax.random` namespace."""
        ...

    @property
    def tree_util(self) -> _JaxTreeUtilProtocol:
        """Provide `jax.tree_util` namespace."""
        ...


@runtime_checkable
class _JnpPEFTModuleProtocol(JnpZerosAndArrayProtocol, Protocol):
    """Protocol for `jax.numpy` exposing `float32`, `asarray`, `savez`, and `load` in MaxText PEFT."""

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `jnp.float32` dtype."""
        ...

    def asarray(self, a: MaxTextParamLeaf, /) -> JaxArrayProtocol:
        """Convert `a` into a `JaxArrayProtocol`.

        Args:
            a: Parameter leaf array or scalar.

        Returns:
            Converted `JaxArrayProtocol`.

        """
        ...

    def savez(self, file: Path | str, /, **kwds: JaxArrayProtocol | TensorType) -> None:
        """Save named adapter arrays to an uncompressed `.npz` archive.

        Args:
            file: Destination `.npz` path.
            **kwds: Flattened adapter arrays keyed by parameter path.

        """
        ...

    def load(self, file: Path | str, /) -> Mapping[str, JaxArrayProtocol]:
        """Load named adapter arrays from a `.npz` archive.

        Args:
            file: Source `.npz` path.

        Returns:
            Mapping from flattened parameter path to `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class _OptaxPEFTModuleProtocol(Protocol):
    """Protocol for `optax` exposing `adam`, `set_to_zero`, and `multi_transform`."""

    def adam(self, learning_rate: float, /) -> _OptaxTransformProtocol:
        """Construct an `adam` optimizer transformation.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_OptaxTransformProtocol`.

        """
        ...

    def set_to_zero(self) -> _OptaxTransformProtocol:
        """Construct a zero-gradient transformation for frozen parameters.

        Returns:
            Frozen `_OptaxTransformProtocol`.

        """
        ...

    def multi_transform(
        self,
        transforms: Mapping[str, _OptaxTransformProtocol | str],
        param_labels: Mapping[str, PyTreeNode],
        /,
    ) -> _OptaxTransformProtocol:
        """Partition parameter updates by `param_labels` across `transforms`.

        Args:
            transforms: Mapping from partition name to `_OptaxTransformProtocol`.
            param_labels: Parameter PyTree of partition label strings.

        Returns:
            Composite `_OptaxTransformProtocol`.

        """
        ...


jax: _JaxPEFTModuleProtocol | None = optional_import("jax", _JaxPEFTModuleProtocol)
jnp: _JnpPEFTModuleProtocol | None = optional_import("jax.numpy", _JnpPEFTModuleProtocol)
optax: _OptaxPEFTModuleProtocol | None = optional_import("optax", _OptaxPEFTModuleProtocol)
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)


@overload
def transform_params_to_lora(
    params: MaxTextParamDict,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rng: JaxArrayProtocol | SupportsFloat | None = None,
) -> tuple[MaxTextParamDict, int]:
    """Transform a `MaxTextParamDict` by injecting low-rank adapter matrices.

    Args:
        params: Nested dictionary representing model parameters.
        target_modules: List of module name substrings/suffixes to adapt.
        lora_r: LoRA rank dimension (`lora_r > 0`).
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability for LoRA layers.
        rng: Optional JAX `PRNGKey` for initializing adapter matrices.

    Returns:
        A tuple of `(transformed_params_dict, count_of_injected_modules)`.

    """


@overload
def transform_params_to_lora(
    params: MaxTextParamDict | MaxTextParamLeaf,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rng: JaxArrayProtocol | SupportsFloat | None = None,
) -> tuple[MaxTextParamDict | MaxTextParamLeaf, int]:
    """Transform a parameter PyTree or leaf by injecting low-rank adapter matrices.

    Args:
        params: Nested dictionary or leaf representing model parameters.
        target_modules: List of module name substrings/suffixes to adapt.
        lora_r: LoRA rank dimension (`lora_r > 0`).
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability for LoRA layers.
        rng: Optional JAX `PRNGKey` for initializing adapter matrices.

    Returns:
        A tuple of `(transformed_params, count_of_injected_modules)`.

    """


def transform_params_to_lora(
    params: MaxTextParamDict | MaxTextParamLeaf,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rng: JaxArrayProtocol | SupportsFloat | None = None,
) -> tuple[MaxTextParamDict | MaxTextParamLeaf, int]:
    """Transform a MaxText parameter PyTree by injecting low-rank adapter matrices.

    For any projection module matching `target_modules` (e.g. `q_proj`, `v_proj`),
    decomposes parameter updates by augmenting the layer with `lora_a` (shape `d_in x r`)
    and `lora_b` (shape `r x d_out`) arrays.

    Args:
        params: Nested dictionary representing model parameters.
        target_modules: List of module name substrings/suffixes to adapt.
        lora_r: LoRA rank dimension (`lora_r > 0`).
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability for LoRA layers.
        rng: Optional JAX `PRNGKey` for initializing adapter matrices.

    Returns:
        A tuple of `(transformed_params_dict, count_of_injected_modules)`.

    Raises:
        DependencyMissingError: If JAX is missing.
        ValueError: If rank `lora_r` is less than or equal to 0.

    """
    _ = lora_dropout
    if jax is None or jnp is None:
        raise DependencyMissingError("JAX dependencies are missing.")
    if lora_r <= 0:
        raise ValueError(f"LoRA rank r must be positive, got {lora_r}")

    active_rng = rng if rng is not None else jax.random.PRNGKey(0)
    injected_count = 0

    def _inject(
        curr: MaxTextParamDict | MaxTextParamLeaf, current_path: str = ""
    ) -> MaxTextParamDict | MaxTextParamLeaf:
        """Inject LoRA factors into matching parameter dictionaries.

        Args:
            curr: Current parameter dictionary or leaf.
            current_path: Dotted parameter tree path.

        Returns:
            Updated parameter dictionary or leaf.

        """
        nonlocal injected_count, active_rng
        assert jax is not None
        assert jnp is not None
        if not isinstance(curr, dict):
            return curr
        res: MaxTextParamDict = {}
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                v_dict: MaxTextParamDict = v
                if "kernel" in v_dict and any(
                    k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in target_modules
                ):
                    kernel = v_dict["kernel"]
                    assert isinstance(kernel, JaxArrayProtocol)
                    d_in, d_out = kernel.shape
                    scale_init = 1.0 / math.sqrt(lora_r)
                    active_rng, sub_rng = jax.random.split(active_rng)
                    lora_a: JaxArrayProtocol = jax.random.uniform(
                        sub_rng,
                        (d_in, lora_r),
                        dtype=kernel.dtype,
                        minval=-scale_init,
                        maxval=scale_init,
                    )
                    lora_b: JaxArrayProtocol = jnp.zeros((lora_r, d_out), dtype=kernel.dtype)
                    new_v: MaxTextParamDict = dict(v_dict)
                    new_v["lora_a"] = lora_a
                    new_v["lora_b"] = lora_b
                    new_v["lora_scale"] = jnp.array(float(lora_alpha) / float(lora_r), dtype=jnp.float32)
                    res[k] = new_v
                    injected_count += 1
                else:
                    res[k] = _inject(v_dict, sub_path)
            else:
                res[k] = v
        return res

    transformed = _inject(params)
    return transformed, injected_count


def segregate_adapter_params(params: MaxTextParamDict | MaxTextParamLeaf) -> tuple[MaxTextParamDict, MaxTextParamDict]:
    """Segregate a parameter dictionary into trainable LoRA adapters and frozen base weights.

    Only `lora_a` and `lora_b` matrices are partitioned into `trainable`; base kernels,
    biases, and the fixed `lora_scale` hyperparameter constant (`alpha / r`) remain frozen,
    matching `create_maxtext_lora_optimizer` and `count_maxtext_parameters`.

    Args:
        params: Model parameter PyTree.

    Returns:
        A tuple of `(trainable_lora_params, frozen_base_params)`.

    """
    trainable: MaxTextParamDict = {}
    frozen: MaxTextParamDict = {}

    def _split(
        src: MaxTextParamDict | MaxTextParamLeaf,
        dst_trainable: MaxTextParamDict,
        dst_frozen: MaxTextParamDict,
    ) -> None:
        """Split parameters into trainable and frozen PyTrees recursively.

        Args:
            src: Source parameter tree node.
            dst_trainable: Destination trainable dictionary.
            dst_frozen: Destination frozen dictionary.

        """
        if not isinstance(src, dict):
            return
        for k, v in src.items():
            if isinstance(v, dict):
                v_dict: MaxTextParamDict = v
                sub_t: MaxTextParamDict = {}
                sub_f: MaxTextParamDict = {}
                _split(v_dict, sub_t, sub_f)
                if sub_t:
                    dst_trainable[k] = sub_t
                if sub_f:
                    dst_frozen[k] = sub_f
            elif k in {"lora_a", "lora_b"}:
                dst_trainable[k] = v
            else:
                dst_frozen[k] = v

    _split(params, trainable, frozen)
    return trainable, frozen


def create_maxtext_lora_optimizer(
    params: MaxTextParamDict,
    base_optimizer: _OptaxTransformProtocol | str | None = None,
    learning_rate: float = 1e-4,
) -> _OptaxTransformProtocol:
    """Create an Optax `multi_transform` optimizer updating only LoRA parameters.

    Freezes all base model weights and `lora_scale` constants by routing them to
    `optax.set_to_zero()` while directing `lora_a` and `lora_b` adapters to `base_optimizer`.

    Args:
        params: Model parameter PyTree containing LoRA adapters.
        base_optimizer: Optax transformation for trainable weights (defaults to `optax.adam(learning_rate)`).
        learning_rate: Learning rate used when `base_optimizer` is `None`.

    Returns:
        An Optax `GradientTransformation` configured for adapter-only training.

    Raises:
        DependencyMissingError: If Optax or JAX dependencies are missing.

    """
    if optax is None or jax is None:
        raise DependencyMissingError("Optax or JAX dependencies are missing.")

    resolved_opt = base_optimizer if base_optimizer is not None else optax.adam(learning_rate)

    def _label_fn(path: Sequence[_KeyPathEntryProtocol | str], _val: MaxTextParamLeaf) -> str:
        """Classify parameter path as trainable adapter or frozen base.

        Args:
            path: PyTree key path sequence.
            _val: Leaf parameter value.

        Returns:
            `'trainable'` for `lora_a` / `lora_b`, otherwise `'frozen'`.

        """
        last_entry = path[-1]
        key = last_entry.key if isinstance(last_entry, _KeyPathEntryProtocol) else str(last_entry)
        if key in {"lora_a", "lora_b"}:
            return "trainable"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, params)
    res: _OptaxTransformProtocol = optax.multi_transform(
        {"trainable": resolved_opt, "frozen": optax.set_to_zero()},
        param_labels,
    )
    return res


def materialize_effective_params(params: MaxTextParamDict | MaxTextParamLeaf) -> MaxTextParamDict | MaxTextParamLeaf:
    """Differentiably materialize effective `kernel` weights from AQT scales and LoRA adapters.

    For each projection dictionary:
    1. If AQT `kernel_scale` is present alongside a quantized `kernel`, dequantizes
       `kernel.astype(float32) * kernel_scale`.
    2. If `lora_a` and `lora_b` are present, computes `kernel + lora_scale * (lora_a @ lora_b)`
       with full JAX autodiff support so gradients flow cleanly into `lora_a` and `lora_b`.

    Args:
        params: Parameter PyTree optionally containing LoRA adapters and/or AQT scales.

    Returns:
        Parameter PyTree with effective floating-point `kernel` tensors and auxiliary keys stripped.

    """
    if not isinstance(params, dict):
        return params
    res: MaxTextParamDict = {}
    for k, v in params.items():
        if isinstance(v, dict):
            v_dict: MaxTextParamDict = v
            raw_kernel = v_dict.get("kernel")
            if isinstance(raw_kernel, JaxArrayProtocol) and ("lora_a" in v_dict or "kernel_scale" in v_dict):
                eff_kernel: JaxArrayProtocol = raw_kernel
                raw_q_scale = v_dict.get("kernel_scale")
                if isinstance(raw_q_scale, JaxArrayProtocol) and jnp is not None:
                    eff_kernel = eff_kernel.astype(jnp.float32) * raw_q_scale
                lora_a = v_dict.get("lora_a")
                lora_b = v_dict.get("lora_b")
                if isinstance(lora_a, JaxArrayProtocol) and isinstance(lora_b, JaxArrayProtocol):
                    raw_scale = v_dict.get("lora_scale", 1.0)
                    delta: JaxArrayProtocol = lora_a @ lora_b
                    if isinstance(raw_scale, JaxArrayProtocol):
                        eff_kernel += delta * raw_scale
                    else:
                        scale_f = float(raw_scale) if isinstance(raw_scale, (int, float, SupportsFloat)) else 1.0
                        eff_kernel += delta * scale_f
                new_module: MaxTextParamDict = {
                    mk: mv
                    for mk, mv in v_dict.items()
                    if mk not in {"lora_a", "lora_b", "lora_scale", "kernel_scale", "aqt_config"}
                }
                new_module["kernel"] = eff_kernel
                res[k] = new_module
            else:
                res[k] = materialize_effective_params(v_dict)
        else:
            res[k] = v
    return res


def merge_lora_weights(params: MaxTextParamDict | MaxTextParamLeaf) -> MaxTextParamDict | MaxTextParamLeaf:
    """Fold LoRA adapter weights back into base MaxText weights.

    Computes `W_merged = W + scale * (lora_a @ lora_b)` and removes adapter parameters,
    returning standard model weights.

    Args:
        params: Model parameter PyTree containing LoRA adapters.

    Returns:
        A new parameter dictionary with folded base weights and adapters removed.

    """
    return materialize_effective_params(params)


def apply_with_lora[OutT](
    model: _MaxTextApplyProtocol[OutT],
    lora_params: Mapping[str, PyTreeNode],
    inputs: OutT,
) -> OutT:
    """Execute a stateless MaxText model forward pass with LoRA adapters active.

    Materializes `W + lora_scale * (lora_a @ lora_b)` differentiably so `lora_a` and `lora_b`
    participate in both forward inference and `jax.grad` / `jax.value_and_grad` optimization.

    Args:
        model: Stateless MaxText model exposing `.apply(params, inputs)`.
        lora_params: Parameter PyTree containing base weights and `lora_a` / `lora_b` adapters.
        inputs: Input token ID or activation array.

    Returns:
        Forward output logits or activations produced by `model.apply(...)`.

    """
    coerced = _coerce_maxtext_param_dict(lora_params)
    effective_tree: Mapping[str, PyTreeNode]
    if coerced is not None:
        materialized = materialize_effective_params(coerced)
        assert isinstance(materialized, dict)
        effective_tree = materialized
    else:
        effective_tree = lora_params
    return model.apply(effective_tree, inputs)


def save_maxtext_adapters(params: MaxTextParamDict | MaxTextParamLeaf, save_path: str | Path) -> None:
    """Save LoRA adapter weights from a parameter PyTree to an NPZ archive.

    Args:
        params: Model parameter PyTree containing LoRA adapters.
        save_path: File path or directory path destination.

    Raises:
        DependencyMissingError: If JAX is missing.

    """
    if jnp is None:
        raise DependencyMissingError("JAX dependency is missing.")

    path = Path(save_path)
    if path.is_dir() or path.suffix != ".npz":
        path /= "maxtext_lora_adapters.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    adapters: dict[str, JaxArrayProtocol | TensorType] = {}

    def _collect(curr: MaxTextParamDict | MaxTextParamLeaf, current_path: str = "") -> None:
        """Collect adapter arrays from parameter tree into flat dict.

        Args:
            curr: Current parameter dictionary or leaf.
            current_path: Dotted key path prefix.

        """
        assert jnp is not None
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                _collect(v, sub_path)
            elif k in {"lora_a", "lora_b", "lora_scale"}:
                adapters[sub_path] = jnp.asarray(v)

    _collect(params)
    jnp.savez(path, **adapters)


def load_maxtext_adapters(params: MaxTextParamDict, load_path: str | Path) -> MaxTextParamDict:
    """Load LoRA adapter weights from an NPZ file into a base parameter PyTree.

    Args:
        params: Model parameter PyTree.
        load_path: Path to the NPZ archive containing adapter weights.

    Returns:
        Updated parameter dictionary with loaded LoRA adapters.

    Raises:
        DependencyMissingError: If JAX is missing.
        FileNotFoundError: If the load path does not exist.

    """
    if jnp is None:
        raise DependencyMissingError("JAX dependency is missing.")

    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found at {load_path}")

    loaded = dict(jnp.load(path))
    res = copy.deepcopy(params)

    for flat_key, array_val in loaded.items():
        parts = flat_key.split(".")
        curr: MaxTextParamDict = res
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            next_curr = curr[part]
            assert isinstance(next_curr, dict)
            curr = next_curr
        curr[parts[-1]] = jnp.asarray(array_val)

    return res


def count_maxtext_parameters(params: MaxTextParamDict | MaxTextParamLeaf) -> tuple[int, int]:
    """Count total and trainable LoRA parameters in a MaxText parameter PyTree.

    Args:
        params: Model parameter PyTree.

    Returns:
        A tuple of `(total_parameter_count, trainable_lora_parameter_count)`.

    """
    trainable_count = 0
    total_count = 0

    def _count(curr: MaxTextParamDict | MaxTextParamLeaf) -> None:
        """Count parameters recursively in parameter dictionary.

        Args:
            curr: Current parameter dictionary or leaf.

        """
        nonlocal trainable_count, total_count
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            if isinstance(v, dict):
                _count(v)
            elif isinstance(v, JaxArrayProtocol):
                size = math.prod(int(dim) for dim in v.shape)
                total_count += size
                if k in {"lora_a", "lora_b"}:
                    trainable_count += size

    _count(params)
    return total_count, trainable_count


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: BackendKwargValue | MaxTextParamDict | Mapping[str, PyTreeNode],
) -> PEFTResultDict:
    """Apply LoRA to a model using the MaxText backend and cache adapted parameters.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including `'params'`, `'output_dir'`, and `'merge'`.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    if jax is None or jnp is None or (Gemma4Model is None and "params" not in kwargs):
        raise DependencyMissingError("MaxText dependencies are missing.")

    def _inject() -> int:
        """Initialize or transform MaxText parameter PyTree with LoRA matrices.

        Returns:
            Number of projection modules into which LoRA adapters were injected.

        """
        from gemma_4_sql.backends.maxtext.inference import cache_maxtext_params

        assert jax is not None
        assert jnp is not None
        raw_params = kwargs.get("params")
        coerced_params = _coerce_maxtext_param_dict(raw_params)
        loaded_model: MaxTextCausalLMProtocol | None = None
        if coerced_params is not None:
            params: MaxTextParamDict = coerced_params
        else:
            assert Gemma4Model is not None
            cand_model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            raw_init = _coerce_maxtext_param_dict(dict(cand_model.init(rng, dummy_input)))
            assert raw_init is not None
            params = raw_init
            loaded_model = cand_model

        params, injected_count = transform_params_to_lora(
            params=params,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        raw_out_dir = kwargs.get("output_dir")
        if raw_out_dir is not None:
            save_maxtext_adapters(params, str(raw_out_dir))

        final_params = merge_lora_weights(params) if kwargs.get("merge") else params
        assert isinstance(final_params, dict)
        cache_maxtext_params(model_name, final_params, model=loaded_model)

        logger.info("MaxText LoRA applied to %d modules", injected_count)
        return injected_count

    wrapper_kwargs = extract_status_kwargs(kwargs)
    return apply_lora_wrapper(
        "maxtext",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **wrapper_kwargs,
    )


__all__ = [
    "Gemma4Model",
    "MaxTextParamDict",
    "MaxTextParamLeaf",
    "PyTreeNode",
    "apply_lora",
    "apply_with_lora",
    "count_maxtext_parameters",
    "create_maxtext_lora_optimizer",
    "load_maxtext_adapters",
    "materialize_effective_params",
    "merge_lora_weights",
    "save_maxtext_adapters",
    "segregate_adapter_params",
    "transform_params_to_lora",
]
