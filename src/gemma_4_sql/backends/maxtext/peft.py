"""MaxText-specific PEFT / LoRA parameter transformation implementation."""

from __future__ import annotations

import contextlib
import copy
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    JaxArrayProtocol,
    MaxTextCausalLMProtocol,
    PEFTResultDict,
    PyTreeNode,
    TensorType,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _KeyPathEntryProtocol(Protocol):
    """Protocol for a JAX PyTree path entry exposing `.key`."""

    @property
    def key(self) -> str:
        """Provide the dictionary key name."""
        ...


@runtime_checkable
class _OptaxStateProtocol(Protocol):
    """Protocol satisfied by any Optax optimizer state tuple or dictionary."""

    def __repr__(self) -> str:
        """Return string representation."""
        ...


@runtime_checkable
class _OptaxTransformProtocol(Protocol):
    """Protocol for an Optax `GradientTransformation`."""

    @property
    def init(self) -> Callable[..., _OptaxStateProtocol]:
        """Provide the optimizer state initialization callable."""
        ...

    @property
    def update(self) -> Callable[..., tuple[_OptaxStateProtocol, _OptaxStateProtocol]]:
        """Provide the optimizer step update callable."""
        ...


type MaxTextParamLeaf = JaxArrayProtocol | TensorType | SupportsFloat | str
type MaxTextParamDict = dict[str, "MaxTextParamDict | MaxTextParamLeaf"]


jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
_raw_gemma4 = optional_attr("maxtext.models.gemma4", "Gemma4Model")
Gemma4Model: Callable[..., MaxTextCausalLMProtocol] | None = _raw_gemma4 if callable(_raw_gemma4) else None  # ty: ignore[invalid-assignment]


def transform_params_to_lora(
    params: MaxTextParamDict | MaxTextParamLeaf,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rng: JaxArrayProtocol | SupportsFloat | None = None,
) -> tuple[MaxTextParamDict, int]:
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

    def _inject(curr: MaxTextParamDict | MaxTextParamLeaf, current_path: str = "") -> MaxTextParamDict | MaxTextParamLeaf:
        """Inject LoRA factors into matching parameter dictionaries."""
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
                if "kernel" in v_dict and any(k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in target_modules):
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
    return transformed, injected_count  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]


def segregate_adapter_params(params: MaxTextParamDict | MaxTextParamLeaf) -> tuple[MaxTextParamDict, MaxTextParamDict]:
    """Segregate a parameter dictionary into trainable LoRA adapters and frozen base weights.

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
        """Split parameters into trainable and frozen PyTrees recursively."""
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
            elif k in ("lora_a", "lora_b", "lora_scale"):
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

    Freezes all base model weights by routing them to `optax.set_to_zero()` while
    directing LoRA adapters to the specified base optimizer.

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
        """Classify parameter path as trainable adapter or frozen base."""
        last_entry = path[-1]
        key = last_entry.key if isinstance(last_entry, _KeyPathEntryProtocol) else str(last_entry)
        if key in ("lora_a", "lora_b"):
            return "trainable"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, params)
    res: _OptaxTransformProtocol = optax.multi_transform(
        {"trainable": resolved_opt, "frozen": optax.set_to_zero()},
        param_labels,
    )
    return res


def merge_lora_weights(params: MaxTextParamDict | MaxTextParamLeaf) -> MaxTextParamDict | MaxTextParamLeaf:
    """Fold LoRA adapter weights back into base MaxText weights.

    Computes:
        `W_merged = W + scale * (lora_a @ lora_b)`
    and removes adapter parameters, returning standard model weights.

    Args:
        params: Model parameter PyTree containing LoRA adapters.

    Returns:
        A new parameter dictionary with folded base weights and adapters removed.

    """
    if not isinstance(params, dict):
        return params
    res: MaxTextParamDict = {}
    for k, v in params.items():
        if isinstance(v, dict):
            v_dict: MaxTextParamDict = v
            if "kernel" in v_dict and "lora_a" in v_dict and "lora_b" in v_dict:
                raw_scale = v_dict.get("lora_scale", 1.0)
                scale = float(raw_scale) if isinstance(raw_scale, (int, float, JaxArrayProtocol)) else 1.0
                lora_a = v_dict["lora_a"]
                lora_b = v_dict["lora_b"]
                kernel = v_dict["kernel"]
                delta_w = scale * (lora_a @ lora_b)  # pyright: ignore[reportOperatorIssue]  # ty: ignore[unsupported-operator]
                merged_kernel: MaxTextParamLeaf = kernel + delta_w  # pyright: ignore[reportOperatorIssue]
                new_module: MaxTextParamDict = {module_k: module_v for module_k, module_v in v_dict.items() if module_k not in ("lora_a", "lora_b", "lora_scale")}
                new_module["kernel"] = merged_kernel
                res[k] = new_module
            else:
                res[k] = merge_lora_weights(v_dict)
        else:
            res[k] = v
    return res


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
        path = path / "maxtext_lora_adapters.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    adapters: dict[str, JaxArrayProtocol | TensorType] = {}

    def _collect(curr: MaxTextParamDict | MaxTextParamLeaf, current_path: str = "") -> None:
        """Collect adapter arrays from parameter tree into flat dict."""
        assert jnp is not None
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                _collect(v, sub_path)
            elif k in ("lora_a", "lora_b", "lora_scale"):
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
        """Count parameters recursively in parameter dictionary."""
        nonlocal trainable_count, total_count
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            if isinstance(v, dict):
                _count(v)
            else:
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    size = int(v.size)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
                    total_count += size
                    if k in ("lora_a", "lora_b"):
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
    """Apply LoRA to a model using the MaxText backend.

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
        """Initialize or transform MaxText parameter PyTree with LoRA matrices."""
        assert jax is not None
        assert jnp is not None
        raw_params = kwargs.get("params")
        if isinstance(raw_params, dict):
            params: MaxTextParamDict = raw_params  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
        else:
            assert Gemma4Model is not None
            model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            raw_init = model.init(rng, dummy_input)
            params = dict(raw_init)  # pyright: ignore[reportAssignmentType]

        params, injected_count = transform_params_to_lora(
            params=params,  # ty: ignore[invalid-argument-type]
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        raw_out_dir = kwargs.get("output_dir")
        if raw_out_dir is not None:
            save_maxtext_adapters(params, str(raw_out_dir))

        if kwargs.get("merge"):
            _ = merge_lora_weights(params)

        logger.info("MaxText LoRA applied to %d modules", injected_count)
        return injected_count

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
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
    "count_maxtext_parameters",
    "create_maxtext_lora_optimizer",
    "load_maxtext_adapters",
    "merge_lora_weights",
    "save_maxtext_adapters",
    "segregate_adapter_params",
    "transform_params_to_lora",
]
