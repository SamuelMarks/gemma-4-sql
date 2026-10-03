"""MaxText-specific PEFT / LoRA parameter transformation implementation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    import numpy as _np
    import optax as _optax

    jax: Any = _jax
    jnp: Any = _jnp
    np: Any = _np
    optax: Any = _optax
except (ImportError, AttributeError):
    jax = None
    jnp = None
    np = None
    optax = None

try:
    import maxtext.models.gemma4 as _gemma4

    Gemma4Model: Any = getattr(_gemma4, "Gemma4Model", None)
except (ImportError, AttributeError):
    Gemma4Model = None


def transform_params_to_lora(
    params: dict[str, Any],
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rng: Any = None,
) -> tuple[dict[str, Any], int]:
    """Transform a MaxText parameter PyTree by injecting low-rank adapter matrices.

    For any projection module matching `target_modules` (e.g. q_proj, v_proj),
    decomposes parameter updates by augmenting the layer with `lora_a` (shape d_in x r)
    and `lora_b` (shape r x d_out) arrays.

    Args:
        params: Nested dictionary representing model parameters.
        target_modules: List of module name substrings/suffixes to adapt.
        lora_r: LoRA rank dimension. Must be > 0.
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability for LoRA layers.
        rng: Optional JAX PRNGKey for initializing adapter matrices.

    Returns:
        A tuple of (transformed_params_dict, count_of_injected_modules).

    Raises:
        DependencyMissingError: If JAX is missing.
        ValueError: If rank r is less than or equal to 0.

    """
    if jax is None or jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing.")
    if lora_r <= 0:
        raise ValueError(f"LoRA rank r must be positive, got {lora_r}")

    if rng is None:
        rng = jax.random.PRNGKey(0)

    injected_count = 0

    def _inject(curr: dict[str, Any], current_path: str = "") -> dict[str, Any]:
        """Inject LoRA factors into matching parameter dictionaries."""
        nonlocal injected_count, rng
        if not isinstance(curr, dict):
            return curr
        res: dict[str, Any] = {}
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                v_dict = cast(dict[str, Any], v)
                # Check if this leaf dictionary represents a target projection module containing 'kernel'
                if "kernel" in v_dict and any(k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in target_modules):
                    kernel: Any = v_dict["kernel"]
                    d_in, d_out = kernel.shape
                    scale_init = 1.0 / math.sqrt(lora_r)
                    rng, sub_rng = jax.random.split(rng)
                    lora_a = jax.random.uniform(
                        sub_rng,
                        (d_in, lora_r),
                        dtype=kernel.dtype,
                        minval=-scale_init,
                        maxval=scale_init,
                    )
                    lora_b = jnp.zeros((lora_r, d_out), dtype=kernel.dtype)
                    new_v: dict[str, Any] = dict(v_dict)
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


def segregate_adapter_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Segregate a parameter dictionary into trainable LoRA adapters and frozen base weights.

    Args:
        params: Model parameter PyTree.

    Returns:
        A tuple of (trainable_lora_params, frozen_base_params).

    """
    trainable: dict[str, Any] = {}
    frozen: dict[str, Any] = {}

    def _split(src: dict[str, Any], dst_trainable: dict[str, Any], dst_frozen: dict[str, Any]) -> None:
        """Split parameters into trainable and frozen PyTrees recursively."""
        if not isinstance(src, dict):
            # If a scalar or array is passed directly, treat it as frozen base weight unless the caller knows otherwise.
            # But the signature says dict[str, Any], so this shouldn't happen for valid inputs.
            return
        for k, v in src.items():
            if isinstance(v, dict):
                v_dict = cast(dict[str, Any], v)
                sub_t: dict[str, Any] = {}
                sub_f: dict[str, Any] = {}
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
    params: dict[str, Any],
    base_optimizer: object = None,
) -> Any:
    """Create an Optax multi_transform optimizer updating only LoRA parameters.

    Freezes all base model weights by routing them to optax.set_to_zero() while
    directing LoRA adapters to the specified base optimizer.

    Args:
        params: Model parameter PyTree containing LoRA adapters.
        base_optimizer: Optax transformation for trainable weights (defaults to optax.adam(1e-4)).

    Returns:
        An Optax GradientTransformation configured for adapter-only training.

    Raises:
        DependencyMissingError: If Optax or JAX dependencies are missing.

    """
    if optax is None or jax is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Optax or JAX dependencies are missing.")

    if base_optimizer is None:
        base_optimizer = optax.adam(1e-4)

    def _label_fn(path: Any, _val: Any) -> str:
        """Classify parameter path as trainable adapter or frozen base."""
        key = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if key in ("lora_a", "lora_b"):
            return "trainable"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, params)
    return optax.multi_transform(
        {"trainable": base_optimizer, "frozen": optax.set_to_zero()},
        param_labels,
    )


def merge_lora_weights(params: dict[str, Any]) -> dict[str, Any]:
    """Fold LoRA adapter weights back into base MaxText weights.

    Computes:
        W_merged = W + scale * (lora_a @ lora_b)
    and removes adapter parameters, returning standard model weights.

    Args:
        params: Model parameter PyTree containing LoRA adapters.

    Returns:
        A new parameter dictionary with folded base weights and adapters removed.

    """
    res: dict[str, Any] = {}
    if not isinstance(params, dict):
        return params
    for k, v in params.items():
        if isinstance(v, dict):
            v_dict = cast(dict[str, Any], v)
            if "kernel" in v_dict and "lora_a" in v_dict and "lora_b" in v_dict:
                scale = float(v_dict.get("lora_scale", 1.0))
                delta_w = scale * (v_dict["lora_a"] @ v_dict["lora_b"])
                merged_kernel = v_dict["kernel"] + delta_w
                new_module = {module_k: module_v for module_k, module_v in v_dict.items() if module_k not in ("lora_a", "lora_b", "lora_scale")}
                new_module["kernel"] = merged_kernel
                res[k] = new_module
            else:
                res[k] = merge_lora_weights(v_dict)
        else:
            res[k] = v
    return res


def save_maxtext_adapters(params: dict[str, Any], save_path: str | Path) -> None:
    """Save LoRA adapter weights from a parameter PyTree to an NPZ archive.

    Args:
        params: Model parameter PyTree containing LoRA adapters.
        save_path: File path or directory path destination.

    Raises:
        DependencyMissingError: If NumPy is missing.

    """
    if np is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("NumPy dependency is missing.")

    path = Path(save_path)
    if path.is_dir() or path.suffix != ".npz":
        path = path / "maxtext_lora_adapters.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    adapters: dict[str, Any] = {}

    def _collect(curr: dict[str, Any], current_path: str = "") -> None:
        """Collect adapter arrays from parameter tree into flat dict."""
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                _collect(cast(dict[str, Any], v), sub_path)
            elif k in ("lora_a", "lora_b", "lora_scale"):
                adapters[sub_path] = np.asarray(v)

    _collect(params)

    np.savez(path, **adapters)


def load_maxtext_adapters(params: dict[str, Any], load_path: str | Path) -> dict[str, Any]:
    """Load LoRA adapter weights from an NPZ file into a base parameter PyTree.

    Args:
        params: Model parameter PyTree.
        load_path: Path to the NPZ archive containing adapter weights.

    Returns:
        Updated parameter dictionary with loaded LoRA adapters.

    Raises:
        DependencyMissingError: If NumPy is missing.
        FileNotFoundError: If the load path does not exist.

    """
    if np is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("NumPy dependency is missing.")

    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found at {load_path}")

    loaded = dict(np.load(path))
    import copy

    res = copy.deepcopy(params)

    for flat_key, array_val in loaded.items():
        parts = flat_key.split(".")
        curr = res
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        if jnp is not None:
            curr[parts[-1]] = jnp.asarray(array_val)
        else:
            curr[parts[-1]] = array_val

    return res


def count_maxtext_parameters(params: dict[str, Any]) -> tuple[int, int]:
    """Count total and trainable LoRA parameters in a MaxText parameter PyTree.

    Args:
        params: Model parameter PyTree.

    Returns:
        A tuple of (total_parameter_count, trainable_lora_parameter_count).

    """
    trainable_count = 0
    total_count = 0

    def _count(curr: dict[str, Any]) -> None:
        """Count parameters recursively in parameter dictionary."""
        nonlocal trainable_count, total_count
        if not isinstance(curr, dict):
            return
        for k, v in curr.items():
            if isinstance(v, dict):
                _count(cast(dict[str, Any], v))
            elif hasattr(v, "size"):
                size = int(v.size)
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
    **kwargs: object,
) -> JSONDict:
    """Apply LoRA to a model using the MaxText backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including 'params', 'output_dir', and 'merge'.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    status = "completed"
    if jax is None or jnp is None or (Gemma4Model is None and "params" not in kwargs):
        raise DependencyMissingError("MaxText dependencies are missing.")

    injected_count = 0
    try:
        params: dict[str, Any]
        if "params" in kwargs and kwargs["params"] is not None and isinstance(kwargs["params"], dict):
            params = cast(dict[str, Any], kwargs["params"])
        else:
            if Gemma4Model is None:
                raise DependencyMissingError("MaxText dependency missing.")
            model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            params_any: Any = model.init(rng, dummy_input)
            params = cast(dict[str, Any], params_any) if isinstance(params_any, dict) else {}

        params, injected_count = transform_params_to_lora(
            params=params,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        if "output_dir" in kwargs and kwargs["output_dir"] is not None:
            save_maxtext_adapters(params, str(kwargs["output_dir"]))

        if kwargs.get("merge"):
            params = merge_lora_weights(params)

        logger.info("MaxText LoRA applied to %d modules", injected_count)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        status = f"failed: {e!s}"

    return {
        "backend": "maxtext",
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
        "injected_modules": injected_count,
    }
