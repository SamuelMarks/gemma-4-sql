"""JAX/Flax NNX Parameter-Efficient Fine-Tuning (PEFT / LoRA) implementation."""

from __future__ import annotations

import logging
import math
from typing import Any

from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.backends.jax.inference import _MODEL_CACHE
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ModelType, PEFTResultDict

logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
nnx = optional_import("flax.nnx")
Gemma4Config = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4Config")
Gemma4DecoderLayer = optional_attr("gemma_4_sql.backends.jax.gemma4.decoder_layer", "Gemma4DecoderLayer")
Gemma4ForCausalLM = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4ForCausalLM")

type Array = Any
_ModuleBase: Any = nnx.Module if nnx is not None else object
_ParamBase: Any = nnx.Param if nnx is not None else object


class NNXParam(_ParamBase):
    """Flax NNX parameter subclass providing uniform `.shape`, `.dtype`, and slice indexing."""

    value: Any

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying parameter value."""
        val = getattr(self, "value", None)
        return tuple(val.shape) if val is not None and hasattr(val, "shape") else ()

    @property
    def dtype(self) -> Any:
        """Return the data type of the underlying parameter value."""
        val = getattr(self, "value", None)
        return val.dtype if val is not None and hasattr(val, "dtype") else None

    def __getitem__(self, idx: Any) -> Any:
        """Index into the underlying parameter array."""
        val = getattr(self, "value", None)
        return val[idx] if val is not None else None

    def __setitem__(self, idx: Any, new_val: Any) -> None:
        """Assign values into the underlying parameter array."""
        val = getattr(self, "value", None)
        if val is not None:
            if idx is Ellipsis or idx == slice(None):
                self.value = new_val
            elif hasattr(val, "at"):
                self.value = val.at[idx].set(new_val)
            else:
                self.value = new_val
        else:
            self.value = new_val


class LoRAParam(NNXParam):
    """Marker parameter class designating low-rank adapter weights in Flax NNX."""


class NNXLoRALinear(_ModuleBase):
    """Low-rank adaptation (LoRA) linear module for Flax NNX.

    Decomposes linear weight updates into low-rank factor matrices:
        y = x W + bias + (alpha / r) * (Dropout(x) @ A) @ B
    where:
        W is the base frozen weight kernel with shape (in_features, out_features),
        A is the down-projection adapter with shape (in_features, r),
        B is the up-projection adapter with shape (r, out_features).
    """

    in_features: int
    out_features: int
    r: int
    lora_alpha: float
    scale: float
    kernel: Any
    bias: Any
    lora_a: Any
    lora_b: Any
    dropout: Any
    dtype: Any

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_bias: bool = False,
        dtype: Any = None,
        *,
        rngs: Any = None,
    ) -> None:
        """Initialize the NNX LoRA linear layer.

        Args:
            in_features: Number of input features (d_in).
            out_features: Number of output features (d_out).
            r: Rank of the low-rank adapters (r). Must be > 0.
            lora_alpha: Scaling factor (alpha).
            lora_dropout: Dropout probability applied to inputs prior to adapter projection.
            use_bias: Whether to maintain a bias parameter array.
            dtype: Computation and parameter data type (defaults to float32).
            rngs: Flax NNX random number generators.

        Raises:
            DependencyMissingError: If JAX or Flax NNX dependencies are missing.
            ValueError: If rank r is less than or equal to 0.

        """
        if nnx is None or jax is None or jnp is None:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("JAX PEFT dependencies are missing.")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scale = float(lora_alpha) / float(r)
        self.dtype = dtype if dtype is not None else jnp.float32

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Base frozen weights
        self.kernel = NNXParam(jnp.zeros((in_features, out_features), dtype=self.dtype))
        if use_bias:
            self.bias = NNXParam(jnp.zeros((out_features,), dtype=self.dtype))
        else:
            self.bias = None

        # Low-rank adapters
        scale_init = 1.0 / math.sqrt(r)
        lora_a_arr = jax.random.uniform(
            rngs.params(),
            (in_features, r),
            dtype=self.dtype,
            minval=-scale_init,
            maxval=scale_init,
        )
        self.lora_a = LoRAParam(lora_a_arr)
        self.lora_b = LoRAParam(jnp.zeros((r, out_features), dtype=self.dtype))

        # LoRA dropout
        if lora_dropout > 0.0:
            self.dropout = nnx.Dropout(rate=lora_dropout, rngs=rngs)
        else:
            self.dropout = None

    @property
    def W(self) -> Array:
        """Provide the base weight array W with shape (in_features, out_features).

        Returns:
            The base weight array value.

        """
        res: Array = getattr(self.kernel, "value", self.kernel)
        return res

    @property
    def A(self) -> Array:
        """Provide the down-projection adapter A with shape (in_features, r).

        Returns:
            The down-projection adapter array value.

        """
        res: Array = getattr(self.lora_a, "value", self.lora_a)
        return res

    @property
    def B(self) -> Array:
        """Provide the up-projection adapter B with shape (r, out_features).

        Returns:
            The up-projection adapter array value.

        """
        res: Array = getattr(self.lora_b, "value", self.lora_b)
        return res

    def train(self, mode: bool = True, **attributes: Any) -> None:
        """Set module training mode to toggle LoRA dropout active/inactive.

        Args:
            mode: True to enable training dropout, False for inference mode.
            **attributes: Optional Flax NNX module state attributes.

        """
        _ = attributes
        if self.dropout is not None:
            if mode:
                self.dropout.train()
            else:
                self.dropout.eval()

    def eval(self, **attributes: Any) -> None:
        """Set module to evaluation/inference mode disabling LoRA dropout.

        Args:
            **attributes: Optional Flax NNX module state attributes.

        """
        self.train(False, **attributes)

    @classmethod
    def from_linear(
        cls,
        linear: Any,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        *,
        rngs: Any = None,
    ) -> NNXLoRALinear:
        """Construct an NNXLoRALinear module wrapping an existing nnx.Linear.

        Args:
            linear: Source flax.nnx.Linear module whose weights will be wrapped.
            r: Rank of the low-rank adapters.
            lora_alpha: Scaling parameter.
            lora_dropout: Dropout rate applied prior to adapter projection.
            rngs: Flax NNX random number generators.

        Returns:
            An NNXLoRALinear instance sharing base kernel and bias with the source linear.

        """
        kernel_val = getattr(linear.kernel, "value", linear.kernel)
        in_features, out_features = kernel_val.shape
        use_bias = hasattr(linear, "bias") and linear.bias is not None and getattr(linear.bias, "value", linear.bias) is not None
        dtype = kernel_val.dtype

        lora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )
        lora_layer.kernel = linear.kernel
        if use_bias:
            lora_layer.bias = linear.bias
        return lora_layer

    def __call__(self, x: Array, *, deterministic: bool | None = None) -> Array:
        """Execute the forward linear projection with low-rank adaptation.

        Computes:
            y = x @ W + bias + scale * (dropout(x) @ A) @ B

        Args:
            x: Input array of shape (..., in_features).
            deterministic: Optional override flag to disable dropout during inference.

        Returns:
            Output array of shape (..., out_features).

        """
        kernel_val = getattr(self.kernel, "value", self.kernel)
        base = jnp.dot(x, kernel_val)
        if self.bias is not None:
            bias_val = getattr(self.bias, "value", self.bias)
            if bias_val is not None:
                base = base + bias_val

        if self.dropout is not None:
            dropped_x = self.dropout(x, deterministic=deterministic)
        else:
            dropped_x = x

        lora_a_val = getattr(self.lora_a, "value", self.lora_a)
        lora_b_val = getattr(self.lora_b, "value", self.lora_b)
        lora_term = jnp.dot(jnp.dot(dropped_x, lora_a_val), lora_b_val)
        out: Array = (base + self.scale * lora_term).astype(self.dtype)
        return out


def inject_lora_to_layer(
    layer: Any,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rngs: Any = None,
) -> int:
    """Inject LoRA adapters into designated projection layers of a Gemma4DecoderLayer.

    Replaces projections matching `target_modules` (e.g. q_proj, k_proj, v_proj,
    o_proj, gate_proj, up_proj, down_proj) with `NNXLoRALinear`.

    Args:
        layer: Target Gemma4DecoderLayer or transformer block.
        target_modules: List of module names to adapt.
        lora_r: Low-rank dimension.
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability.
        rngs: Optional Flax NNX random number generators.

    Returns:
        The number of adapted projection modules injected into the layer.

    """
    injected_count = 0

    # Self-attention projections: q_proj, k_proj, v_proj, o_proj
    if hasattr(layer, "self_attention"):
        attn = layer.self_attention
        for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if target in target_modules and hasattr(attn, target):
                mod = getattr(attn, target)
                if isinstance(mod, nnx.Linear):
                    lora_mod = NNXLoRALinear.from_linear(
                        mod,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        rngs=rngs,
                    )
                    setattr(attn, target, lora_mod)
                    injected_count += 1

    # MLP projections: gate_proj, up_proj, down_proj
    if hasattr(layer, "mlp"):
        mlp_targets = [layer.mlp]
        if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
            mlp_targets.append(layer.mlp.shared_experts)

        for mlp_obj in mlp_targets:
            for target in ("gate_proj", "up_proj", "down_proj"):
                if target in target_modules and hasattr(mlp_obj, target):
                    mod = getattr(mlp_obj, target)
                    if isinstance(mod, nnx.Linear):
                        lora_mod = NNXLoRALinear.from_linear(
                            mod,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            rngs=rngs,
                        )
                        setattr(mlp_obj, target, lora_mod)
                        injected_count += 1

    return injected_count


def inject_lora(
    model: ModelType,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rngs: Any = None,
) -> tuple[Any, int]:
    """Inject LoRA adapters into designated projection modules of a JAX/Flax NNX model.

    Traverses model layers and submodules, replacing matching `Linear` modules
    with `NNXLoRALinear`.

    Args:
        model: JAX model (such as Gemma4ForCausalLM or arbitrary NNX module).
        target_modules: List of target submodule names (e.g. ['q_proj', 'v_proj']).
        lora_r: Rank dimension for LoRA adapters.
        lora_alpha: Scaling coefficient for LoRA.
        lora_dropout: Dropout probability applied before adapter projection.
        rngs: Optional Flax NNX random number generators.

    Returns:
        A tuple of (adapted_model, total_injected_count).

    Raises:
        DependencyMissingError: If JAX or Flax NNX dependencies are missing.

    """
    if nnx is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    if rngs is None:
        rngs = nnx.Rngs(0)

    if not target_modules or not hasattr(model, "__dict__"):
        return model, 0

    total_injected = 0

    # If the model is a Gemma4ForCausalLM or contains layers sequence
    layers: list[Any] | None = None
    model_any: Any = model
    if hasattr(model_any, "model") and hasattr(model_any.model, "layers"):
        layers = list(model_any.model.layers)
    elif hasattr(model_any, "layers") and isinstance(model_any.layers, list):
        layers = model_any.layers

    if layers is not None:
        for layer in layers:
            total_injected += inject_lora_to_layer(
                layer,
                target_modules=target_modules,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                rngs=rngs,
            )

    # General recursive traversal fallback for remaining target modules
    visited: set[int] = set()

    def _traverse(curr: Any) -> None:
        """Traverse object attributes recursively to replace matching Linear layers with LoRA."""
        nonlocal total_injected
        curr_id = id(curr)
        if curr_id in visited or not hasattr(curr, "__dict__"):
            return
        visited.add(curr_id)

        for attr_name, val in list(curr.__dict__.items()):
            if isinstance(val, nnx.Linear):
                if any(attr_name == t or attr_name.endswith(f"_{t}") or t in attr_name for t in target_modules):
                    lora_mod = NNXLoRALinear.from_linear(
                        val,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        rngs=rngs,
                    )
                    setattr(curr, attr_name, lora_mod)
                    total_injected += 1
            elif isinstance(val, nnx.Module) or hasattr(val, "__dict__"):
                _traverse(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, nnx.Module):
                        _traverse(item)
            elif isinstance(val, dict):
                for item in val.values():
                    if isinstance(item, nnx.Module):
                        _traverse(item)

    if total_injected == 0:
        _traverse(model)

    return model, total_injected


def create_lora_optimizer(
    model: ModelType | NNXLoRALinear | Any,
    tx: Any = None,
) -> Any:
    """Create a Flax NNX Optimizer configured to update only LoRA adapter parameters.

    Ensures only `LoRAParam` nodes are updated while freezing base model parameters.

    Args:
        model: Flax NNX model with injected LoRA adapters.
        tx: Optional Optax gradient transformation (defaults to optax.adam(1e-4)).

    Returns:
        An nnx.Optimizer bound to update only LoRAParam parameters.

    Raises:
        DependencyMissingError: If Flax NNX or Optax dependencies are missing.

    """
    if nnx is None or optax is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    if tx is None:
        tx = optax.adam(1e-4)

    return nnx.Optimizer(model, tx, wrt=LoRAParam)


def count_parameters(model: ModelType) -> tuple[int, int]:
    """Count total and trainable LoRA parameters in a Flax NNX model.

    Args:
        model: Flax NNX model with LoRA adapters.

    Returns:
        A tuple of (total_parameters_count, trainable_lora_parameters_count).

    Raises:
        DependencyMissingError: If JAX or Flax NNX dependencies are missing.

    """
    if nnx is None or jax is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    _, lora_state, base_state = nnx.split(model, LoRAParam, ...)
    lora_count = sum(int(x.size) for x in jax.tree.leaves(lora_state))
    base_count = sum(int(x.size) for x in jax.tree.leaves(base_state))
    return lora_count + base_count, lora_count


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: object,
) -> PEFTResultDict:
    """Apply LoRA to a model using the JAX backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including 'model', 'config', and 'rngs'.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If JAX PEFT dependencies are missing.

    """
    if optax is None or jax is None or nnx is None or Gemma4ForCausalLM is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    def _inject() -> int:
        """Instantiate or reuse JAX model and inject LoRA layers."""
        rngs = kwargs.get("rngs")
        if rngs is None:
            rngs = nnx.Rngs(0)

        if "model" in kwargs and kwargs["model"] is not None:
            model_obj: Any = kwargs["model"]
        else:
            config = kwargs.get("config")
            if config is None and Gemma4Config is not None:
                from_preset_fn = getattr(Gemma4Config, "from_preset", None)
                config = from_preset_fn(model_name) if callable(from_preset_fn) else Gemma4Config.gemma4_e2b()
            model_obj = Gemma4ForCausalLM(config, rngs=rngs)

        model, injected_count = inject_lora(
            model=model_obj,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            rngs=rngs,
        )
        _ = nnx.split(model, LoRAParam, ...)
        _MODEL_CACHE.put_model(model_name, model)
        logger.info("Injected LoRA into %d targets", injected_count)
        return injected_count

    return apply_lora_wrapper(
        "jax",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **kwargs,
    )
