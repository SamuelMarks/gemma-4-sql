"""Keras-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ModelType, TensorType

logger = logging.getLogger(__name__)

keras = optional_import("keras")
ops = optional_attr("keras", "ops")

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import PEFTResultDict

    class _LayerBase:
        """Static base class stub representing `keras.layers.Layer` during type checking."""

        built: bool

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize layer base."""

        def add_weight(self, *args: object, **kwargs: object) -> Any:
            """Create a layer weight variable."""
            return None

        def build(self, input_shape: Any) -> None:
            """Build layer weights for `input_shape`."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Execute the layer forward pass."""
            return None

else:
    _LayerBase = getattr(getattr(keras, "layers", None), "Layer", object) if keras is not None else object


class KerasLoRADense(_LayerBase):
    """Low-rank adaptation (LoRA) layer wrapper for Keras Dense layers.

    Decomposes the weight update of a Dense layer into low-rank factor matrices:
        y = W x + bias + (alpha / r) * (Dropout(x) @ A) @ B
    where:
        W is the base frozen weight kernel with shape (in_features, units),
        A is the down-projection adapter with shape (in_features, r),
        B is the up-projection adapter with shape (r, units).
    """

    dense: Any
    r: int
    lora_alpha: float
    scale: float
    lora_dropout: float
    dropout: Any
    lora_a: Any
    lora_b: Any
    _lora_built: bool

    def __init__(
        self,
        dense: Any,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        **kwargs: object,
    ) -> None:
        """Initialize the Keras LoRA Dense layer wrapper.

        Args:
            dense: The base Keras Dense layer to adapt.
            r: Rank of the low-rank adaptation matrices. Must be > 0.
            lora_alpha: Scaling parameter (alpha).
            lora_dropout: Dropout probability applied to inputs prior to LoRA projection.
            **kwargs: Additional keyword arguments for the Keras layer base class.

        Raises:
            DependencyMissingError: If Keras is missing.
            ValueError: If rank r is less than or equal to 0.

        """
        if keras is None or ops is None:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("Keras dependencies are missing.")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")

        super().__init__(**kwargs)
        self.dense = dense
        self.dense.trainable = False
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scale = float(lora_alpha) / float(r)
        self.lora_dropout = lora_dropout
        self._lora_built = False
        if lora_dropout > 0.0:
            self.dropout = keras.layers.Dropout(rate=lora_dropout)
        else:
            self.dropout = None

        if getattr(self.dense, "built", False):
            self.build()

    def build(self, input_shape: Any = None) -> None:
        """Build the LoRA layer parameters.

        Args:
            input_shape: Shape tuple of the incoming tensor.

        """
        if getattr(self, "_lora_built", False):
            return

        if not getattr(self.dense, "built", False) and input_shape is not None:
            self.dense.build(input_shape)

        in_features = self.dense.kernel.shape[0]
        units = self.dense.kernel.shape[1]

        scale_init = 1.0 / math.sqrt(self.r)

        self.lora_a = self.add_weight(
            shape=(in_features, self.r),
            initializer=keras.initializers.RandomUniform(minval=-scale_init, maxval=scale_init),
            trainable=True,
            name="lora_a",
        )
        self.lora_b = self.add_weight(
            shape=(self.r, units),
            initializer="zeros",
            trainable=True,
            name="lora_b",
        )
        self._lora_built = True
        super().build(input_shape)

    def call(self, inputs: TensorType, training: bool | None = None) -> Any:
        """Execute the forward pass combining frozen base projection and low-rank adapter.

        Args:
            inputs: Input tensor.
            training: Boolean indicating training or inference mode for dropout.

        Returns:
            Output tensor of shape (..., units).

        """
        base_out = self.dense(inputs)
        dropped = self.dropout(inputs, training=training) if self.dropout is not None else inputs
        lora_out = ops.matmul(ops.matmul(dropped, self.lora_a), self.lora_b)
        return base_out + self.scale * lora_out

    @property
    def kernel(self) -> Any:
        """Provide the base layer kernel weight.

        Returns:
            The frozen base layer kernel.

        """
        return self.dense.kernel

    @property
    def bias(self) -> Any:
        """Provide the base layer bias weight if present.

        Returns:
            The frozen base layer bias tensor, or None.

        """
        return getattr(self.dense, "bias", None)

    @property
    def W(self) -> Any:
        """Provide alias for base weight kernel.

        Returns:
            The frozen base weight tensor.

        """
        return self.dense.kernel

    @property
    def A(self) -> Any:
        """Provide alias for down-projection adapter matrix.

        Returns:
            The trainable lora_a tensor.

        """
        return self.lora_a

    @property
    def B(self) -> Any:
        """Provide alias for up-projection adapter matrix.

        Returns:
            The trainable lora_b tensor.

        """
        return self.lora_b

    def merge_weights(self) -> Any:
        """Fold low-rank adapter weights into the base Dense kernel and return the base layer.

        Computes:
            W_merged = W + scale * (lora_a @ lora_b)
        and assigns it to the underlying Dense layer kernel.

        Returns:
            The original Keras Dense layer with updated weights.

        """
        if not getattr(self, "_lora_built", False):
            return self.dense

        delta = self.scale * ops.matmul(self.lora_a, self.lora_b)
        self.dense.kernel.assign(self.dense.kernel + delta)
        self.dense.trainable = True
        return self.dense


def _traverse_keras_layers(model: Any, visitor_fn: Any) -> None:
    """Recursively traverse a Keras model or layer hierarchy with tracker lock/unlock management.

    Args:
        model: Root Keras model, layer, or container object.
        visitor_fn: Callback `(parent_obj, attr_name, child_val, recurse_fn)` invoked for each public attribute.

    """
    visited: set[int] = set()

    def _traverse(curr: Any) -> None:
        """Traverse object attributes and invoke `visitor_fn` while unlocking Keras 3 tracker."""
        curr_id = id(curr)
        if curr_id in visited:
            return
        visited.add(curr_id)

        has_tracker = hasattr(curr, "_tracker")
        if has_tracker:
            curr._tracker.unlock()

        curr_any: Any = curr
        try:
            attrs: list[str] = list(vars(curr_any).keys()) if hasattr(curr_any, "__dict__") else []
            if hasattr(curr_any, "_modules") and isinstance(curr_any._modules, dict):
                attrs.extend(curr_any._modules.keys())
            for attr_name in set(attrs):
                if attr_name.startswith("_"):
                    continue
                val = getattr(curr_any, attr_name, None)
                if val is None:
                    continue
                visitor_fn(curr_any, attr_name, val, _traverse)
        finally:
            if has_tracker:
                curr_any._tracker.lock()

    _traverse(model)


def inject_lora(
    model: ModelType,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[Any, int]:
    """Inject KerasLoRADense adapter wrappers into target dense layers of a Keras model.

    Freezes all non-adapter layers explicitly (layer.trainable = False) and replaces
    layers matching `target_modules` with `KerasLoRADense`.

    Args:
        model: Target Keras model or layer.
        target_modules: List of target layer names to wrap with LoRA.
        lora_r: Rank dimension of LoRA adapters.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability applied before adapter projection.

    Returns:
        A tuple of (adapted_model, count_of_injected_adapters).

    Raises:
        DependencyMissingError: If Keras is not available.

    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    if not target_modules or not hasattr(model, "__dict__"):
        return model, 0

    injected_count = 0
    adapted_ids: set[int] = set()

    def _visit_inject(curr_any: Any, attr_name: str, val: Any, recurse_fn: Any) -> None:
        """Visit a single attribute during LoRA injection."""
        nonlocal injected_count
        dense_cls: Any = getattr(getattr(keras, "layers", None), "Dense", type(None))
        if keras is not None and isinstance(val, dense_cls) and id(val) not in adapted_ids:
            if any(attr_name == t or t in attr_name or getattr(val, "name", "") == t for t in target_modules):
                lora_layer = KerasLoRADense(
                    dense=val,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(curr_any, attr_name, lora_layer)
                adapted_ids.add(id(val))
                injected_count += 1
            else:
                val.trainable = False
        elif isinstance(val, list):
            list_modified = False
            for idx, item in enumerate(val):
                if keras is not None and isinstance(item, dense_cls) and id(item) not in adapted_ids:
                    if any(f"{attr_name}_{idx}" == t or getattr(item, "name", "") == t for t in target_modules):
                        lora_layer = KerasLoRADense(
                            dense=item,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        )
                        val[idx] = lora_layer
                        adapted_ids.add(id(item))
                        injected_count += 1
                        list_modified = True
                    else:
                        item.trainable = False
                else:
                    recurse_fn(item)
            if list_modified:
                setattr(curr_any, attr_name, list(val))
        elif hasattr(val, "__dict__"):
            recurse_fn(val)

    _traverse_keras_layers(model, _visit_inject)
    return model, injected_count


def merge_lora_weights(model: ModelType) -> Any:
    """Fold all LoRA adapter weights back into base Dense layers across a model.

    Traverses the model and replaces every `KerasLoRADense` with its merged base Dense layer.

    Args:
        model: Keras model or layer containing KerasLoRADense layers.

    Returns:
        The model with all LoRA layers folded back into native Dense layers.

    """
    if not hasattr(model, "__dict__"):
        return model

    def _visit_merge(curr_any: Any, attr_name: str, val: Any, recurse_fn: Any) -> None:
        """Visit a single attribute during LoRA weight merging."""
        if isinstance(val, KerasLoRADense):
            merged_dense = val.merge_weights()
            setattr(curr_any, attr_name, merged_dense)
        elif isinstance(val, list):
            list_modified = False
            for idx, item in enumerate(val):
                if isinstance(item, KerasLoRADense):
                    val[idx] = item.merge_weights()
                    list_modified = True
                else:
                    recurse_fn(item)
            if list_modified:
                setattr(curr_any, attr_name, list(val))
        elif hasattr(val, "__dict__"):
            recurse_fn(val)

    _traverse_keras_layers(model, _visit_merge)
    return model


def count_parameters(model: ModelType) -> tuple[int, int]:
    """Count total and trainable weights in a Keras model.

    Args:
        model: Keras model or layer.

    Returns:
        A tuple of (total_variable_weights_count, trainable_weights_count).

    """
    total = sum(int(ops.size(w)) for w in getattr(model, "weights", [])) if hasattr(model, "weights") else 0
    trainable = sum(int(ops.size(w)) for w in getattr(model, "trainable_weights", [])) if hasattr(model, "trainable_weights") else 0
    return total, trainable


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: object,
) -> PEFTResultDict:
    """Apply LoRA to a model using the Keras backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including 'model' and 'merge'.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.

    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    def _inject() -> int:
        """Load or reuse Keras model and inject or merge LoRA weights."""
        if "model" in kwargs and kwargs["model"] is not None:
            model = kwargs["model"]
        else:
            gemma_causal_lm_cls = optional_attr("keras_nlp.models", "GemmaCausalLM")
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model = gemma_causal_lm_cls.from_preset(model_name)

        model_any: Any = model
        if hasattr(model_any, "backbone") and hasattr(model_any.backbone, "enable_lora"):
            model_any.backbone.enable_lora(rank=lora_r)
            for layer in getattr(model_any.backbone, "layers", []):
                if not getattr(layer, "trainable_variables", []):
                    layer.trainable = False
            logger.info("Enabled Keras native LoRA with rank %d", lora_r)
            injected_count = len(target_modules)
        else:
            model, injected_count = inject_lora(
                model=model_any,
                target_modules=target_modules,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        if kwargs.get("merge"):
            _ = merge_lora_weights(model_any)
        return injected_count

    return apply_lora_wrapper(
        "keras",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **kwargs,
    )
