"""MLX-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

try:
    import mlx.core as _mx
    import mlx.nn as _nn

    mx: Any = _mx
    nn: Any = _nn
except (ImportError, AttributeError):
    mx = None
    nn = None

try:
    from mlx_lm import load as _load

    load: Any = _load
except (ImportError, AttributeError):
    load = None

_ModuleBase: type = nn.Module if nn is not None else object


class MLXLoRALinear(_ModuleBase):
    """Low-rank adaptation (LoRA) linear module for MLX.

    Decomposes weight updates into low-rank matrices:
        y = x W^T + bias + (alpha / r) * (Dropout(x) @ A) @ B
    where:
        W is the base weight matrix with shape (out_features, in_features),
        A is the down-projection adapter with shape (in_features, r),
        B is the up-projection adapter with shape (r, out_features).
    """

    weight: Any
    lora_a: Any
    lora_b: Any
    bias: Any
    dropout: Any
    in_features: int
    out_features: int
    r: int
    lora_alpha: float
    scale: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = False,
    ) -> None:
        """Initialize the MLX LoRA linear layer.

        Args:
            in_features: Number of input features (d_in).
            out_features: Number of output features (d_out).
            r: Rank of the low-rank adapters (r). Must be > 0.
            lora_alpha: Scaling parameter (alpha).
            lora_dropout: Dropout probability applied to input before LoRA adapter.
            bias: Whether to allocate and apply an additive bias vector.

        Raises:
            DependencyMissingError: If MLX is not installed.
            ValueError: If rank r is less than or equal to 0.
        """
        if nn is None or mx is None:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("MLX dependencies are missing.")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scale = float(lora_alpha) / float(r)

        self.weight = mx.zeros((out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        scale_init = 1.0 / math.sqrt(r)
        self.lora_a = mx.random.uniform(
            low=-scale_init,
            high=scale_init,
            shape=(in_features, r),
        )
        self.lora_b = mx.zeros((r, out_features))

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = nn.Identity()

    @property
    def W(self) -> Any:
        """Return the base weight matrix W with shape (out_features, in_features).

        Returns:
            The frozen base weight matrix.
        """
        return self.weight

    @property
    def A(self) -> Any:
        """Return the down-projection adapter A with shape (in_features, r).

        Returns:
            The down-projection adapter tensor.
        """
        return self.lora_a

    @property
    def B(self) -> Any:
        """Return the up-projection adapter B with shape (r, out_features).

        Returns:
            The up-projection adapter tensor.
        """
        return self.lora_b

    def __setattr__(self, key: str, val: Any) -> None:
        """Set attribute, redirecting W, A, and B alias assignments to underlying parameters.

        Args:
            key: Attribute name.
            val: Attribute value.
        """
        if key == "W":
            self.weight = val
        elif key == "A":
            self.lora_a = val
        elif key == "B":
            self.lora_b = val
        else:
            super().__setattr__(key, val)

    @classmethod
    def from_linear(
        cls,
        linear: Any,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> MLXLoRALinear:
        """Construct an MLXLoRALinear instance wrapping an existing nn.Linear module.

        Args:
            linear: The source mlx.nn.Linear module whose weights will be copied.
            r: Rank of the low-rank adapters.
            lora_alpha: Scaling parameter.
            lora_dropout: Dropout probability applied prior to LoRA projection.

        Returns:
            An MLXLoRALinear layer initialized with the linear module's weights.
        """
        has_bias = hasattr(linear, "bias") and linear.bias is not None
        weight = linear.weight
        out_features, in_features = weight.shape
        lora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=has_bias,
        )
        lora_layer.weight = weight
        if has_bias:
            lora_layer.bias = linear.bias
        return lora_layer

    def __call__(self, x: Any) -> Any:
        """Execute the LoRA-augmented linear forward pass.

        Computes:
            y = x @ W^T + bias + scale * (dropout(x) @ A) @ B

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base = x @ self.weight.T
        if self.bias is not None:
            base = base + self.bias
        lora_term = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return base + self.scale * lora_term

    def save_adapters(self, file_path: str | Path) -> None:
        """Save this layer's LoRA adapter weights (lora_a, lora_b) to a safetensors file.

        Args:
            file_path: Destination path for the safetensors file.

        Raises:
            DependencyMissingError: If MLX is missing.
        """
        if mx is None:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("MLX dependencies are missing.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adapters = {"lora_a": self.lora_a, "lora_b": self.lora_b}
        mx.save_safetensors(str(path), adapters)

    def load_adapters(self, file_path: str | Path) -> None:
        """Load this layer's LoRA adapter weights (lora_a, lora_b) from a safetensors file.

        Args:
            file_path: Source path of the safetensors file.

        Raises:
            DependencyMissingError: If MLX is missing.
            KeyError: If required adapter keys are missing from the safetensors file.
        """
        if mx is None:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("MLX dependencies are missing.")
        weights = mx.load(str(file_path))
        if "lora_a" not in weights or "lora_b" not in weights:
            raise KeyError("Safetensors file missing 'lora_a' or 'lora_b' keys.")
        self.lora_a = weights["lora_a"]
        self.lora_b = weights["lora_b"]


def inject_lora(
    model: Any,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[Any, int]:
    """Inject LoRA adapters into designated projection modules of an MLX model.

    Traverses model modules, replaces targets matching `target_modules` with
    `MLXLoRALinear`, and freezes all base model weights while keeping adapter
    parameters trainable.

    Args:
        model: Target MLX neural network model.
        target_modules: List of module name suffixes or substrings to adapt (e.g. ['q_proj', 'v_proj']).
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability applied before LoRA projection.

    Returns:
        A tuple of (adapted_model, injected_count).

    Raises:
        DependencyMissingError: If MLX is not installed.
    """
    if nn is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")

    if hasattr(model, "freeze"):
        model.freeze()

    if not target_modules or not hasattr(model, "named_modules"):
        return model, 0

    to_replace: list[tuple[str, Any]] = []
    for name, submodule in model.named_modules():
        if isinstance(submodule, nn.Linear):
            leaf_name = name.split(".")[-1]
            if name in target_modules or leaf_name in target_modules or any(name.endswith(f".{t}") or f".{t}." in name for t in target_modules):
                to_replace.append((name, submodule))

    injected_count = 0
    for name, submodule in to_replace:
        parts = name.split(".")
        curr = model
        for part in parts[:-1]:
            if part.isdigit() and isinstance(curr, (list, tuple)):
                curr = curr[int(part)]
            else:
                curr = getattr(curr, part)

        lora_module = MLXLoRALinear.from_linear(
            submodule,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_module.freeze()
        lora_module.unfreeze(keys=["lora_a", "lora_b"])

        last = parts[-1]
        if last.isdigit() and isinstance(curr, list):
            curr[int(last)] = lora_module
        else:
            setattr(curr, last, lora_module)
        injected_count += 1

    return model, injected_count


def save_adapter_weights(model: Any, save_path: str | Path) -> None:
    """Save all LoRA adapter weights from an MLX model to a safetensors file.

    Extracts all trainable parameters (LoRA adapters) and serializes them into
    the specified safetensors file.

    Args:
        model: MLX model containing LoRA adapters.
        save_path: File path or directory path to save the adapter weights.

    Raises:
        DependencyMissingError: If MLX is missing.
    """
    if mx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    from mlx.utils import tree_flatten

    path = Path(save_path)
    if path.is_dir() or path.suffix != ".safetensors":
        path = path / "adapter.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)

    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), trainable_dict)


def load_adapter_weights(model: Any, load_path: str | Path) -> None:
    """Load LoRA adapter weights into an MLX model from a safetensors file.

    Args:
        model: MLX model containing LoRA adapters.
        load_path: Path to the safetensors file.

    Raises:
        DependencyMissingError: If MLX is missing.
        FileNotFoundError: If the adapter weights file does not exist.
    """
    if mx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {load_path}")
    if hasattr(model, "load_weights"):
        model.load_weights(str(path), strict=False)


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: object,
) -> JSONDict:
    """Apply LoRA to a model using the MLX backend.

    Args:
        model_name: The name or path of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including 'output_dir' and 'model'.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
    """
    if nn is None or load is None or mx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    status = "completed"
    try:
        if "model" in kwargs and kwargs["model"] is not None:
            model = kwargs["model"]
        else:
            (model, _) = load(model_name)

        model, _ = inject_lora(
            model=model,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        if "output_dir" in kwargs and kwargs["output_dir"] is not None:
            save_adapter_weights(model, str(kwargs["output_dir"]))
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        status = f"failed: {e!s}"
    return {
        "backend": "mlx",
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
    }
