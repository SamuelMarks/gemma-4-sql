"""Strongly-typed JAX and `jax.numpy` protocols eliminating `device: Unknown` stub leakage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql._types_tensors import NestedComplexScalarList
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    ForwardOutput,
    JaxArrayProtocol,
    MlxArrayProtocol,
    NDArrayProtocol,
    NestedScalarList,
    TensorProtocol,
    TfTensorProtocol,
    ToListArrayProtocol,
    TorchTensorProtocol,
)

with catch_optional_imports():
    import jax
    import jax.numpy as _jnp_types
    from jax.lax import Precision


@runtime_checkable
class _ShapedArrayInputProtocol(Protocol):
    """Structural protocol for array-like objects exposing `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide array dimensions."""
        ...


type JaxConcreteDType = _jnp_types.dtype[_jnp_types.generic] | type[_jnp_types.generic]
type JaxDTypeSpec = JaxConcreteDType | DTypeProtocol | type[int] | type[float] | type[bool] | str | None
type JaxShapeSpec = int | Sequence[int]
type JaxArrayInput = (
    jax.Array
    | JaxArrayProtocol
    | _ShapedArrayInputProtocol
    | TensorProtocol
    | TorchTensorProtocol
    | TfTensorProtocol
    | MlxArrayProtocol
    | NDArrayProtocol
    | ToListArrayProtocol
    | ForwardOutput
    | NestedScalarList
    | NestedComplexScalarList
    | Sequence[JaxArrayInput]
    | Mapping[str, JaxArrayInput]
    | SupportsFloat
    | complex
)


@runtime_checkable
class TypedJaxDeviceProtocol(Protocol):
    """Protocol for a JAX `Device` descriptor returned by `jax.devices(...)`."""

    @property
    def platform(self) -> str:
        """Provide the device accelerator platform (`'cpu'`, `'gpu'`, `'tpu'`)."""
        ...


@runtime_checkable
class TypedJaxNumpyProtocol(Protocol):
    """Protocol for `jax.numpy` with fully-known `jax.Array` signatures."""

    @property
    def bfloat16(self) -> type[_jnp_types.generic]:
        """Provide `jnp.bfloat16` dtype."""
        ...

    @property
    def float16(self) -> type[_jnp_types.generic]:
        """Provide `jnp.float16` dtype."""
        ...

    @property
    def float32(self) -> type[_jnp_types.generic]:
        """Provide `jnp.float32` dtype."""
        ...

    @property
    def int8(self) -> type[_jnp_types.generic]:
        """Provide `jnp.int8` dtype."""
        ...

    @property
    def int16(self) -> type[_jnp_types.generic]:
        """Provide `jnp.int16` dtype."""
        ...

    @property
    def int32(self) -> type[_jnp_types.generic]:
        """Provide `jnp.int32` dtype."""
        ...

    @property
    def inf(self) -> float:
        """Provide positive floating-point infinity."""
        ...

    @property
    def generic(self) -> type[_jnp_types.generic]:
        """Provide base `jnp.generic` scalar type."""
        ...

    def dtype(self, dtype: JaxDTypeSpec, /) -> _jnp_types.dtype[_jnp_types.generic]:
        """Construct or normalize a JAX `dtype` object.

        Args:
            dtype: Dtype specification.

        Returns:
            Normalized `jax.numpy.dtype`.

        """
        ...

    def array(self, obj: JaxArrayInput, /, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Construct a `jax.Array` from `obj`.

        Args:
            obj: Input array, sequence, or scalar.
            dtype: Optional target dtype.

        Returns:
            Constructed `jax.Array`.

        """
        ...

    def asarray(self, a: JaxArrayInput, /, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Convert `a` to a `jax.Array`.

        Args:
            a: Input array, sequence, or scalar.
            dtype: Optional target dtype.

        Returns:
            Converted `jax.Array`.

        """
        ...

    def zeros(self, shape: JaxShapeSpec, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Create a zero-filled `jax.Array`.

        Args:
            shape: Output dimensions.
            dtype: Optional target dtype.

        Returns:
            Zero-initialized `jax.Array`.

        """
        ...

    def zeros_like(self, a: jax.Array, /, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Create a zero-filled `jax.Array` matching `a`.

        Args:
            a: Reference array.
            dtype: Optional target dtype.

        Returns:
            Zero-initialized `jax.Array`.

        """
        ...

    def ones(self, shape: JaxShapeSpec, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Create a ones-filled `jax.Array`.

        Args:
            shape: Output dimensions.
            dtype: Optional target dtype.

        Returns:
            Ones-initialized `jax.Array`.

        """
        ...

    def ones_like(self, a: jax.Array, /, dtype: JaxDTypeSpec = None) -> jax.Array:
        """Create a ones-filled `jax.Array` matching `a`.

        Args:
            a: Reference array.
            dtype: Optional target dtype.

        Returns:
            Ones-initialized `jax.Array`.

        """
        ...

    def full(
        self,
        shape: JaxShapeSpec,
        fill_value: float | int | bool | jax.Array,
        dtype: JaxDTypeSpec = None,
    ) -> jax.Array:
        """Create a `jax.Array` filled with `fill_value`.

        Args:
            shape: Output dimensions.
            fill_value: Constant scalar or 0D fill value.
            dtype: Optional target dtype.

        Returns:
            Constant-filled `jax.Array`.

        """
        ...

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float | None = None,
        dtype: JaxDTypeSpec = None,
    ) -> jax.Array:
        """Create evenly spaced values within a half-open interval.

        Args:
            start: Start value (or stop value when `stop` is `None`).
            stop: Optional exclusive stop bound.
            step: Optional step increment.
            dtype: Optional target dtype.

        Returns:
            1D range `jax.Array`.

        """
        ...

    def linspace(
        self,
        start: float | int,
        stop: float | int,
        num: int = 50,
        *,
        dtype: JaxDTypeSpec = None,
    ) -> jax.Array:
        """Return `num` evenly spaced samples over `[start, stop]`.

        Args:
            start: Starting value.
            stop: End value.
            num: Number of samples.
            dtype: Optional target dtype.

        Returns:
            1D `jax.Array`.

        """
        ...

    def pad(
        self,
        array: jax.Array,
        pad_width: int | Sequence[int] | Sequence[tuple[int, int]],
        mode: str = "constant",
        *,
        constant_values: float | int = 0,
    ) -> jax.Array:
        """Pad an array along its axes.

        Args:
            array: Input `jax.Array`.
            pad_width: Per-axis `(before, after)` padding widths.
            mode: Padding mode name.
            constant_values: Fill constant for `'constant'` mode.

        Returns:
            Padded `jax.Array`.

        """
        ...

    def einsum(
        self,
        subscripts: str,
        /,
        *operands: jax.Array,
        precision: Precision | str | None = None,
    ) -> jax.Array:
        """Evaluate Einstein summation convention over `operands`.

        Args:
            subscripts: Einsum equation string.
            *operands: Input `jax.Array` operands.
            precision: Optional XLA matmul precision hint.

        Returns:
            Contracted `jax.Array`.

        """
        ...

    def reshape(self, a: jax.Array, shape: JaxShapeSpec, /) -> jax.Array:
        """Reshape `a` to `shape`.

        Args:
            a: Input `jax.Array`.
            shape: Target dimensions.

        Returns:
            Reshaped `jax.Array`.

        """
        ...

    def transpose(self, a: jax.Array, axes: Sequence[int] | None = None) -> jax.Array:
        """Permute array dimensions.

        Args:
            a: Input `jax.Array`.
            axes: Optional axis permutation.

        Returns:
            Transposed `jax.Array`.

        """
        ...

    def expand_dims(self, a: jax.Array, axis: int | Sequence[int]) -> jax.Array:
        """Insert length-1 axis/axes at `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Axis or axes to insert.

        Returns:
            Expanded `jax.Array`.

        """
        ...

    def broadcast_to(self, array: jax.Array, shape: Sequence[int]) -> jax.Array:
        """Broadcast `array` to `shape`.

        Args:
            array: Input `jax.Array`.
            shape: Target broadcast shape.

        Returns:
            Broadcast `jax.Array`.

        """
        ...

    def repeat(
        self,
        a: jax.Array,
        repeats: int | jax.Array,
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
    ) -> jax.Array:
        """Repeat elements of `a` along `axis`.

        Args:
            a: Input `jax.Array`.
            repeats: Repetition count(s).
            axis: Optional axis along which to repeat.
            total_repeat_length: Optional static output length for JIT.

        Returns:
            Repeated `jax.Array`.

        """
        ...

    def concatenate(self, arrays: Sequence[jax.Array], axis: int = 0) -> jax.Array:
        """Join `arrays` along an existing `axis`.

        Args:
            arrays: Sequence of `jax.Array` tensors.
            axis: Axis index.

        Returns:
            Concatenated `jax.Array`.

        """
        ...

    def stack(self, arrays: Sequence[jax.Array], axis: int = 0) -> jax.Array:
        """Join `arrays` along a new `axis`.

        Args:
            arrays: Sequence of `jax.Array` tensors.
            axis: New axis index.

        Returns:
            Stacked `jax.Array`.

        """
        ...

    def split(
        self,
        ary: jax.Array,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
    ) -> list[jax.Array]:
        """Split `ary` into sub-arrays along `axis`.

        Args:
            ary: Input `jax.Array`.
            indices_or_sections: Equal section count or split indices.
            axis: Split axis index.

        Returns:
            List of sub-arrays.

        """
        ...

    def copy(self, a: jax.Array, /) -> jax.Array:
        """Return an array copy of `a`.

        Args:
            a: Input `jax.Array`.

        Returns:
            Copied `jax.Array`.

        """
        ...

    def where(
        self,
        condition: jax.Array | bool,
        x: jax.Array | float | int,
        y: jax.Array | float | int,
        /,
    ) -> jax.Array:
        """Select elements from `x` or `y` based on `condition`.

        Args:
            condition: Boolean mask array.
            x: True-branch array or scalar.
            y: False-branch array or scalar.

        Returns:
            Selected `jax.Array`.

        """
        ...

    def take(
        self,
        a: jax.Array,
        indices: jax.Array | Sequence[int],
        axis: int | None = None,
    ) -> jax.Array:
        """Take elements from `a` along `axis`.

        Args:
            a: Source `jax.Array`.
            indices: Index array or sequence.
            axis: Optional gather axis.

        Returns:
            Gathered `jax.Array`.

        """
        ...

    def take_along_axis(
        self,
        arr: jax.Array,
        indices: jax.Array,
        axis: int | None = -1,
    ) -> jax.Array:
        """Take values from `arr` by matching 1D index slices along `axis`.

        Args:
            arr: Source `jax.Array`.
            indices: Index `jax.Array`.
            axis: Gather axis index.

        Returns:
            Gathered `jax.Array`.

        """
        ...

    def nonzero(self, a: jax.Array, *, size: int | None = None) -> tuple[jax.Array, ...]:
        """Return indices of non-zero elements in `a`.

        Args:
            a: Input `jax.Array`.
            size: Optional static output length.

        Returns:
            Tuple of index arrays per dimension.

        """
        ...

    def abs(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise absolute value.

        Args:
            x: Input `jax.Array`.

        Returns:
            Magnitude `jax.Array`.

        """
        ...

    def sin(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise sine.

        Args:
            x: Input `jax.Array`.

        Returns:
            Sine `jax.Array`.

        """
        ...

    def cos(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise cosine.

        Args:
            x: Input `jax.Array`.

        Returns:
            Cosine `jax.Array`.

        """
        ...

    def tanh(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise hyperbolic tangent.

        Args:
            x: Input `jax.Array`.

        Returns:
            Hyperbolic tangent `jax.Array`.

        """
        ...

    def exp(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise exponential.

        Args:
            x: Input `jax.Array`.

        Returns:
            Exponential `jax.Array`.

        """
        ...

    def sqrt(self, x: jax.Array | float, /) -> jax.Array:
        """Compute elementwise principal square root.

        Args:
            x: Non-negative `jax.Array` or float.

        Returns:
            Square-root `jax.Array`.

        """
        ...

    def square(self, x: jax.Array, /) -> jax.Array:
        """Compute elementwise square.

        Args:
            x: Input `jax.Array`.

        Returns:
            Squared `jax.Array`.

        """
        ...

    def round(self, a: jax.Array, decimals: int = 0) -> jax.Array:
        """Round elements of `a` to `decimals`.

        Args:
            a: Input `jax.Array`.
            decimals: Decimal places to round to.

        Returns:
            Rounded `jax.Array`.

        """
        ...

    def clip(
        self,
        arr: jax.Array,
        min: float | int | jax.Array | None = None,
        max: float | int | jax.Array | None = None,
    ) -> jax.Array:
        """Clamp `arr` into `[min, max]`.

        Args:
            arr: Input `jax.Array`.
            min: Lower bound.
            max: Upper bound.

        Returns:
            Clamped `jax.Array`.

        """
        ...

    def maximum(self, x1: jax.Array | float, x2: jax.Array | float, /) -> jax.Array:
        """Compute elementwise maximum of `x1` and `x2`.

        Args:
            x1: First operand.
            x2: Second operand.

        Returns:
            Elementwise maximum `jax.Array`.

        """
        ...

    def matmul(self, a: jax.Array, b: jax.Array, /) -> jax.Array:
        """Matrix-multiply `a` and `b`.

        Args:
            a: Left-hand `jax.Array`.
            b: Right-hand `jax.Array`.

        Returns:
            Product `jax.Array`.

        """
        ...

    def dot(self, a: jax.Array, b: jax.Array, /) -> jax.Array:
        """Dot-product `a` and `b`.

        Args:
            a: Left-hand `jax.Array`.
            b: Right-hand `jax.Array`.

        Returns:
            Dot-product `jax.Array`.

        """
        ...

    def sum(
        self,
        a: jax.Array,
        axis: int | Sequence[int] | None = None,
        *,
        keepdims: bool = False,
    ) -> jax.Array:
        """Sum elements of `a` over `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Optional reduction axis/axes.
            keepdims: Whether to keep reduced dimensions of size 1.

        Returns:
            Summed `jax.Array`.

        """
        ...

    def mean(
        self,
        a: jax.Array,
        axis: int | Sequence[int] | None = None,
        *,
        keepdims: bool = False,
    ) -> jax.Array:
        """Compute arithmetic mean over `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Optional reduction axis/axes.
            keepdims: Whether to keep reduced dimensions of size 1.

        Returns:
            Mean `jax.Array`.

        """
        ...

    def max(
        self,
        a: jax.Array,
        axis: int | Sequence[int] | None = None,
        *,
        keepdims: bool = False,
    ) -> jax.Array:
        """Compute maximum over `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Optional reduction axis/axes.
            keepdims: Whether to keep reduced dimensions of size 1.

        Returns:
            Maximum `jax.Array`.

        """
        ...

    def cumsum(self, a: jax.Array, axis: int | None = None) -> jax.Array:
        """Compute cumulative sum along `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Optional axis index.

        Returns:
            Cumulative sum `jax.Array`.

        """
        ...

    def argmax(
        self,
        a: jax.Array,
        axis: int | None = None,
        *,
        keepdims: bool = False,
    ) -> jax.Array:
        """Return indices of maximum values along `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Optional reduction axis.
            keepdims: Whether to keep reduced dimensions of size 1.

        Returns:
            Argmax index `jax.Array`.

        """
        ...

    def argsort(
        self,
        a: jax.Array,
        axis: int = -1,
        *,
        descending: bool = False,
    ) -> jax.Array:
        """Return indices sorting `a` along `axis`.

        Args:
            a: Input `jax.Array`.
            axis: Sort axis index.
            descending: Whether to sort in descending order.

        Returns:
            Sorted index `jax.Array`.

        """
        ...

    def all(self, a: jax.Array | bool, axis: int | Sequence[int] | None = None) -> jax.Array:
        """Test whether all array elements along `axis` evaluate to `True`.

        Args:
            a: Boolean `jax.Array`.
            axis: Optional reduction axis.

        Returns:
            Boolean `jax.Array`.

        """
        ...

    def isfinite(self, x: jax.Array, /) -> jax.Array:
        """Test elementwise finiteness.

        Args:
            x: Input `jax.Array`.

        Returns:
            Boolean `jax.Array`.

        """
        ...

    def isclose(
        self,
        a: jax.Array,
        b: jax.Array | float,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> jax.Array:
        """Return boolean array where `a` and `b` are elementwise equal within tolerance.

        Args:
            a: First array.
            b: Second array or scalar.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Elementwise boolean `jax.Array`.

        """
        ...

    def allclose(
        self,
        a: jax.Array,
        b: jax.Array | float,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> jax.Array:
        """Return scalar boolean array indicating whether all elements are close.

        Args:
            a: First array.
            b: Second array or scalar.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Scalar boolean `jax.Array`.

        """
        ...

    def squeeze(self, a: jax.Array, axis: int | Sequence[int] | None = None) -> jax.Array:
        """Remove length-1 dimensions from `a`.

        Args:
            a: Input `jax.Array`.
            axis: Optional singleton axis or axes to remove.

        Returns:
            Squeezed `jax.Array`.

        """
        ...

    def not_equal(self, x1: jax.Array, x2: jax.Array | int | float, /) -> jax.Array:
        """Return elementwise `x1 != x2` boolean array.

        Args:
            x1: Left-hand `jax.Array`.
            x2: Right-hand array or scalar.

        Returns:
            Boolean `jax.Array`.

        """
        ...

    def array_equal(self, a1: jax.Array, a2: jax.Array) -> jax.Array:
        """Return scalar boolean array indicating whether `a1` and `a2` have identical shape and elements.

        Args:
            a1: First array.
            a2: Second array.

        Returns:
            Scalar boolean `jax.Array`.

        """
        ...


def is_jnp_missing(mod: TypedJaxNumpyProtocol | None) -> bool:
    """Return whether a `jnp` binding is `None` (supporting runtime test overrides).

    Args:
        mod: Loaded `TypedJaxNumpyProtocol` or `None`.

    Returns:
        `True` when `mod` is `None`.

    """
    return mod is None


__all__ = [
    "JaxArrayInput",
    "JaxConcreteDType",
    "JaxDTypeSpec",
    "JaxShapeSpec",
    "TypedJaxDeviceProtocol",
    "TypedJaxNumpyProtocol",
    "is_jnp_missing",
]
