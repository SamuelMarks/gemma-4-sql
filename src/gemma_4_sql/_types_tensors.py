"""Tensor, array, data type, and scalar extraction protocols across backends."""

from __future__ import annotations

import types
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Protocol,
    Self,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    Union,
    overload,
    runtime_checkable,
)

TensorIndex = Union[
    int,
    slice,
    types.EllipsisType,
    None,
    Sequence[int],
    Sequence[bool],
    "TensorProtocol",
    "NDArrayProtocol",
    tuple[
        Union[int, slice, types.EllipsisType, Sequence[int], Sequence[bool], "TensorProtocol", "NDArrayProtocol", None],
        ...,
    ],
]

type NestedScalar = bool | int | float
type NestedScalarList = NestedScalar | list[float] | list[int] | list[bool] | list[NestedScalarList]
type DTypeSpec = type[int | float | bool] | str | None


@runtime_checkable
class DTypeProtocol(Protocol):
    """Protocol representing a backend tensor data type descriptor."""

    def __str__(self) -> str:
        """Return the string representation of the data type.

        Returns:
            Canonical string name of the data type.

        """
        ...


@runtime_checkable
class NDArrayProtocol(Protocol):
    """Protocol representing an n-dimensional backend array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array.

        Returns:
            Integer rank of the array.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the element data type of the array.

        Returns:
            Data type descriptor or specification.

        """
        ...

    def tolist(self) -> NestedScalarList:
        """Convert the array into nested Python lists or a scalar.

        Returns:
            Nested Python list or scalar matching array rank.

        """
        ...

    def reshape(self, *shape: int | tuple[int, ...]) -> NDArrayProtocol:
        """Reshape the array without changing its data.

        Args:
            *shape: Target dimension sizes or shape tuple.

        Returns:
            Reshaped array view or copy.

        """
        ...

    def transpose(self, *axes: int | tuple[int, ...]) -> NDArrayProtocol:
        """Permute the dimensions of the array.

        Args:
            *axes: Optional permutation of dimension indices.

        Returns:
            Transposed array.

        """
        ...

    def astype(self, dtype: DTypeProtocol | DTypeSpec) -> NDArrayProtocol:
        """Return a copy of the array cast to a specified type.

        Args:
            dtype: Target element data type.

        Returns:
            Array cast to `dtype`.

        """
        ...

    def flatten(self) -> NDArrayProtocol:
        """Return a copy of the array collapsed into one dimension.

        Returns:
            1D flattened array.

        """
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension.

        Returns:
            Length along axis 0.

        """
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> NDArrayProtocol:
        """Slice or index into the array.

        Args:
            item: Integer index, slice, or tuple of indices/slices.

        Returns:
            Indexed subarray.

        """
        ...

    def __setitem__(
        self,
        item: int | slice | tuple[int | slice, ...],
        value: NDArrayProtocol | float | bool | Sequence[float],
        /,
    ) -> None:
        """Assign values to a slice or index of the array.

        Args:
            item: Target index, slice, or tuple of indices/slices.
            value: Replacement scalar, sequence, or array values.

        """
        ...

    def __sub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Subtract an array or scalar from this array.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise difference array.

        """
        ...

    def __rsub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-subtract this array from an array or scalar.

        Args:
            other: Left-hand array or scalar operand.

        Returns:
            Elementwise difference array.

        """
        ...

    def __add__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Add an array or scalar to this array.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise sum array.

        """
        ...

    def __radd__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-add this array to an array or scalar.

        Args:
            other: Left-hand array or scalar operand.

        Returns:
            Elementwise sum array.

        """
        ...

    def __mul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Multiply this array by an array or scalar.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise product array.

        """
        ...

    def __rmul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-multiply this array by an array or scalar.

        Args:
            other: Left-hand array or scalar operand.

        Returns:
            Elementwise product array.

        """
        ...

    def __truediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Divide this array by an array or scalar.

        Args:
            other: Right-hand divisor array or scalar.

        Returns:
            Elementwise quotient array.

        """
        ...

    def __rtruediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-divide an array or scalar by this array.

        Args:
            other: Left-hand dividend array or scalar.

        Returns:
            Elementwise quotient array.

        """
        ...

    def __neg__(self) -> NDArrayProtocol:
        """Negate the array elementwise.

        Returns:
            Elementwise negated array.

        """
        ...


@runtime_checkable
class TensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol defining the interface for framework tensor objects (JAX, PyTorch, TF, MLX)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor.

        Returns:
            Integer rank of the tensor.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor.

        Returns:
            Element data type descriptor.

        """
        ...

    def reshape(self, *shape: int) -> Self:
        """Reshape the tensor to the given dimensions.

        Args:
            *shape: Target dimension lengths.

        Returns:
            Reshaped tensor.

        """
        ...

    def transpose(self, ax0: int, ax1: int, /) -> Self:
        """Transpose two tensor axes.

        Args:
            ax0: First axis index.
            ax1: Second axis index.

        Returns:
            Transposed tensor.

        """
        ...

    def mean(self) -> Self:
        """Compute the arithmetic mean of the tensor elements.

        Returns:
            Scalar tensor holding the mean value.

        """
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor.

        Returns:
            Extracted Python scalar (`float`, `int`, or `bool`).

        """
        ...

    def __float__(self) -> float:
        """Convert single-element tensor to Python float.

        Returns:
            Floating-point scalar value.

        """
        ...

    def __index__(self) -> int:
        """Convert scalar integer tensor to Python index.

        Returns:
            Integer scalar value.

        """
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | Self | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Slice or index into the tensor.

        Args:
            item: Index, slice, ellipsis, mask tensor, or tuple of indices.

        Returns:
            Indexed subtensor.

        """
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor.

        Args:
            other: Left-hand scalar or tensor operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __truediv__(self, other: Self | float, /) -> Self:
        """Divide this tensor by another tensor or scalar.

        Args:
            other: Right-hand divisor tensor or scalar.

        Returns:
            Elementwise quotient tensor.

        """
        ...

    def __rtruediv__(self, other: Self | float, /) -> Self:
        """Right-divide a scalar or tensor by this tensor.

        Args:
            other: Left-hand dividend scalar or tensor.

        Returns:
            Elementwise quotient tensor.

        """
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise.

        Returns:
            Elementwise negated tensor.

        """
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two tensors.

        Args:
            other: Right-hand matrix/tensor operand.

        Returns:
            Matrix product tensor.

        """
        ...


@runtime_checkable
class TorchTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for `torch.Tensor` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor.

        Returns:
            Integer rank of the tensor.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor.

        Returns:
            Data type descriptor of the tensor.

        """
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor.

        Returns:
            Extracted Python scalar.

        """
        ...

    def tolist(self) -> NestedScalarList:
        """Convert the tensor to a nested Python list or scalar.

        Returns:
            Nested Python list or scalar value.

        """
        ...

    def to(self, device: str, /) -> Self:
        """Move the tensor to the specified device identifier.

        Args:
            device: Target device identifier string (e.g., `'cpu'`, `'cuda'`, `'mps'`).

        Returns:
            Tensor on the requested device.

        """
        ...

    def detach(self) -> Self:
        """Return a new tensor detached from the current computation graph.

        Returns:
            Detached tensor sharing storage.

        """
        ...

    def clone(self) -> Self:
        """Return a copy of the tensor.

        Returns:
            Cloned tensor with independent storage.

        """
        ...

    def view(self, *shape: int) -> Self:
        """Return a new tensor with the same data but a different shape.

        Args:
            *shape: Target dimension sizes.

        Returns:
            Reshaped tensor view.

        """
        ...

    def size(self, dim: int, /) -> int:
        """Return the size of the tensor along dimension `dim`.

        Args:
            dim: Dimension index to query.

        Returns:
            Size along `dim`.

        """
        ...

    def backward(self) -> None:
        """Compute the gradient of current tensor with respect to graph leaves."""
        ...

    @overload
    def mean(self) -> Self:
        """Compute the arithmetic mean of all tensor elements."""

    @overload
    def mean(self, dim: int) -> Self:
        """Compute the arithmetic mean along `dim`."""

    def mean(self, dim: int = -1) -> Self:
        """Compute the arithmetic mean of the tensor elements.

        Args:
            dim: Dimension along which to average.

        Returns:
            Reduced mean tensor.

        """
        ...

    @overload
    def sum(self) -> Self:
        """Compute the sum of all tensor elements."""

    @overload
    def sum(self, dim: int) -> Self:
        """Compute the sum along `dim`."""

    def sum(self, dim: int = -1) -> Self:
        """Compute the sum of the tensor elements.

        Args:
            dim: Dimension along which to sum.

        Returns:
            Reduced sum tensor.

        """
        ...

    def unsqueeze(self, dim: int, /) -> Self:
        """Insert a singleton dimension at `dim`.

        Args:
            dim: Axis index at which to insert dimension of size 1.

        Returns:
            Expanded tensor view.

        """
        ...

    def squeeze(self, dim: int = -1, /) -> Self:
        """Remove a singleton dimension at `dim`.

        Args:
            dim: Axis index to squeeze.

        Returns:
            Squeezed tensor view.

        """
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension.

        Returns:
            Length along axis 0.

        """
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> Self:
        """Slice or index into the tensor.

        Args:
            item: Index, slice, or tuple of indices/slices.

        Returns:
            Indexed subtensor.

        """
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor.

        Args:
            other: Left-hand scalar or tensor operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise.

        Returns:
            Elementwise negated tensor.

        """
        ...


@runtime_checkable
class TfTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for TensorFlow `tf.Tensor` and Keras tensor objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the tensor shape tuple.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the rank (number of dimensions) of the tensor.

        Returns:
            Integer rank of the tensor.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor.

        Returns:
            Data type descriptor of the tensor.

        """
        ...

    def numpy(self) -> NDArrayProtocol:
        """Convert the TensorFlow tensor to an n-dimensional array.

        Returns:
            Host NumPy-compatible n-dimensional array.

        """
        ...

    def __sub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor.

        Args:
            other: Right-hand tensor, array, or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __rsub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar.

        Args:
            other: Left-hand tensor, array, or scalar operand.

        Returns:
            Elementwise difference tensor.

        """
        ...

    def __add__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Add another tensor or scalar to this tensor.

        Args:
            other: Right-hand tensor, array, or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __radd__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar.

        Args:
            other: Left-hand tensor, array, or scalar operand.

        Returns:
            Elementwise sum tensor.

        """
        ...

    def __mul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar.

        Args:
            other: Right-hand tensor, array, or scalar operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __rmul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor.

        Args:
            other: Left-hand scalar, array, or tensor operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise.

        Returns:
            Elementwise negated tensor.

        """
        ...


@runtime_checkable
class JaxArrayProtocol(TensorProtocol, Protocol):
    """Protocol for `jax.Array` objects supporting `.astype(...)` and `.tolist()`."""

    def astype(self, dtype: DTypeProtocol | DTypeSpec, /) -> Self:
        """Cast the JAX array to `dtype`.

        Args:
            dtype: Target JAX/NumPy dtype descriptor.

        Returns:
            Array cast to `dtype`.

        """
        ...

    def tolist(self) -> NestedScalarList:
        """Convert the JAX array into nested Python lists or a scalar.

        Returns:
            Nested Python list or scalar value.

        """
        ...


type NestedComplexScalar = bool | int | float | complex
type NestedComplexScalarList = NestedComplexScalar | list[NestedComplexScalarList]


@runtime_checkable
class MlxArrayProtocol(SupportsFloat, SupportsInt, Protocol):
    """Protocol for Apple Silicon `mlx.core.array` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the array.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array.

        Returns:
            Integer rank of the array.

        """
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the array.

        Returns:
            MLX data type descriptor.

        """
        ...

    @property
    def T(self) -> Self:
        """Provide the transpose of the array.

        Returns:
            Transposed MLX array.

        """
        ...

    def flatten(self) -> Self:
        """Return a 1D flattened view of the array.

        Returns:
            Flattened 1D MLX array.

        """
        ...

    def item(self) -> NestedComplexScalar:
        """Extract a Python scalar from a single-element array.

        Returns:
            Extracted Python scalar value.

        """
        ...

    def tolist(self) -> NestedComplexScalarList:
        """Convert the array to a nested Python list or scalar.

        Returns:
            Nested Python list or scalar value.

        """
        ...

    def mean(self, axis: int = -1) -> Self:
        """Compute the arithmetic mean of the array elements.

        Args:
            axis: Axis along which to compute the mean.

        Returns:
            Reduced mean array.

        """
        ...

    def sum(self, axis: int = -1) -> Self:
        """Compute the sum of the array elements.

        Args:
            axis: Axis along which to compute the sum.

        Returns:
            Reduced sum array.

        """
        ...

    def __getitem__(self, item: TensorIndex, /) -> Self:
        """Slice or index into the array.

        Args:
            item: Index, slice, array mask, or tuple of indices.

        Returns:
            Indexed subarray.

        """
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another array or scalar from this array.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise difference array.

        """
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this array from another array or scalar.

        Args:
            other: Left-hand array or scalar operand.

        Returns:
            Elementwise difference array.

        """
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another array or scalar to this array.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise sum array.

        """
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this array to another array or scalar.

        Args:
            other: Left-hand array or scalar operand.

        Returns:
            Elementwise sum array.

        """
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this array by another array or scalar.

        Args:
            other: Right-hand array or scalar operand.

        Returns:
            Elementwise product array.

        """
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this array by a scalar or array.

        Args:
            other: Left-hand scalar or array operand.

        Returns:
            Elementwise product array.

        """
        ...

    def __truediv__(self, other: Self | float, /) -> Self:
        """Divide this array by another array or scalar.

        Args:
            other: Right-hand divisor array or scalar.

        Returns:
            Elementwise quotient array.

        """
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two arrays.

        Args:
            other: Right-hand matrix/array operand.

        Returns:
            Matrix product array.

        """
        ...

    def __neg__(self) -> Self:
        """Negate the array elementwise.

        Returns:
            Elementwise negated array.

        """
        ...


@runtime_checkable
class ToListArrayProtocol(Protocol):
    """Protocol for any array or tensor supporting `.tolist()` and `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    def tolist(self) -> NestedScalarList | NestedComplexScalarList:
        """Convert the array into nested Python lists.

        Returns:
            Nested Python list or scalar value.

        """
        ...


@runtime_checkable
class ScalarItemExtractorProtocol(Protocol):
    """Protocol for scalar tensors supporting `.item()` extraction."""

    def item(self) -> NestedComplexScalar:
        """Extract a Python scalar value.

        Returns:
            Extracted Python scalar (`bool`, `int`, `float`, or `complex`).

        """
        ...


def extract_scalar_float(val: ScalarItemExtractorProtocol | SupportsFloat | complex) -> float:
    """Extract a real Python float from a scalar tensor, complex number, or numeric value.

    Args:
        val: Scalar tensor implementing `ScalarItemExtractorProtocol` or numeric scalar.

    Returns:
        Extracted real `float` value.

    """
    raw = val.item() if isinstance(val, ScalarItemExtractorProtocol) else val
    return float(raw.real) if isinstance(raw, complex) else float(raw)


TensorType = Union[TensorProtocol, TorchTensorProtocol, TfTensorProtocol, MlxArrayProtocol, NDArrayProtocol]
ImageInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, TensorProtocol, ToListArrayProtocol, None]
AudioInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, TensorProtocol, ToListArrayProtocol, None]

__all__ = [
    "AudioInput",
    "DTypeProtocol",
    "DTypeSpec",
    "ImageInput",
    "JaxArrayProtocol",
    "MlxArrayProtocol",
    "NDArrayProtocol",
    "NestedComplexScalar",
    "NestedComplexScalarList",
    "NestedScalar",
    "NestedScalarList",
    "ScalarItemExtractorProtocol",
    "TensorIndex",
    "TensorProtocol",
    "TensorType",
    "TfTensorProtocol",
    "ToListArrayProtocol",
    "TorchTensorProtocol",
    "extract_scalar_float",
]
