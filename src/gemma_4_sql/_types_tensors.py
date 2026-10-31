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

type NestedScalarList = list[float] | list[int] | list[bool] | list[list[float]] | list[list[int]] | list[list[bool]] | list[list[list[float]]] | list[list[list[int]]] | list[list[list[list[float]]]]
type DTypeSpec = type[int | float | bool] | str | None


@runtime_checkable
class DTypeProtocol(Protocol):
    """Protocol representing a backend tensor data type descriptor."""

    def __str__(self) -> str:
        """Return the string representation of the data type."""
        ...


@runtime_checkable
class NDArrayProtocol(Protocol):
    """Protocol representing an n-dimensional backend array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the element data type of the array."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the array into nested Python lists."""
        ...

    def reshape(self, *shape: int | tuple[int, ...]) -> NDArrayProtocol:
        """Reshape the array without changing its data."""
        ...

    def transpose(self, *axes: int | tuple[int, ...]) -> NDArrayProtocol:
        """Permute the dimensions of the array."""
        ...

    def astype(self, dtype: DTypeProtocol | DTypeSpec) -> NDArrayProtocol:
        """Return a copy of the array cast to a specified type."""
        ...

    def flatten(self) -> NDArrayProtocol:
        """Return a copy of the array collapsed into one dimension."""
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> NDArrayProtocol:
        """Slice or index into the array."""
        ...

    def __setitem__(
        self,
        item: int | slice | tuple[int | slice, ...],
        value: NDArrayProtocol | float | bool | Sequence[float],
        /,
    ) -> None:
        """Assign values to a slice or index of the array."""
        ...

    def __sub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Subtract an array or scalar from this array."""
        ...

    def __rsub__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-subtract this array from an array or scalar."""
        ...

    def __add__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Add an array or scalar to this array."""
        ...

    def __radd__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-add this array to an array or scalar."""
        ...

    def __mul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Multiply this array by an array or scalar."""
        ...

    def __rmul__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-multiply this array by an array or scalar."""
        ...

    def __truediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Divide this array by an array or scalar."""
        ...

    def __rtruediv__(self, other: NDArrayProtocol | float, /) -> NDArrayProtocol:
        """Right-divide an array or scalar by this array."""
        ...

    def __neg__(self) -> NDArrayProtocol:
        """Negate the array elementwise."""
        ...


@runtime_checkable
class TensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol defining the interface for framework tensor objects (JAX, PyTorch, TF, MLX)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def reshape(self, *shape: int) -> Self:
        """Reshape the tensor to the given dimensions."""
        ...

    def transpose(self, ax0: int, ax1: int, /) -> Self:
        """Transpose two tensor axes."""
        ...

    def mean(self) -> Self:
        """Compute the arithmetic mean of the tensor elements."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor."""
        ...

    def __float__(self) -> float:
        """Convert single-element tensor to Python float."""
        ...

    def __index__(self) -> int:
        """Convert scalar integer tensor to Python index."""
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | Self | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Slice or index into the tensor."""
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __truediv__(self, other: Self | float, /) -> Self:
        """Divide this tensor by another tensor or scalar."""
        ...

    def __rtruediv__(self, other: Self | float, /) -> Self:
        """Right-divide a scalar or tensor by this tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two JAX arrays."""
        ...


@runtime_checkable
class TorchTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for `torch.Tensor` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element tensor."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the tensor to a nested Python list or scalar."""
        ...

    def to(self, device_or_dtype: DTypeProtocol | DTypeSpec | None, /) -> Self:
        """Move or cast the tensor to the specified device or dtype."""
        ...

    def detach(self) -> Self:
        """Return a new tensor detached from the current computation graph."""
        ...

    def clone(self) -> Self:
        """Return a copy of the tensor."""
        ...

    def view(self, *shape: int) -> Self:
        """Return a new tensor with the same data but a different shape."""
        ...

    def size(self, dim: int, /) -> int:
        """Return the size of the tensor along dimension `dim`."""
        ...

    def backward(self) -> None:
        """Compute the gradient of current tensor w.r.t. graph leaves."""
        ...

    def mean(self, dim: int | None = None) -> Self:
        """Compute the arithmetic mean of the tensor elements."""
        ...

    def sum(self, dim: int = -1) -> Self:
        """Compute the sum of the tensor elements."""
        ...

    def unsqueeze(self, dim: int, /) -> Self:
        """Insert a singleton dimension at `dim`."""
        ...

    def squeeze(self, dim: int = -1, /) -> Self:
        """Remove a singleton dimension at `dim`."""
        ...

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        ...

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> Self:
        """Slice or index into the tensor."""
        ...

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


@runtime_checkable
class TfTensorProtocol(SupportsFloat, SupportsIndex, Protocol):
    """Protocol for TensorFlow `tf.Tensor` and Keras tensor objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the tensor shape tuple."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the rank (number of dimensions) of the tensor."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the tensor."""
        ...

    def numpy(self) -> NDArrayProtocol:
        """Convert the TensorFlow tensor to an n-dimensional array."""
        ...

    def __sub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __rsub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-subtract this tensor from another tensor or scalar."""
        ...

    def __add__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __radd__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-add this tensor to another tensor or scalar."""
        ...

    def __mul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


@runtime_checkable
class JaxArrayProtocol(TensorProtocol, Protocol):
    """Protocol for `jax.Array` objects supporting `.astype(...)` and `.tolist()`."""

    def astype(self, dtype: DTypeProtocol | DTypeSpec, /) -> Self:
        """Cast the JAX array to `dtype`."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the JAX array into nested Python lists or a scalar."""
        ...


@runtime_checkable
class MlxArrayProtocol(SupportsFloat, SupportsInt, Protocol):
    """Protocol for Apple Silicon `mlx.core.array` objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol | DTypeSpec:
        """Provide the data type of the array."""
        ...

    @property
    def T(self) -> Self:
        """Provide the transpose of the array."""
        ...

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a single-element array."""
        ...

    def tolist(self) -> list[int] | list[float] | list[list[int]] | list[list[float]] | float | int:
        """Convert the array to a nested Python list or scalar."""
        ...

    def mean(self, axis: int = -1) -> Self:
        """Compute the arithmetic mean of the array elements."""
        ...

    def sum(self, axis: int = -1) -> Self:
        """Compute the sum of the array elements."""
        ...

    def __getitem__(self, item: TensorIndex, /) -> Self:
        """Slice or index into the array."""
        ...

    def __sub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Subtract another array or scalar from this array."""
        ...

    def __rsub__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-subtract this array from another array or scalar."""
        ...

    def __add__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Add another array or scalar to this array."""
        ...

    def __radd__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-add this array to another array or scalar."""
        ...

    def __mul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Multiply this array by another array or scalar."""
        ...

    def __rmul__(self, other: Self | NDArrayProtocol | float, /) -> Self:
        """Right-multiply this array by a scalar or array."""
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two arrays."""
        ...

    def __neg__(self) -> Self:
        """Negate the array elementwise."""
        ...


@runtime_checkable
class ToListArrayProtocol(Protocol):
    """Protocol for any array or tensor supporting `.tolist()` and `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    def tolist(self) -> NestedScalarList | float | int | bool:
        """Convert the array into nested Python lists."""
        ...


@runtime_checkable
class ScalarItemExtractorProtocol(Protocol):
    """Protocol for scalar tensors supporting `.item()` extraction."""

    def item(self) -> float | int | bool:
        """Extract a Python scalar value."""
        ...


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
    "NestedScalarList",
    "ScalarItemExtractorProtocol",
    "TensorIndex",
    "TensorProtocol",
    "TensorType",
    "TfTensorProtocol",
    "ToListArrayProtocol",
    "TorchTensorProtocol",
]
