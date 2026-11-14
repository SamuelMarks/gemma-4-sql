"""Tensor comparison and fixture generation utilities for tests."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Concatenate, ParamSpec, Protocol, Self, TypeVar, runtime_checkable

from gemma_4_sql._types_models import (
    KerasCompiledOptimizerProtocol,
    KerasFitHistoryProtocol,
    KerasLayerProtocol,
    KerasPreprocessorProtocol,
    KerasSamplerProtocol,
)
from gemma_4_sql.backends.keras.peft import KerasDenseLayerProtocol, _KerasTensorInputProtocol
from gemma_4_sql.backends.lazy_loader import (
    ModuleNamespaceProtocol,
    catch_optional_imports,
    import_optional_symbol,
    optional_import,
)
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    ForwardArg,
    KerasVariableProtocol,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    NestedScalarList,
    TensorProtocol,
    TfTensorProtocol,
)

with catch_optional_imports():
    import jax
    import torch

type _NestedTensorScalar = bool | int | float | complex | Sequence[_NestedTensorScalar]


@runtime_checkable
class FlattenableTensorProtocol(Protocol):
    """Protocol for tensors supporting `.shape`, `.flatten()`, and `.tolist()`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide tensor dimensions."""
        ...

    def flatten(self) -> FlattenableTensorProtocol:
        """Return a 1D flattened tensor view.

        Returns:
            Flattened `FlattenableTensorProtocol` view.

        """
        ...

    def tolist(self) -> _NestedTensorScalar:
        """Convert flattened tensor to a list of floats.

        Returns:
            Nested scalar representation.

        """
        ...


type ComparableTensor = FlattenableTensorProtocol | Sequence[float] | Sequence[Sequence[float]]


def create_dummy_weight_matrix(rows: int, cols: int, seed: float = 0.42) -> list[list[float]]:
    """Generate a deterministic 2D matrix of floating-point values for testing.

    Args:
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
        seed: Initial floating-point seed value for pseudorandom simulation.

    Returns:
        2D nested list representing the generated weight matrix.

    """
    matrix: list[list[float]] = []
    val = seed
    for _r in range(rows):
        row: list[float] = []
        for _c in range(cols):
            val = (val * 1.337 + 0.123) % 2.0 - 1.0
            row.append(round(val, 4))
        matrix.append(row)
    return matrix


def assert_sequences_close(
    actual: Sequence[float],
    expected: Sequence[float],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Assert that two numerical sequences are element-wise equal within tolerance.

    Args:
        actual: The actual output numerical sequence.
        expected: The reference expected numerical sequence.
        rtol: Relative tolerance limit.
        atol: Absolute tolerance limit.

    Raises:
        AssertionError: If lengths differ or any element diverges beyond tolerance.

    """
    if len(actual) != len(expected):
        raise AssertionError(f"Length mismatch: {len(actual)} vs {len(expected)}")
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        diff = abs(a - b)
        limit = atol + rtol * abs(b)
        if diff > limit:
            raise AssertionError(f"Divergence at index {i}: actual={a} != expected={b} (diff={diff} > limit={limit})")


def compute_snr(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in decibels between signals.

    Args:
        original: Uncompressed or target numerical signal.
        reconstructed: Quantized or reconstructed numerical signal.

    Returns:
        SNR in decibels (B).

    Raises:
        ValueError: If `original` and `reconstructed` have different lengths.

    """
    if len(original) != len(expected := reconstructed):
        raise ValueError("Sequence lengths must match for SNR computation.")
    signal_power = sum(x * x for x in original)
    noise_power = sum((x - y) * (x - y) for x, y in zip(original, expected, strict=True))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _flatten_nested_scalars(val: _NestedTensorScalar) -> list[float]:
    """Recursively flatten a scalar or nested sequence into a 1D list of floats.

    Args:
        val: Scalar or nested sequence returned by `.tolist()`.

    Returns:
        Flattened 1D list of `float` values.

    """
    if isinstance(val, complex):
        return [float(val.real)]
    if isinstance(val, (bool, int, float)):
        return [float(val)]
    return [elem for sub in val for elem in _flatten_nested_scalars(sub)]


def _extract_shape_and_flat(tensor: ComparableTensor) -> tuple[list[int], list[float]]:
    """Extract dimensions and flattened float values from a tensor or sequence.

    Args:
        tensor: Tensor or nested numeric sequence to inspect.

    Returns:
        Tuple of `(shape_dimensions, flattened_float_values)`.

    """
    if isinstance(tensor, FlattenableTensorProtocol):
        flat_list = tensor.flatten().tolist()
        return list(tensor.shape), _flatten_nested_scalars(flat_list)
    if tensor and isinstance(tensor[0], Sequence):
        rows: list[Sequence[float]] = [r for r in tensor if isinstance(r, Sequence)]
        return [len(rows), len(rows[0])], [float(x) for row in rows for x in row]
    return [len(tensor)], [float(x) for x in tensor if isinstance(x, int | float)]


def assert_tensors_close(
    actual: ComparableTensor,
    expected: ComparableTensor,
    backend: str = "pytorch",
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Compare tensor values flexibly across frameworks.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        backend: Backend framework name.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If shapes differ or elements diverge.
        ValueError: If `backend` is not recognized.

    """
    if backend not in {"pytorch", "jax", "mlx"}:
        raise ValueError(f"Unknown backend for tensor comparison: {backend}")
    actual_shape, actual_flat = _extract_shape_and_flat(actual)
    expected_shape, expected_flat = _extract_shape_and_flat(expected)
    assert actual_shape == expected_shape
    assert_sequences_close(actual_flat, expected_flat, rtol=rtol, atol=atol)


@runtime_checkable
class TypedKerasVariableProtocol(KerasVariableProtocol, Protocol):
    """Typed `keras.Variable` protocol exposing `.dtype` and `.trainable`."""

    trainable: bool

    @property
    def dtype(self) -> DTypeProtocol | str:
        """Provide variable dtype."""
        ...


@runtime_checkable
class TypedKerasTensorProtocol(TfTensorProtocol, Protocol):
    """Typed Keras tensor supporting arithmetic, `.copy()`, `.shape`, and `.dtype`."""

    def copy(self) -> TypedKerasTensorProtocol:
        """Return a copy of the tensor.

        Returns:
            Copied `TypedKerasTensorProtocol`.

        """
        ...

    def __bool__(self) -> bool:
        """Convert scalar boolean tensor to Python `bool`.

        Returns:
            Boolean truth value.

        """
        ...


@runtime_checkable
class TypedKerasDenseInstanceProtocol(KerasDenseLayerProtocol, KerasLayerProtocol, Protocol):
    """Typed `keras.layers.Dense` instance protocol."""

    @property
    def kernel(self) -> TypedKerasVariableProtocol:
        """Provide dense kernel variable."""
        ...

    @property
    def weights(self) -> Sequence[TypedKerasVariableProtocol]:
        """Provide dense weights."""
        ...

    @property
    def trainable_weights(self) -> Sequence[TypedKerasVariableProtocol]:
        """Provide trainable dense weights."""
        ...

    def __call__(
        self,
        inputs: TfTensorProtocol | _KerasTensorInputProtocol,
        *,
        training: bool | None = None,
    ) -> TypedKerasTensorProtocol:
        """Apply dense projection.

        Args:
            inputs: Input activation tensor.
            training: Optional training mode flag.

        Returns:
            Projected `TypedKerasTensorProtocol`.

        """
        ...

    def quantize(self, mode: str, /) -> None:
        """Quantize dense weights in-place.

        Args:
            mode: Quantization mode (`'int8'` or `'int4'`).

        """
        ...


@runtime_checkable
class TypedKerasModelInstanceProtocol(KerasLayerProtocol, Protocol):
    """Typed `keras.Model` / `keras.Sequential` instance protocol."""

    sampler: KerasSamplerProtocol | str | None
    preprocessor: KerasPreprocessorProtocol | None

    @property
    def layers(self) -> list[KerasLayerProtocol]:
        """Provide list of child layers."""
        ...

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> TypedKerasTensorProtocol:
        """Run forward pass.

        Args:
            *args: Input tensors or arguments.
            **kwargs: Optional `training` or forward keyword arguments.

        Returns:
            Output `TypedKerasTensorProtocol`.

        """
        ...

    def generate(
        self,
        *args: ForwardArg | Sequence[str],
        **kwargs: ForwardArg,
    ) -> TfTensorProtocol | Sequence[str] | str:
        """Generate token sequences or SQL strings.

        Args:
            *args: Prompt inputs.
            **kwargs: Generation options.

        Returns:
            Generated output.

        """
        ...

    def compile(
        self,
        *args: str | Callable[..., TensorProtocol] | None,
        **kwargs: (
            str
            | float
            | Sequence[str]
            | Callable[..., TensorProtocol]
            | KerasSamplerProtocol
            | KerasCompiledOptimizerProtocol
            | None
        ),
    ) -> None:
        """Compile model.

        Args:
            *args: Positional compile arguments.
            **kwargs: Keyword compile arguments.

        """
        ...

    def fit(self, *args: ForwardArg, **kwargs: ForwardArg) -> KerasFitHistoryProtocol:
        """Fit model on training batches.

        Args:
            *args: Dataset or tensor arguments.
            **kwargs: Fit options.

        Returns:
            `KerasFitHistoryProtocol` history object.

        """
        ...

    def save(self, filepath: Path | str, /) -> None:
        """Save model to `.keras` archive.

        Args:
            filepath: Destination `.keras` archive path.

        """
        ...

    def load_weights(self, filepath: Path | str, /) -> None:
        """Load weights from `filepath`.

        Args:
            filepath: Source weights path.

        """
        ...


@runtime_checkable
class TypedKerasLayersNamespaceProtocol(Protocol):
    """Typed `keras.layers` namespace protocol for tests."""

    @property
    def Layer(self) -> type[KerasLayerProtocol]:
        """Provide `keras.layers.Layer` class."""
        ...

    @property
    def Dense(self) -> Callable[..., TypedKerasDenseInstanceProtocol]:
        """Provide `keras.layers.Dense` class/constructor."""
        ...

    def Input(self, *, shape: tuple[int, ...], dtype: str | None = None) -> TypedKerasTensorProtocol:
        """Create symbolic Keras input tensor.

        Args:
            shape: Unbatched input shape tuple.
            dtype: Optional dtype name.

        Returns:
            Symbolic `TypedKerasTensorProtocol`.

        """
        ...

    def Embedding(self, *, input_dim: int, output_dim: int) -> Callable[[TfTensorProtocol], TypedKerasTensorProtocol]:
        """Construct an `Embedding` layer.

        Args:
            input_dim: Vocabulary size.
            output_dim: Embedding dimension.

        Returns:
            Callable embedding layer.

        """
        ...


@runtime_checkable
class TypedKerasSavingNamespaceProtocol(Protocol):
    """Typed `keras.saving` namespace protocol."""

    def load_model(self, filepath: Path | str, /) -> TypedKerasModelInstanceProtocol:
        """Load a saved `.keras` model.

        Args:
            filepath: Source `.keras` archive path.

        Returns:
            Reloaded `TypedKerasModelInstanceProtocol`.

        """
        ...


@runtime_checkable
class TypedKerasConfigNamespaceProtocol(Protocol):
    """Typed `keras.config` namespace protocol."""

    def set_dtype_policy(self, policy: DTypeProtocol | str, /) -> None:
        """Set active global Keras dtype policy.

        Args:
            policy: Dtype policy name or object.

        """
        ...


@runtime_checkable
class TypedKerasModuleProtocol(Protocol):
    """Typed `keras` module protocol for test suites."""

    @property
    def layers(self) -> TypedKerasLayersNamespaceProtocol:
        """Provide `keras.layers` namespace."""
        ...

    @property
    def saving(self) -> TypedKerasSavingNamespaceProtocol:
        """Provide `keras.saving` namespace."""
        ...

    @property
    def config(self) -> TypedKerasConfigNamespaceProtocol:
        """Provide `keras.config` namespace."""
        ...

    @property
    def Variable(self) -> Callable[..., KerasVariableProtocol]:
        """Provide `keras.Variable` constructor."""
        ...

    def Input(self, *, shape: tuple[int, ...], dtype: str | None = None) -> TypedKerasTensorProtocol:
        """Create symbolic Keras input tensor.

        Args:
            shape: Unbatched input shape tuple.
            dtype: Optional dtype name.

        Returns:
            Symbolic `TypedKerasTensorProtocol`.

        """
        ...

    def Model(
        self,
        *,
        inputs: TfTensorProtocol,
        outputs: TfTensorProtocol,
    ) -> TypedKerasModelInstanceProtocol:
        """Construct functional `keras.Model`.

        Args:
            inputs: Symbolic input tensor.
            outputs: Symbolic output tensor.

        Returns:
            Constructed `TypedKerasModelInstanceProtocol`.

        """
        ...

    def Sequential(
        self,
        layers: Sequence[KerasLayerProtocol | TfTensorProtocol],
        /,
    ) -> TypedKerasModelInstanceProtocol:
        """Construct `keras.Sequential` model.

        Args:
            layers: Sequence of Keras layers.

        Returns:
            Constructed `TypedKerasModelInstanceProtocol`.

        """
        ...


@runtime_checkable
class TypedKerasOpsProtocol(Protocol):
    """Typed `keras.ops` protocol for test suites."""

    def ones(self, shape: tuple[int, ...], *, dtype: str | None = None) -> TypedKerasTensorProtocol:
        """Create ones tensor.

        Args:
            shape: Output shape tuple.
            dtype: Optional dtype string.

        Returns:
            Ones `TypedKerasTensorProtocol`.

        """
        ...

    def zeros(self, shape: tuple[int, ...], *, dtype: str | None = None) -> TypedKerasTensorProtocol:
        """Create zeros tensor.

        Args:
            shape: Output shape tuple.
            dtype: Optional dtype string.

        Returns:
            Zeros `TypedKerasTensorProtocol`.

        """
        ...

    def full(
        self,
        shape: Sequence[int | None],
        fill_value: float | int,
        *,
        dtype: str | None = None,
    ) -> TypedKerasTensorProtocol:
        """Create constant-filled tensor.

        Args:
            shape: Output shape sequence.
            fill_value: Constant fill value.
            dtype: Optional dtype string.

        Returns:
            Constant `TypedKerasTensorProtocol`.

        """
        ...

    def linspace(self, start: float, stop: float, num: int = 50) -> TypedKerasTensorProtocol:
        """Create evenly spaced tensor.

        Args:
            start: Start value.
            stop: Stop value.
            num: Sample count.

        Returns:
            1D `TypedKerasTensorProtocol`.

        """
        ...

    def reshape(self, x: TfTensorProtocol, newshape: tuple[int, ...], /) -> TypedKerasTensorProtocol:
        """Reshape tensor `x`.

        Args:
            x: Input tensor.
            newshape: Target dimensions.

        Returns:
            Reshaped `TypedKerasTensorProtocol`.

        """
        ...

    def stop_gradient(self, variable: TfTensorProtocol | KerasVariableProtocol, /) -> TypedKerasTensorProtocol:
        """Detach `variable` from gradient tracking.

        Args:
            variable: Tensor or variable.

        Returns:
            Detached `TypedKerasTensorProtocol`.

        """
        ...

    def copy(self, x: TfTensorProtocol, /) -> TypedKerasTensorProtocol:
        """Return a copy of tensor `x`.

        Args:
            x: Input tensor.

        Returns:
            Copied `TypedKerasTensorProtocol`.

        """
        ...

    def isclose(
        self,
        x1: TfTensorProtocol,
        x2: TfTensorProtocol,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> TypedKerasTensorProtocol:
        """Compare `x1` and `x2` elementwise within tolerance.

        Args:
            x1: First tensor.
            x2: Second tensor.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Elementwise boolean `TypedKerasTensorProtocol`.

        """
        ...

    def all(self, x: TfTensorProtocol, /) -> TypedKerasTensorProtocol:
        """Reduce boolean tensor with logical AND.

        Args:
            x: Boolean tensor.

        Returns:
            Scalar boolean `TypedKerasTensorProtocol`.

        """
        ...


def get_typed_keras() -> TypedKerasModuleProtocol:
    """Return the loaded `keras` module narrowed to `TypedKerasModuleProtocol`.

    Returns:
        `TypedKerasModuleProtocol` instance.

    """
    from gemma_4_sql.backends.keras import _env as _keras_env

    _ = _keras_env.ACTIVE_KERAS_BACKEND
    mod = optional_import("keras", TypedKerasModuleProtocol)
    assert mod is not None
    return mod


def get_typed_keras_ops() -> TypedKerasOpsProtocol:
    """Return the loaded `keras.ops` module narrowed to `TypedKerasOpsProtocol`.

    Returns:
        `TypedKerasOpsProtocol` instance.

    """
    from gemma_4_sql.backends.keras import _env as _keras_env

    _ = _keras_env.ACTIVE_KERAS_BACKEND
    mod = optional_import("keras.ops", TypedKerasOpsProtocol)
    assert mod is not None
    return mod


@runtime_checkable
class _TorchSaveSafetensorsProtocol(Protocol):
    """Protocol for `safetensors.torch.save_file`."""

    def __call__(self, tensors: dict[str, torch.Tensor], filename: Path | str, /) -> None:
        """Save PyTorch tensors to `filename`.

        Args:
            tensors: Mapping of tensor names to `torch.Tensor` values.
            filename: Destination `.safetensors` file path.

        """
        ...


@runtime_checkable
class _TorchLoadSafetensorsProtocol(Protocol):
    """Protocol for `safetensors.torch.load_file`."""

    def __call__(self, filename: Path | str, /) -> dict[str, torch.Tensor]:
        """Load PyTorch tensors from `filename`.

        Args:
            filename: Source `.safetensors` file path.

        Returns:
            Loaded state dictionary.

        """
        ...


@runtime_checkable
class _FlaxSaveSafetensorsProtocol(Protocol):
    """Protocol for `safetensors.flax.save_file`."""

    def __call__(self, tensors: dict[str, jax.Array], filename: Path | str, /) -> None:
        """Save JAX arrays to `filename`.

        Args:
            tensors: Mapping of tensor names to `jax.Array` values.
            filename: Destination `.safetensors` file path.

        """
        ...


def torch_save_safetensors(tensors: dict[str, torch.Tensor], filename: Path | str) -> None:
    """Save `tensors` via `safetensors.torch.save_file` with strict typing.

    Args:
        tensors: Mapping of tensor names to `torch.Tensor` values.
        filename: Destination `.safetensors` file path.

    """
    fn = import_optional_symbol("safetensors.torch", "save_file", _TorchSaveSafetensorsProtocol)
    assert fn is not None
    fn(tensors, filename)


def torch_load_safetensors(filename: Path | str) -> dict[str, torch.Tensor]:
    """Load `dict[str, torch.Tensor]` via `safetensors.torch.load_file` with strict typing.

    Args:
        filename: Source `.safetensors` file path.

    Returns:
        Loaded state dictionary.

    """
    fn = import_optional_symbol("safetensors.torch", "load_file", _TorchLoadSafetensorsProtocol)
    assert fn is not None
    return fn(filename)


def flax_save_safetensors(tensors: dict[str, jax.Array], filename: Path | str) -> None:
    """Save `tensors` via `safetensors.flax.save_file` with strict typing.

    Args:
        tensors: Mapping of tensor names to `jax.Array` values.
        filename: Destination `.safetensors` file path.

    """
    fn = import_optional_symbol("safetensors.flax", "save_file", _FlaxSaveSafetensorsProtocol)
    assert fn is not None
    fn(tensors, filename)


@runtime_checkable
class _MlxShapeLeafProtocol(Protocol):
    """Structural protocol for MLX parameter tensors exposing `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide tensor shape dimensions."""
        ...


type MxEvalInput = (
    MlxArrayProtocol
    | _MlxShapeLeafProtocol
    | TensorProtocol
    | Mapping[str, MxEvalInput]
    | Sequence[MxEvalInput]
    | float
    | int
    | str
    | None
)


@runtime_checkable
class _MxEvalModuleProtocol(Protocol):
    """Protocol for `mlx.core` exposing `eval` and `save_safetensors`."""

    def eval(self, *args: MxEvalInput) -> None:
        """Evaluate lazy MLX arrays and parameter trees.

        Args:
            *args: Arrays or nested parameter structures.

        """
        ...

    def save_safetensors(self, file: Path | str, arrays: Mapping[str, MlxArrayProtocol], /) -> None:
        """Save MLX arrays to `file`.

        Args:
            file: Destination `.safetensors` path.
            arrays: Named MLX arrays.

        """
        ...


@runtime_checkable
class _MlxParameterizedModelProtocol(Protocol):
    """Protocol for an MLX `Module` exposing `.parameters()`."""

    def parameters(self) -> MxEvalInput:
        """Return parameter dictionary for lazy graph evaluation."""
        ...


def mx_eval(*args: MxEvalInput) -> None:
    """Evaluate lazy MLX arrays/trees without untyped `mx.eval` stub diagnostics.

    Args:
        *args: Arrays, parameter mappings, or optimizer states.

    """
    mod = optional_import("mlx.core", _MxEvalModuleProtocol)
    assert mod is not None
    mod.eval(*args)


def mx_eval_model(*models: _MlxParameterizedModelProtocol) -> None:
    """Evaluate `.parameters()` on one or more MLX `nn.Module` instances without `Unknown` dict leakage.

    Args:
        *models: MLX `nn.Module` instances whose parameters should be materialized.

    """
    mod = optional_import("mlx.core", _MxEvalModuleProtocol)
    assert mod is not None
    for model in models:
        mod.eval(model.parameters())


@runtime_checkable
class _TorchSeedModuleProtocol(Protocol):
    """Protocol for `torch` exposing `.manual_seed(...)`."""

    def manual_seed(self, seed: int, /) -> _TorchGeneratorSeedResultProtocol:
        """Seed PyTorch RNG."""
        ...


@runtime_checkable
class _TorchGeneratorSeedResultProtocol(Protocol):
    """Structural marker protocol satisfied by `torch.Generator`."""


def torch_manual_seed(seed: int) -> None:
    """Seed PyTorch global RNG cleanly without private `torch.random.default_generator` access.

    Args:
        seed: Deterministic integer random seed.

    """
    mod = optional_import("torch", _TorchSeedModuleProtocol)
    assert mod is not None
    _ = mod.manual_seed(seed)


def mx_save_safetensors(file: Path | str, arrays: Mapping[str, MlxArrayProtocol]) -> None:
    """Save MLX arrays via `mlx.core.save_safetensors` with strict typing.

    Args:
        file: Destination `.safetensors` path.
        arrays: Mapping of tensor names to MLX arrays.

    """
    mod = optional_import("mlx.core", _MxEvalModuleProtocol)
    assert mod is not None
    mod.save_safetensors(str(file), arrays)


_TMlxGradModel = TypeVar("_TMlxGradModel")
_PMlxGrad = ParamSpec("_PMlxGrad")


@runtime_checkable
class TypedMlxRandomProtocol(Protocol):
    """Typed `mlx.core.random` namespace protocol for deterministic MLX test fixtures."""

    def seed(self, seed: int, /) -> None:
        """Seed the MLX random number generator.

        Args:
            seed: Integer random seed.

        """
        ...

    def normal(
        self,
        shape: Sequence[int],
        *,
        dtype: DTypeProtocol | None = None,
    ) -> MlxArrayProtocol:
        """Sample standard normal MLX array values.

        Args:
            shape: Output tensor shape dimensions.
            dtype: Optional MLX dtype.

        Returns:
            Sampled `MlxArrayProtocol`.

        """
        ...

    def uniform(
        self,
        *,
        low: float,
        high: float,
        shape: tuple[int, int],
    ) -> MlxArrayProtocol:
        """Sample uniform random MLX array values in `[low, high)`.

        Args:
            low: Lower uniform bound.
            high: Upper uniform bound.
            shape: 2D shape tuple.

        Returns:
            Sampled `MlxArrayProtocol`.

        """
        ...


type TypedMlxArrayInput = (
    MlxArrayProtocol
    | TensorProtocol
    | NestedScalarList
    | Sequence[int]
    | Sequence[float]
    | Sequence[Sequence[int]]
    | Sequence[Sequence[float]]
)


@runtime_checkable
class TypedMlxCoreProtocol(ModuleNamespaceProtocol, Protocol):
    """Typed `mlx.core` module protocol avoiding `Unknown` diagnostics when MLX is uninstalled."""

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `mlx.core.float32` dtype."""
        ...

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `mlx.core.int32` dtype."""
        ...

    @property
    def random(self) -> TypedMlxRandomProtocol:
        """Provide `mlx.core.random` namespace."""
        ...

    def array(
        self,
        val: TypedMlxArrayInput,
        /,
        *,
        dtype: DTypeProtocol | None = None,
    ) -> MlxArrayProtocol:
        """Construct an MLX array from `val`.

        Args:
            val: Input scalar, nested sequence, or tensor.
            dtype: Optional target MLX data type.

        Returns:
            Constructed `MlxArrayProtocol`.

        """
        ...

    def zeros(
        self,
        shape: int | Sequence[int],
        /,
        *,
        dtype: DTypeProtocol | None = None,
    ) -> MlxArrayProtocol:
        """Construct a zero-filled MLX array.

        Args:
            shape: Target dimension size or shape tuple.
            dtype: Optional target MLX data type.

        Returns:
            Zero-initialized `MlxArrayProtocol`.

        """
        ...

    def ones(
        self,
        shape: int | Sequence[int],
        /,
        *,
        dtype: DTypeProtocol | None = None,
    ) -> MlxArrayProtocol:
        """Construct a ones-filled MLX array.

        Args:
            shape: Target dimension size or shape tuple.
            dtype: Optional target MLX data type.

        Returns:
            Ones-initialized `MlxArrayProtocol`.

        """
        ...

    def stack(self, arrays: Sequence[MlxArrayProtocol], /, axis: int = 0) -> MlxArrayProtocol:
        """Stack a sequence of MLX arrays along a new axis.

        Args:
            arrays: Sequence of arrays to stack.
            axis: Insertion axis index.

        Returns:
            Stacked `MlxArrayProtocol`.

        """
        ...

    def reshape(self, a: MlxArrayProtocol, shape: tuple[int, int], /) -> MlxArrayProtocol:
        """Reshape `a` to `shape`.

        Args:
            a: Input MLX array.
            shape: Target 2D shape tuple.

        Returns:
            Reshaped `MlxArrayProtocol`.

        """
        ...

    def mean(self, a: MlxArrayProtocol, /, axis: int | None = None) -> MlxArrayProtocol:
        """Compute arithmetic mean over `a`.

        Args:
            a: Input MLX array.
            axis: Optional reduction axis.

        Returns:
            Reduced mean `MlxArrayProtocol`.

        """
        ...

    def abs(self, a: MlxArrayProtocol, /) -> MlxArrayProtocol:
        """Compute elementwise absolute value of `a`.

        Args:
            a: Input MLX array.

        Returns:
            Absolute value `MlxArrayProtocol`.

        """
        ...

    def max(self, a: MlxArrayProtocol, /, axis: int | None = None) -> MlxArrayProtocol:
        """Compute maximum value over `a`.

        Args:
            a: Input MLX array.
            axis: Optional reduction axis.

        Returns:
            Maximum `MlxArrayProtocol`.

        """
        ...

    def allclose(
        self,
        a: MlxArrayProtocol,
        b: MlxArrayProtocol,
        /,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> MlxArrayProtocol:
        """Compare two MLX arrays elementwise within tolerance.

        Args:
            a: Left-hand MLX array.
            b: Right-hand MLX array.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Scalar boolean `MlxArrayProtocol`.

        """
        ...

    def array_equal(self, a: MlxArrayProtocol, b: MlxArrayProtocol, /) -> MlxArrayProtocol:
        """Check exact elementwise equality of two MLX arrays.

        Args:
            a: Left-hand MLX array.
            b: Right-hand MLX array.

        Returns:
            Scalar boolean `MlxArrayProtocol`.

        """
        ...

    def eval(self, *args: MxEvalInput) -> None:
        """Evaluate lazy MLX computation graphs.

        Args:
            *args: Lazy arrays or parameter trees.

        """
        ...

    def save_safetensors(self, file: str, arrays: Mapping[str, MlxArrayProtocol], /) -> None:
        """Save named MLX arrays to a `.safetensors` file.

        Args:
            file: Destination path string.
            arrays: Mapping from tensor names to `MlxArrayProtocol`.

        """
        ...

    def load(self, file: str, /) -> dict[str, MlxArrayProtocol]:
        """Load named MLX arrays from a `.safetensors` file.

        Args:
            file: Source `.safetensors` path string.

        Returns:
            Loaded dictionary mapping tensor names to `MlxArrayProtocol`.

        """
        ...


@runtime_checkable
class TypedMlxModuleProtocol(Protocol):
    """Typed `mlx.nn.Module` container protocol supporting dictionary submodule composition and parameter trees."""

    def __init__(self) -> None:
        """Initialize an empty `mlx.nn.Module` container."""
        ...

    def parameters(self) -> dict[str, MlxArrayProtocol | TensorProtocol]:
        """Return nested parameter dictionary.

        Returns:
            Dictionary of module parameters.

        """
        ...

    def trainable_parameters(self) -> dict[str, MlxArrayProtocol | TensorProtocol]:
        """Return nested trainable parameter dictionary.

        Returns:
            Dictionary of trainable module parameters.

        """
        ...

    def update(
        self,
        parameters: Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]],
    ) -> Self | None:
        """Update module parameters in-place.

        Args:
            parameters: Replacement parameter tree.

        Returns:
            Optional self reference or `None`.

        """
        ...

    def freeze(self) -> Self | None:
        """Freeze module parameters.

        Returns:
            Optional self reference or `None`.

        """
        ...

    def named_modules(self) -> Iterable[tuple[str, TypedMlxModuleProtocol]]:
        """Yield `(dotted_name, submodule)` pairs across the module hierarchy.

        Returns:
            Iterable of `(name, module)` pairs.

        """
        ...

    def __getitem__(self, key: str, /) -> TypedMlxSubmoduleEntry:
        """Retrieve a child module, list of modules, or parameter array at `key`.

        Args:
            key: Child module or parameter key.

        Returns:
            Stored submodule, module list, or array.

        """
        ...

    def __setitem__(self, key: str, value: TypedMlxSubmoduleEntry, /) -> None:
        """Register a child module, list of modules, or parameter array under `key`.

        Args:
            key: Child module or parameter key.
            value: Submodule, module list, or array to store.

        """
        ...


@runtime_checkable
class TypedMlxCallableModuleProtocol(TypedMlxModuleProtocol, Protocol):
    """Typed callable `mlx.nn.Module` subclass protocol (e.g., `Linear`, `Embedding`, `Sequential`)."""

    def __call__(
        self,
        *args: ForwardArg | MlxArrayProtocol,
        **kwargs: ForwardArg | MlxArrayProtocol,
    ) -> MlxArrayProtocol:
        """Execute module forward pass.

        Args:
            *args: Positional input tensors.
            **kwargs: Keyword forward options.

        Returns:
            Output `MlxArrayProtocol`.

        """
        ...


@runtime_checkable
class TypedMlxLinearProtocol(TypedMlxCallableModuleProtocol, Protocol):
    """Typed `mlx.nn.Linear` layer protocol."""

    @property
    def weight(self) -> MlxArrayProtocol:
        """Provide 2D linear weight matrix."""
        ...

    @property
    def bias(self) -> MlxArrayProtocol:
        """Provide 1D linear bias vector."""
        ...

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        """Initialize an MLX `Linear` layer.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether to include bias.

        """
        ...


@runtime_checkable
class TypedMlxQuantizedLinearProtocol(TypedMlxCallableModuleProtocol, Protocol):
    """Typed `mlx.nn.QuantizedLinear` layer protocol."""

    @property
    def weight(self) -> MlxArrayProtocol:
        """Provide packed quantized weight buffer."""
        ...

    @property
    def scales(self) -> MlxArrayProtocol:
        """Provide quantization group scales."""
        ...

    @property
    def biases(self) -> MlxArrayProtocol:
        """Provide quantization group biases."""
        ...


@runtime_checkable
class TypedMlxEmbeddingProtocol(TypedMlxCallableModuleProtocol, Protocol):
    """Typed `mlx.nn.Embedding` layer protocol."""

    @property
    def weight(self) -> MlxArrayProtocol:
        """Provide 2D embedding lookup matrix."""
        ...

    def __init__(self, num_embeddings: int, dims: int) -> None:
        """Initialize an MLX `Embedding` table.

        Args:
            num_embeddings: Vocabulary size.
            dims: Embedding feature dimension.

        """
        ...


@runtime_checkable
class TypedMlxRMSNormProtocol(TypedMlxCallableModuleProtocol, Protocol):
    """Typed `mlx.nn.RMSNorm` layer protocol."""

    @property
    def weight(self) -> MlxArrayProtocol:
        """Provide 1D RMSNorm scale vector."""
        ...

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        """Initialize an MLX `RMSNorm` layer.

        Args:
            dims: Feature dimension.
            eps: Numerical stability epsilon.

        """
        ...


@runtime_checkable
class TypedMlxSequentialProtocol(TypedMlxCallableModuleProtocol, Protocol):
    """Typed `mlx.nn.Sequential` container protocol."""

    layers: list[TypedMlxCallableModuleProtocol]

    def __init__(self, *modules: TypedMlxCallableModuleProtocol) -> None:
        """Initialize a sequential chain of MLX modules.

        Args:
            *modules: Ordered child MLX modules.

        """
        ...


type TypedMlxSubmoduleEntry = (
    TypedMlxModuleProtocol
    | TypedMlxCallableModuleProtocol
    | MlxCausalLMProtocol
    | MlxArrayProtocol
    | list[TypedMlxCallableModuleProtocol | MlxCausalLMProtocol]
    | None
)


@runtime_checkable
class TypedMlxDropoutCallableProtocol(Protocol):
    """Protocol for an MLX `Dropout` or `Identity` callable layer."""

    def __call__(self, x: MlxArrayProtocol, /) -> MlxArrayProtocol:
        """Apply dropout or identity pass-through to `x`.

        Args:
            x: Input array.

        Returns:
            Output `MlxArrayProtocol`.

        """
        ...


@runtime_checkable
class TypedMlxNNProtocol(ModuleNamespaceProtocol, Protocol):
    """Typed `mlx.nn` module protocol for MLX test suites."""

    @property
    def Module(self) -> type[TypedMlxModuleProtocol]:
        """Provide `mlx.nn.Module` base class."""
        ...

    @property
    def Linear(self) -> type[TypedMlxLinearProtocol]:
        """Provide `mlx.nn.Linear` class."""
        ...

    @property
    def QuantizedLinear(self) -> type[TypedMlxQuantizedLinearProtocol]:
        """Provide `mlx.nn.QuantizedLinear` class."""
        ...

    @property
    def Embedding(self) -> type[TypedMlxEmbeddingProtocol]:
        """Provide `mlx.nn.Embedding` class."""
        ...

    @property
    def RMSNorm(self) -> type[TypedMlxRMSNormProtocol]:
        """Provide `mlx.nn.RMSNorm` class."""
        ...

    @property
    def Sequential(self) -> type[TypedMlxSequentialProtocol]:
        """Provide `mlx.nn.Sequential` class."""
        ...

    def Dropout(self, p: float = 0.5) -> TypedMlxDropoutCallableProtocol:
        """Construct an MLX `Dropout` module.

        Args:
            p: Dropout probability.

        Returns:
            Callable dropout module.

        """
        ...

    def Identity(self) -> TypedMlxDropoutCallableProtocol:
        """Construct an MLX `Identity` module.

        Returns:
            Callable identity module.

        """
        ...

    def value_and_grad(
        self,
        model: _TMlxGradModel,
        fn: Callable[Concatenate[_TMlxGradModel, _PMlxGrad], MlxArrayProtocol],
    ) -> Callable[Concatenate[_TMlxGradModel, _PMlxGrad], tuple[MlxArrayProtocol, dict[str, MlxArrayProtocol]]]:
        """Transform scalar-loss `fn` to compute `(loss, grads)` with respect to `model`.

        Args:
            model: Target MLX module.
            fn: Loss function returning scalar `MlxArrayProtocol`.

        Returns:
            Differentiated `(loss, grads)` callable.

        """
        ...

    def quantize(
        self,
        model: TypedMlxModuleProtocol | MlxCausalLMProtocol,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        """Quantize linear layers of `model` in-place.

        Args:
            model: Target MLX module.
            group_size: Quantization block size.
            bits: Bit width per weight.

        """
        ...


@runtime_checkable
class TypedMlxOptimizerInstanceProtocol(Protocol):
    """Typed MLX optimizer instance protocol exposing `.state` and `.update(...)`."""

    @property
    def state(self) -> MxEvalInput:
        """Provide optimizer state tree."""
        ...

    def update(
        self,
        model: _MlxParameterizedModelProtocol | MlxCausalLMProtocol,
        gradients: Mapping[str, MxEvalInput],
    ) -> None:
        """Apply parameter updates from `gradients` onto `model`.

        Args:
            model: Target trainable MLX module.
            gradients: Gradient tree matching trainable parameters.

        """
        ...


@runtime_checkable
class TypedMlxOptimProtocol(ModuleNamespaceProtocol, Protocol):
    """Typed `mlx.optimizers` module protocol."""

    def AdamW(self, learning_rate: float) -> TypedMlxOptimizerInstanceProtocol:
        """Construct an `AdamW` optimizer.

        Args:
            learning_rate: Step size scalar.

        Returns:
            Initialized `TypedMlxOptimizerInstanceProtocol`.

        """
        ...


@runtime_checkable
class TypedMlxUtilsProtocol(ModuleNamespaceProtocol, Protocol):
    """Typed `mlx.utils` module protocol exposing `tree_flatten`."""

    def tree_flatten(self, tree: MxEvalInput, /) -> list[tuple[str, MlxArrayProtocol]]:
        """Flatten nested MLX parameter dictionary/list tree into `(dotted_key, array)` pairs.

        Args:
            tree: Nested parameter tree.

        Returns:
            List of `(dotted_key, MlxArrayProtocol)` entries.

        """
        ...


@runtime_checkable
class TypedMlxLmProtocol(ModuleNamespaceProtocol, Protocol):
    """Typed `mlx_lm` module protocol exposing `load`."""

    def load(self, path_or_hf_repo: str, /) -> tuple[MlxCausalLMProtocol, str | None]:
        """Load an MLX model and tokenizer.

        Args:
            path_or_hf_repo: Model repository or local path.

        Returns:
            `(model, tokenizer)` pair.

        """
        ...


def get_typed_mx() -> TypedMlxCoreProtocol:
    """Return the loaded `mlx.core` module narrowed to `TypedMlxCoreProtocol`.

    Returns:
        `TypedMlxCoreProtocol` instance.

    """
    mod = optional_import("mlx.core", TypedMlxCoreProtocol)
    assert mod is not None
    return mod


def get_typed_mlx_nn() -> TypedMlxNNProtocol:
    """Return the loaded `mlx.nn` module narrowed to `TypedMlxNNProtocol`.

    Returns:
        `TypedMlxNNProtocol` instance.

    """
    mod = optional_import("mlx.nn", TypedMlxNNProtocol)
    assert mod is not None
    return mod


def get_typed_mlx_optim() -> TypedMlxOptimProtocol:
    """Return the loaded `mlx.optimizers` module narrowed to `TypedMlxOptimProtocol`.

    Returns:
        `TypedMlxOptimProtocol` instance.

    """
    mod = optional_import("mlx.optimizers", TypedMlxOptimProtocol)
    assert mod is not None
    return mod


def get_typed_mlx_utils() -> TypedMlxUtilsProtocol:
    """Return the loaded `mlx.utils` module narrowed to `TypedMlxUtilsProtocol`.

    Returns:
        `TypedMlxUtilsProtocol` instance.

    """
    mod = optional_import("mlx.utils", TypedMlxUtilsProtocol)
    assert mod is not None
    return mod


def get_typed_mlx_lm() -> TypedMlxLmProtocol:
    """Return the loaded `mlx_lm` module narrowed to `TypedMlxLmProtocol`.

    Returns:
        `TypedMlxLmProtocol` instance.

    """
    mod = optional_import("mlx_lm", TypedMlxLmProtocol)
    assert mod is not None
    return mod
