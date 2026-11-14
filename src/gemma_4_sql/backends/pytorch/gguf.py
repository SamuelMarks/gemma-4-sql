"""GGUF v3 binary serialization, tensor quantization, validation, and parameter extraction.

This module provides native conversion of PyTorch and Gemma 4 model architectures
into binary GGUF v3 containers for high-performance llama.cpp inference.
"""

from __future__ import annotations

import io
import logging
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, TypedDict, runtime_checkable

from gemma_4_sql.backends.lazy_loader import catch_optional_imports, optional_import
from gemma_4_sql.exceptions import ExportError
from gemma_4_sql.type_hints import TensorProtocol, ToListArrayProtocol, TorchCausalLMProtocol

with catch_optional_imports():
    import torch as _torch_types

logger = logging.getLogger(__name__)


@runtime_checkable
class _TorchGGUFModuleProtocol(Protocol):
    """Protocol for `torch` tensor creation and block-quantization ops in GGUF serialization."""

    @property
    def float16(self) -> _torch_types.dtype:
        """Provide `torch.float16` dtype."""
        ...

    @property
    def float32(self) -> _torch_types.dtype:
        """Provide `torch.float32` dtype."""
        ...

    @property
    def int8(self) -> _torch_types.dtype:
        """Provide `torch.int8` dtype."""
        ...

    @property
    def uint8(self) -> _torch_types.dtype:
        """Provide `torch.uint8` dtype."""
        ...

    def tensor(self, data: Sequence[float], *, dtype: _torch_types.dtype) -> _torch_types.Tensor:
        """Construct a 1D tensor from `data`.

        Args:
            data: Sequence of scalar floats.
            dtype: Target PyTorch dtype.

        Returns:
            Constructed `torch.Tensor`.

        """
        ...

    def zeros(self, size: int | tuple[int, ...], *, dtype: _torch_types.dtype) -> _torch_types.Tensor:
        """Construct a zero-filled tensor.

        Args:
            size: 1D length or shape tuple.
            dtype: Target PyTorch dtype.

        Returns:
            Zero-initialized `torch.Tensor`.

        """
        ...

    def empty(self, size: tuple[int, ...], *, dtype: _torch_types.dtype) -> _torch_types.Tensor:
        """Allocate an uninitialized tensor buffer.

        Args:
            size: Shape tuple.
            dtype: Target PyTorch dtype.

        Returns:
            Allocated `torch.Tensor`.

        """
        ...

    def ones_like(self, input_tensor: _torch_types.Tensor, /) -> _torch_types.Tensor:
        """Construct a ones tensor matching `input_tensor` shape and dtype.

        Args:
            input_tensor: Reference tensor.

        Returns:
            Ones `torch.Tensor`.

        """
        ...

    def cat(self, tensors: Sequence[_torch_types.Tensor], dim: int = 0) -> _torch_types.Tensor:
        """Concatenate `tensors` along `dim`.

        Args:
            tensors: Sequence of tensors to join.
            dim: Concatenation dimension index.

        Returns:
            Concatenated `torch.Tensor`.

        """
        ...

    def abs(self, input_tensor: _torch_types.Tensor, /) -> _torch_types.Tensor:
        """Compute elementwise absolute value.

        Args:
            input_tensor: Input tensor.

        Returns:
            Elementwise magnitude tensor.

        """
        ...

    def amax(self, input_tensor: _torch_types.Tensor, dim: int) -> _torch_types.Tensor:
        """Compute maximum values along `dim`.

        Args:
            input_tensor: Input tensor.
            dim: Reduction dimension index.

        Returns:
            Reduced maximum tensor.

        """
        ...

    def amin(self, input_tensor: _torch_types.Tensor, dim: int) -> _torch_types.Tensor:
        """Compute minimum values along `dim`.

        Args:
            input_tensor: Input tensor.
            dim: Reduction dimension index.

        Returns:
            Reduced minimum tensor.

        """
        ...

    def where(
        self,
        condition: _torch_types.Tensor,
        x: _torch_types.Tensor,
        y: _torch_types.Tensor,
        /,
    ) -> _torch_types.Tensor:
        """Select elements from `x` or `y` according to boolean `condition`.

        Args:
            condition: Boolean mask tensor.
            x: True-branch tensor.
            y: False-branch tensor.

        Returns:
            Elementwise selected tensor.

        """
        ...

    def round(self, input_tensor: _torch_types.Tensor, /) -> _torch_types.Tensor:
        """Round tensor elements to nearest integer values.

        Args:
            input_tensor: Input floating-point tensor.

        Returns:
            Rounded tensor.

        """
        ...

    def clamp(
        self,
        input_tensor: _torch_types.Tensor,
        min: float | int | None = None,
        max: float | int | None = None,
    ) -> _torch_types.Tensor:
        """Clamp tensor elements into `[min, max]`.

        Args:
            input_tensor: Input tensor.
            min: Optional minimum bound.
            max: Optional maximum bound.

        Returns:
            Clamped tensor.

        """
        ...


torch: _TorchGGUFModuleProtocol | None = optional_import("torch", _TorchGGUFModuleProtocol)

# GGML Numerical Quantization Types
GGML_TYPE_F32: int = 0
GGML_TYPE_F16: int = 1
GGML_TYPE_Q4_0: int = 2
GGML_TYPE_Q4_1: int = 3
GGML_TYPE_Q8_0: int = 8
GGML_TYPE_Q4_K: int = 12

# GGUF Metadata Value Types
GGUF_TYPE_UINT8: int = 0
GGUF_TYPE_INT8: int = 1
GGUF_TYPE_UINT16: int = 2
GGUF_TYPE_INT16: int = 3
GGUF_TYPE_UINT32: int = 4
GGUF_TYPE_INT32: int = 5
GGUF_TYPE_FLOAT32: int = 6
GGUF_TYPE_BOOL: int = 7
GGUF_TYPE_STRING: int = 8
GGUF_TYPE_ARRAY: int = 9
GGUF_TYPE_UINT64: int = 10
GGUF_TYPE_INT64: int = 11
GGUF_TYPE_FLOAT64: int = 12

GGUF_ALIGNMENT: int = 32


type GGUFMetadataScalar = str | int | float | bool
type GGUFMetadataArray = Sequence["GGUFMetadataScalar | GGUFMetadataArray"]
type GGUFMetadataValue = GGUFMetadataScalar | GGUFMetadataArray | Path | Mapping[str, GGUFMetadataScalar] | None
type NestedFloatSeq = float | int | complex | Sequence["NestedFloatSeq"]
type TensorInput = (
    TensorProtocol
    | _FlattenableArrayProtocol
    | ToListArrayProtocol
    | SupportsFloat
    | bytes
    | bytearray
    | float
    | int
    | complex
    | NestedFloatSeq
)


class GGUFTensorDescriptor(TypedDict):
    """Typed dictionary describing a single tensor header entry inside a GGUF file.

    Attributes:
        name: Canonical GGML tensor name.
        shape: GGML column-major tensor dimension list (`ne[0], ne[1], ...`).
        type: Integer GGML quantization type code.
        offset: Aligned byte offset from start of tensor payload section.

    """

    name: str
    shape: list[int]
    type: int
    offset: int


class GGUFHeaderInfo(TypedDict):
    """Typed dictionary returned by `validate_gguf_file` describing a GGUF binary header.

    Attributes:
        version: GGUF binary container format version (`2` or `3`).
        tensor_count: Total number of tensor descriptors in the file.
        metadata: Key-value metadata dictionary parsed from the header.
        tensors: Ordered list of `GGUFTensorDescriptor` entries.

    """

    version: int
    tensor_count: int
    metadata: dict[str, GGUFMetadataValue]
    tensors: list[GGUFTensorDescriptor]


@runtime_checkable
class _FlattenableArrayProtocol(Protocol):
    """Protocol for a PyTorch tensor or array supporting `.shape` and `.flatten().tolist()`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the array shape.

        Returns:
            N-dimensional integer shape tuple.

        """
        ...

    def flatten(self) -> _FlattenableArrayProtocol:
        """Flatten array to 1D.

        Returns:
            Flattened 1D array view.

        """
        ...

    def tolist(self) -> list[float]:
        """Convert 1D array to a Python list of floats.

        Returns:
            Flat list of floating-point values.

        """
        ...


@runtime_checkable
class _StateDictModelProtocol(Protocol):
    """Protocol for a model exposing a `.state_dict()` method."""

    def state_dict(self) -> Mapping[str, TensorInput]:
        """Return a mapping of parameter names to tensors.

        Returns:
            Mapping from layer parameter names to tensors.

        """
        ...


def _to_contiguous_float32_1d(arr: TensorInput) -> tuple[list[float], tuple[int, ...]]:
    """Convert a tensor, byte buffer, scalar, or nested sequence into flat floats and shape.

    Args:
        arr: Input tensor, array, byte buffer, scalar, or N-dimensional nested sequence.

    Returns:
        A 2-tuple of `(flat_float_values, row_major_shape_tuple)`.

    Raises:
        TypeError: If `arr` cannot be converted into floating-point elements.

    """
    if isinstance(arr, (bytes, bytearray)):
        vals = [float(b) for b in arr]
        return (vals, (len(vals),))
    if isinstance(arr, _FlattenableArrayProtocol):
        raw_shape = tuple(int(d) for d in arr.shape)
        shape = raw_shape if len(raw_shape) > 0 else (1,)
        vals = [float(x) for x in arr.flatten().tolist()]
        return (vals, shape)
    if isinstance(arr, ToListArrayProtocol):
        return _to_contiguous_float32_1d(arr.tolist())
    if isinstance(arr, (list, tuple)):
        if len(arr) == 0:
            return ([], (0,))
        first_elem = arr[0]
        if isinstance(first_elem, (list, tuple)):
            first_vals, sub_shape = _to_contiguous_float32_1d(first_elem)
            flat: list[float] = list(first_vals)
            for sub in arr[1:]:
                sub_vals, _ = _to_contiguous_float32_1d(sub)
                flat.extend(sub_vals)
            return (flat, (len(arr), *sub_shape))
        flat_1d: list[float] = []
        for item in arr:
            item_vals, _ = _to_contiguous_float32_1d(item)
            flat_1d.extend(item_vals)
        return (flat_1d, (len(flat_1d),))
    assert isinstance(arr, (SupportsFloat, complex))
    scalar_val = arr.real if isinstance(arr, complex) else float(arr)
    return ([float(scalar_val)], (1,))


def _flatten_values(arr: TensorInput) -> list[float]:
    """Flatten array-like structures into a 1D list of floats.

    Args:
        arr: Array, tensor, or nested list of numerical values.

    Returns:
        Flat 1D list of floating point values.

    """
    vals, _ = _to_contiguous_float32_1d(arr)
    return vals


def _uint8_tensor_to_bytes(tensor: ToListArrayProtocol) -> bytes:
    """Serialize a 1D `uint8` tensor into raw `bytes` via `ToListArrayProtocol`.

    Args:
        tensor: Flattened `uint8` tensor.

    Returns:
        Serialized raw `bytes`.

    """
    raw = tensor.tolist()
    assert isinstance(raw, list)
    return bytes(int(b) for b in raw if isinstance(b, int))


def quantize_tensor_f16(arr: TensorInput) -> bytes:
    """Quantize an input array or tensor to 16-bit half precision float bytes.

    Args:
        arr: Input array or tensor to quantize.

    Returns:
        Raw bytes representation in IEEE 754 float16 format.

    """
    flat, _ = _to_contiguous_float32_1d(arr)
    if torch is not None:
        t = torch.tensor(flat, dtype=torch.float16)
        return _uint8_tensor_to_bytes(t.view(torch.uint8))
    return struct.pack(f"<{len(flat)}e", *flat)


def quantize_tensor_q8_0(arr: TensorInput, block_size: int = 32) -> bytes:
    """Quantize an input float array into GGML Q8_0 blocks.

    Each block consists of a 16-bit float scale followed by `block_size` int8 values.

    Args:
        arr: Array-like float values to quantize.
        block_size: Number of values per block (standard: 32).

    Returns:
        Serialized Q8_0 binary byte stream.

    Raises:
        ValueError: If `block_size` is non-positive.

    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    flat, _ = _to_contiguous_float32_1d(arr)
    if torch is not None:
        a = torch.tensor(flat, dtype=torch.float32)
        rem = int(a.numel()) % block_size
        if rem != 0:
            a = torch.cat([a, torch.zeros(block_size - rem, dtype=torch.float32)])
        n_blocks = int(a.numel()) // block_size
        blocks = a.reshape(n_blocks, block_size)
        max_vals = torch.amax(torch.abs(blocks), dim=1)
        scales = torch.where(max_vals > 0, max_vals / 127.0, torch.ones_like(max_vals)).to(torch.float16)
        qs = torch.clamp(torch.round(blocks / scales.to(torch.float32).unsqueeze(1)), -128, 127).to(torch.int8)
        scale_bytes = scales.view(torch.uint8).reshape(n_blocks, 2)
        qs_bytes = qs.view(torch.uint8)
        return _uint8_tensor_to_bytes(torch.cat([scale_bytes, qs_bytes], dim=1).reshape(-1))

    rem = len(flat) % block_size
    if rem != 0:
        flat.extend([0.0] * (block_size - rem))

    out = bytearray()
    for i in range(0, len(flat), block_size):
        chunk = flat[i : i + block_size]
        max_val = max(abs(x) for x in chunk)
        scale = max_val / 127.0 if max_val > 0.0 else 1.0
        out.extend(struct.pack("<e", scale))
        for x in chunk:
            q = round(x / scale)
            q = max(-128, min(127, q))
            out.extend(struct.pack("<b", q))
    return bytes(out)


def quantize_tensor_q4_0(arr: TensorInput, block_size: int = 32) -> bytes:
    """Quantize an input float array into GGML Q4_0 blocks.

    Each block consists of a 16-bit float scale followed by packed 4-bit nibbles.

    Args:
        arr: Array-like float values to quantize.
        block_size: Number of values per block (standard: 32).

    Returns:
        Serialized Q4_0 binary byte stream.

    Raises:
        ValueError: If `block_size` is non-positive or not even.

    """
    if block_size <= 0 or block_size % 2 != 0:
        raise ValueError("block_size must be a positive even integer.")

    half_block = block_size // 2
    flat, _ = _to_contiguous_float32_1d(arr)

    if torch is not None:
        a = torch.tensor(flat, dtype=torch.float32)
        rem = int(a.numel()) % block_size
        if rem != 0:
            a = torch.cat([a, torch.zeros(block_size - rem, dtype=torch.float32)])
        n_blocks = int(a.numel()) // block_size
        blocks = a.reshape(n_blocks, block_size)
        max_vals = torch.amax(torch.abs(blocks), dim=1)
        scales = torch.where(max_vals > 0, max_vals / 8.0, torch.ones_like(max_vals)).to(torch.float16)
        s_f32 = scales.to(torch.float32).unsqueeze(1)
        denom = torch.where(s_f32 != 0, s_f32, torch.ones_like(s_f32))
        v0 = torch.clamp(torch.round(blocks[:, :half_block] / denom) + 8, 0, 15).to(torch.uint8)
        v1 = torch.clamp(torch.round(blocks[:, half_block:] / denom) + 8, 0, 15).to(torch.uint8)
        packed = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
        scale_bytes = scales.view(torch.uint8).reshape(n_blocks, 2)
        return _uint8_tensor_to_bytes(torch.cat([scale_bytes, packed], dim=1).reshape(-1))

    rem = len(flat) % block_size
    if rem != 0:
        flat.extend([0.0] * (block_size - rem))

    out = bytearray()
    for i in range(0, len(flat), block_size):
        chunk = flat[i : i + block_size]
        max_val = max(abs(x) for x in chunk)
        scale = max_val / 8.0 if max_val > 0.0 else 1.0
        out.extend(struct.pack("<e", scale))
        for j in range(half_block):
            v0 = round(chunk[j] / scale) + 8
            v1 = round(chunk[j + half_block] / scale) + 8
            v0 = max(0, min(15, v0))
            v1 = max(0, min(15, v1))
            byte_val = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
            out.append(byte_val)
    return bytes(out)


def _pack_q4_k_scales_mins(sc: Sequence[int], m: Sequence[int]) -> bytes:
    """Pack 8 6-bit sub-block scales and 8 6-bit sub-block minimums into 12 GGML Q4_K bytes.

    Args:
        sc: Sequence of 8 integers in `[0, 63]` representing sub-block scales.
        m: Sequence of 8 integers in `[0, 63]` representing sub-block minimums.

    Returns:
        12-byte packed `scales_mins` header for a Q4_K super-block.

    """
    meta = bytearray(12)
    for j in range(4):
        meta[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
        meta[j + 4] = (m[j] & 63) | ((m[j + 4] >> 4) << 6)
        meta[j + 8] = (sc[j + 4] & 15) | ((m[j + 4] & 15) << 4)
    return bytes(meta)


def _unpack_q4_k_scales_mins(meta: bytes | Sequence[int]) -> tuple[list[int], list[int]]:
    """Unpack 12 GGML Q4_K bytes into 8 6-bit sub-block scales and 8 6-bit minimums.

    Args:
        meta: 12-byte packed `scales_mins` sequence.

    Returns:
        Tuple of `(scales_list, mins_list)` each containing 8 integers in `[0, 63]`.

    """
    sc = [0] * 8
    m = [0] * 8
    for j in range(4):
        sc[j] = int(meta[j]) & 63
        m[j] = int(meta[j + 4]) & 63
        sc[j + 4] = (int(meta[j + 8]) & 15) | ((int(meta[j]) >> 6) << 4)
        m[j + 4] = (int(meta[j + 8]) >> 4) | ((int(meta[j + 4]) >> 6) << 4)
    return sc, m


def quantize_tensor_q4_k_m(arr: TensorInput) -> bytes:
    """Quantize an input float array into GGML Q4_K_M 256-element super-blocks.

    Groups elements into super-blocks of 256 elements (8 sub-blocks of 32 elements)
    with 6-bit packed sub-block scales and minimums in the 12-byte `scales_mins` header.

    Args:
        arr: Array-like float values to quantize.

    Returns:
        Serialized Q4_K_M binary byte stream (144 bytes per 256 elements).

    """
    block_size = 256
    flat, _ = _to_contiguous_float32_1d(arr)

    if torch is not None:
        a = torch.tensor(flat, dtype=torch.float32)
        rem = int(a.numel()) % block_size
        if rem != 0:
            a = torch.cat([a, torch.zeros(block_size - rem, dtype=torch.float32)])
        n_blocks = int(a.numel()) // block_size
        sub_blocks = a.reshape(n_blocks, 8, 32)
        sub_mins = torch.clamp(-torch.amin(sub_blocks, dim=2), min=0.0)
        sub_dynamic = torch.amax(sub_blocks + sub_mins.unsqueeze(2), dim=2) / 15.0

        max_scale = torch.amax(sub_dynamic, dim=1)
        max_min = torch.amax(sub_mins, dim=1)
        d = torch.where(max_scale > 0.0, max_scale / 63.0, torch.ones_like(max_scale)).to(torch.float16)
        dmin = torch.where(max_min > 0.0, max_min / 63.0, torch.ones_like(max_min)).to(torch.float16)

        d_f32 = d.to(torch.float32).unsqueeze(1)
        dmin_f32 = dmin.to(torch.float32).unsqueeze(1)
        sc_q = torch.clamp(
            torch.round(sub_dynamic / torch.where(d_f32 > 0.0, d_f32, torch.ones_like(d_f32))), 0, 63
        ).to(torch.uint8)
        m_q = torch.clamp(
            torch.round(sub_mins / torch.where(dmin_f32 > 0.0, dmin_f32, torch.ones_like(dmin_f32))), 0, 63
        ).to(torch.uint8)

        eff_scale = (d_f32 * sc_q.to(torch.float32)).unsqueeze(2)
        eff_min = (dmin_f32 * m_q.to(torch.float32)).unsqueeze(2)
        denom = torch.where(eff_scale > 0.0, eff_scale, torch.ones_like(eff_scale))
        q_sub = torch.clamp(torch.round((sub_blocks + eff_min) / denom), 0, 15).to(torch.uint8)
        quant = q_sub.reshape(n_blocks, block_size)
        v0 = quant[:, 0::2]
        v1 = quant[:, 1::2]
        packed = (v0 & 0x0F) | ((v1 & 0x0F) << 4)

        meta = torch.empty((n_blocks, 12), dtype=torch.uint8)
        for j in range(4):
            meta[:, j] = (sc_q[:, j] & 63) | ((sc_q[:, j + 4] >> 4) << 6)
            meta[:, j + 4] = (m_q[:, j] & 63) | ((m_q[:, j + 4] >> 4) << 6)
            meta[:, j + 8] = (sc_q[:, j + 4] & 15) | ((m_q[:, j + 4] & 15) << 4)

        d_bytes = d.view(torch.uint8).reshape(n_blocks, 2)
        dmin_bytes = dmin.view(torch.uint8).reshape(n_blocks, 2)
        return _uint8_tensor_to_bytes(torch.cat([d_bytes, dmin_bytes, meta, packed], dim=1).reshape(-1))

    rem = len(flat) % block_size
    if rem != 0:
        flat.extend([0.0] * (block_size - rem))

    out = bytearray()
    for i in range(0, len(flat), block_size):
        chunk = flat[i : i + block_size]
        sub_s: list[float] = []
        sub_m: list[float] = []
        for b_idx in range(8):
            sub = chunk[b_idx * 32 : (b_idx + 1) * 32]
            m_val = max(0.0, -min(sub))
            s_val = max(x + m_val for x in sub) / 15.0
            sub_m.append(m_val)
            sub_s.append(s_val)

        max_s = max(sub_s)
        max_m = max(sub_m)
        d_val = max_s / 63.0 if max_s > 0.0 else 1.0
        dmin_val = max_m / 63.0 if max_m > 0.0 else 1.0
        d_bytes = struct.pack("<e", d_val)
        dmin_bytes = struct.pack("<e", dmin_val)
        d_f = struct.unpack("<e", d_bytes)[0]
        dmin_f = struct.unpack("<e", dmin_bytes)[0]

        sc_ints = [max(0, min(63, round(s / (d_f if d_f > 0.0 else 1.0)))) for s in sub_s]
        m_ints = [max(0, min(63, round(mv / (dmin_f if dmin_f > 0.0 else 1.0)))) for mv in sub_m]

        out.extend(d_bytes)
        out.extend(dmin_bytes)
        out.extend(_pack_q4_k_scales_mins(sc_ints, m_ints))

        q_vals: list[int] = []
        for b_idx in range(8):
            eff_s = d_f * sc_ints[b_idx]
            eff_m = dmin_f * m_ints[b_idx]
            denom_f = eff_s if eff_s > 0.0 else 1.0
            q_vals.extend(max(0, min(15, round((x + eff_m) / denom_f))) for x in chunk[b_idx * 32 : (b_idx + 1) * 32])

        for j in range(0, block_size, 2):
            out.append((q_vals[j] & 0x0F) | ((q_vals[j + 1] & 0x0F) << 4))
    return bytes(out)


def dequantize_tensor_q4_k_m(payload: bytes) -> list[float]:
    """Dequantize a GGML Q4_K_M byte stream back into a flat list of floats.

    Args:
        payload: Serialized Q4_K_M binary byte stream (multiple of 144 bytes).

    Returns:
        Reconstructed floating-point values.

    Raises:
        ValueError: If `len(payload)` is not a multiple of 144 bytes.

    """
    if len(payload) % 144 != 0:
        msg = f"Invalid Q4_K_M payload length {len(payload)}; must be a multiple of 144."
        raise ValueError(msg)

    reconstructed: list[float] = []
    for offset in range(0, len(payload), 144):
        d = float(struct.unpack("<e", payload[offset : offset + 2])[0])
        dmin = float(struct.unpack("<e", payload[offset + 2 : offset + 4])[0])
        meta = payload[offset + 4 : offset + 16]
        qs = payload[offset + 16 : offset + 144]
        sc, m = _unpack_q4_k_scales_mins(meta)

        nibbles: list[int] = []
        for b in qs:
            nibbles.extend((b & 15, b >> 4 & 15))

        for b_idx in range(8):
            eff_s = d * sc[b_idx]
            eff_m = dmin * m[b_idx]
            reconstructed.extend(eff_s * q - eff_m for q in nibbles[b_idx * 32 : (b_idx + 1) * 32])
    return reconstructed


def _load_state_dict_from_local_path(p: Path, model_or_name: str) -> dict[str, TensorInput] | None:
    """Attempt to load a PyTorch state dictionary from a local `.safetensors` file or HF checkpoint directory.

    Args:
        p: Existing filesystem path to a `.safetensors` file or checkpoint directory.
        model_or_name: Original model identifier string for diagnostic logging.

    Returns:
        Extracted dictionary of tensor names to `TensorInput`, or `None` if loading did not succeed.

    """
    from gemma_4_sql.backends.lazy_loader import import_optional_symbol

    st_file = p if p.is_file() and p.suffix == ".safetensors" else (p / "model.safetensors" if p.is_dir() else None)
    if st_file is not None and st_file.is_file():

        @runtime_checkable
        class _GGUFSafetensorsLoadProtocol(Protocol):
            """Protocol for `safetensors.torch.load_file(filename)`."""

            def __call__(self, filename: str, /) -> Mapping[str, TensorInput]:
                """Load tensor mapping from `filename`."""
                ...

        try:
            from safetensors import SafetensorError
        except ImportError as e:
            logger.debug("Safetensors import failed: %s", e)
        else:
            load_file = import_optional_symbol("safetensors.torch", "load_file", _GGUFSafetensorsLoadProtocol)
            if load_file is not None:
                try:
                    loaded_st = load_file(str(st_file))
                    return {str(k): v for k, v in loaded_st.items()}
                except (SafetensorError, ValueError, RuntimeError, OSError) as e:
                    logger.debug("Local safetensors load failed for '%s': %s", st_file, e)

    @runtime_checkable
    class _GGUFStateDictModelProtocol(Protocol):
        """Protocol for a causal LM exposing `.state_dict()`."""

        def state_dict(self) -> Mapping[str, TensorInput]:
            """Return parameter state dictionary."""
            ...

    @runtime_checkable
    class _GGUFAutoModelProtocol(Protocol):
        """Protocol for `AutoModelForCausalLM` exposing `.from_pretrained(...)`."""

        def from_pretrained(
            self,
            model_path: str,
            *,
            local_files_only: bool = False,
        ) -> _GGUFStateDictModelProtocol:
            """Load local pretrained model."""
            ...

    auto_cls = import_optional_symbol("transformers", "AutoModelForCausalLM", _GGUFAutoModelProtocol)
    if auto_cls is not None:
        try:
            mod = auto_cls.from_pretrained(str(p), local_files_only=True)
            state_dict_res = mod.state_dict()
            return {str(k): v for k, v in state_dict_res.items()}
        except (ValueError, RuntimeError, AttributeError, OSError) as e:
            logger.debug("Local AutoModelForCausalLM load failed for '%s': %s", model_or_name, e)
    return None


def extract_pytorch_state_dict(
    model_or_name: Mapping[str, TensorInput] | _StateDictModelProtocol | TorchCausalLMProtocol | str | int,
    *,
    allow_dummy_fallback: bool = False,
) -> dict[str, TensorInput]:
    """Extract named weight tensors from a PyTorch model, state dictionary, or checkpoint directory.

    Maps architecture layer names to canonical GGUF llama.cpp layer identifiers.

    Args:
        model_or_name: Model instance, state dictionary, or local checkpoint string identifier.
        allow_dummy_fallback: Whether to fall back to a compact native `Gemma4Config` when
            `model_or_name` is a preset name that is not found on disk (default `False`).

    Returns:
        Dictionary mapping canonical GGUF tensor names to tensor or array objects.

    Raises:
        ExportError: If weight extraction fails or architecture is unrecognized.

    """
    raw_dict: dict[str, TensorInput] = {}

    if isinstance(model_or_name, Mapping):
        raw_dict = dict(model_or_name)
    elif isinstance(model_or_name, _StateDictModelProtocol):
        state_dict_res = model_or_name.state_dict()
        raw_dict = {str(k): v for k, v in state_dict_res.items()}
    elif isinstance(model_or_name, str):
        loaded = False
        from gemma_4_sql.backends.pytorch.inference import MODEL_CACHE

        cached = MODEL_CACHE.get_model(model_or_name)
        if cached is not None and isinstance(cached[0], _StateDictModelProtocol):
            raw_dict = {str(k): v for k, v in cached[0].state_dict().items()}
            loaded = True

        p = Path(model_or_name)
        if not loaded and p.exists() and torch is not None:
            local_dict = _load_state_dict_from_local_path(p, model_or_name)
            if local_dict is not None:
                raw_dict = local_dict
                loaded = True

        if not loaded:
            if not allow_dummy_fallback:
                raise ExportError(
                    f"Failed to load model state dict from '{model_or_name}' and allow_dummy_fallback is False."
                )
            try:
                from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
                from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

                preset_cfg = Gemma4Config.from_preset(model_or_name)
                cfg = Gemma4Config(
                    vocab_size=min(256, preset_cfg.vocab_size),
                    hidden_size=min(64, preset_cfg.hidden_size),
                    intermediate_size=min(128, preset_cfg.intermediate_size),
                    num_hidden_layers=min(2, preset_cfg.num_hidden_layers),
                    num_attention_heads=min(4, preset_cfg.num_attention_heads),
                    num_key_value_heads=min(2, preset_cfg.num_key_value_heads),
                    head_dim=min(16, preset_cfg.head_dim),
                )
                native_mod = Gemma4ForCausalLM(cfg)
                raw_dict = dict(native_mod.state_dict())
            except (ImportError, ValueError, RuntimeError, AttributeError, OSError) as e:
                raise ExportError(f"Failed to extract state dict for model '{model_or_name}': {e}") from e

    canonical: dict[str, TensorInput] = {}
    for name, param in raw_dict.items():
        mapped_name = name
        if "embed_tokens" in name or "token_embd" in name:
            mapped_name = "token_embd.weight"
        elif "layers." in name:
            parts = name.split(".")
            try:
                layer_idx = parts[parts.index("layers") + 1]
                if "self_attn.q_proj" in name or "attn_q" in name:
                    mapped_name = f"blk.{layer_idx}.attn_q.weight"
                elif "self_attn.k_proj" in name or "attn_k" in name:
                    mapped_name = f"blk.{layer_idx}.attn_k.weight"
                elif "self_attn.v_proj" in name or "attn_v" in name:
                    mapped_name = f"blk.{layer_idx}.attn_v.weight"
                elif "self_attn.o_proj" in name or "attn_output" in name:
                    mapped_name = f"blk.{layer_idx}.attn_output.weight"
                elif "mlp.gate_proj" in name or "ffn_gate" in name:
                    mapped_name = f"blk.{layer_idx}.ffn_gate.weight"
                elif "mlp.up_proj" in name or "ffn_up" in name:
                    mapped_name = f"blk.{layer_idx}.ffn_up.weight"
                elif "mlp.down_proj" in name or "ffn_down" in name:
                    mapped_name = f"blk.{layer_idx}.ffn_down.weight"
                elif "input_layernorm" in name or "attn_norm" in name:
                    mapped_name = f"blk.{layer_idx}.attn_norm.weight"
                elif "post_attention_layernorm" in name or "ffn_norm" in name:
                    mapped_name = f"blk.{layer_idx}.ffn_norm.weight"
            except (ValueError, IndexError):
                pass
        elif name in {"model.norm.weight", "norm.weight"} or (name.startswith("norm.") and "weight" in name):
            mapped_name = "output_norm.weight"
        elif "lm_head" in name or (name.startswith("output.") and "weight" in name):
            mapped_name = "output.weight"
        canonical[mapped_name] = param

    if not canonical:
        raise ExportError("Extracted parameter dictionary is empty.")
    return canonical


def _write_gguf_metadata_value(f: io.BufferedWriter, v: GGUFMetadataValue) -> None:
    """Write a typed GGUF v3 metadata value tag and payload to `f`.

    Args:
        f: Open binary buffered writer.
        v: Metadata scalar, sequence, or path-like value.

    """
    if isinstance(v, str):
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        vb = v.encode("utf-8")
        f.write(struct.pack("<Q", len(vb)) + vb)
    elif isinstance(v, bool):
        f.write(struct.pack("<I", GGUF_TYPE_BOOL))
        f.write(struct.pack("<B", 1 if v else 0))
    elif isinstance(v, int):
        if -0x80000000 <= v <= 0x7FFFFFFF:
            f.write(struct.pack("<I", GGUF_TYPE_INT32))
            f.write(struct.pack("<i", v))
        elif 0 <= v <= 0xFFFFFFFF:
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", v))
        elif v > 0xFFFFFFFF:
            f.write(struct.pack("<I", GGUF_TYPE_UINT64))
            f.write(struct.pack("<Q", v))
        else:
            f.write(struct.pack("<I", GGUF_TYPE_INT64))
            f.write(struct.pack("<q", v))
    elif isinstance(v, float):
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<f", v))
    elif isinstance(v, (list, tuple)):
        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        f.write(struct.pack("<Q", len(v)))
        for item in v:
            ib = str(item).encode("utf-8")
            f.write(struct.pack("<Q", len(ib)) + ib)
    else:
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        sb = str(v).encode("utf-8")
        f.write(struct.pack("<Q", len(sb)) + sb)


def write_gguf_v3(
    file_path: str | Path,
    metadata: Mapping[str, GGUFMetadataValue],
    tensors: Mapping[str, TensorInput],
    out_type: str = "q4_k_m",
) -> None:
    """Serialize metadata and model tensors into a validated GGUF v3 binary file.

    Reverses multi-dimensional row-major PyTorch shapes into GGML little-endian
    column-major dimension order (`ne[0] = cols, ne[1] = rows, ...`).

    Args:
        file_path: Output file path for the `.gguf` file.
        metadata: Key-value metadata dictionary.
        tensors: Dictionary mapping tensor names to tensor arrays.
        out_type: Quantization format identifier (`'f32'`, `'f16'`, `'q8_0'`, `'q4_0'`, `'q4_k_m'`).

    Raises:
        ValueError: If `out_type` is unrecognized or arguments are invalid.

    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_type_lower = out_type.lower()
    ggml_type: int
    if out_type_lower in {"f32", "fp32"}:
        ggml_type = GGML_TYPE_F32
    elif out_type_lower in {"f16", "fp16"}:
        ggml_type = GGML_TYPE_F16
    elif out_type_lower == "q8_0":
        ggml_type = GGML_TYPE_Q8_0
    elif out_type_lower == "q4_0":
        ggml_type = GGML_TYPE_Q4_0
    elif out_type_lower in {"q4_k_m", "q4_k"}:
        ggml_type = GGML_TYPE_Q4_K
    else:
        raise ValueError(f"Unsupported GGUF quantization format: {out_type}")

    meta_dict: dict[str, GGUFMetadataValue] = dict(metadata)
    meta_dict.setdefault("general.architecture", "gemma4")
    meta_dict.setdefault("general.file_type", ggml_type)

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(meta_dict)))

        for k, v in meta_dict.items():
            kb = k.encode("utf-8")
            f.write(struct.pack("<Q", len(kb)) + kb)
            _write_gguf_metadata_value(f, v)

        raw_payloads: list[bytes] = []
        descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
        data_offset = 0

        for t_name, arr in tensors.items():
            flat_vals, row_major_shape = _to_contiguous_float32_1d(arr)
            ggml_shape = tuple(reversed(row_major_shape)) if len(row_major_shape) > 1 else row_major_shape

            payload: bytes
            if isinstance(arr, (bytes, bytearray)):
                payload = bytes(arr)
            elif ggml_type == GGML_TYPE_F32:
                payload = struct.pack(f"<{len(flat_vals)}f", *flat_vals)
            elif ggml_type == GGML_TYPE_F16:
                payload = quantize_tensor_f16(arr)
            elif ggml_type == GGML_TYPE_Q8_0:
                payload = quantize_tensor_q8_0(arr)
            elif ggml_type == GGML_TYPE_Q4_0:
                payload = quantize_tensor_q4_0(arr)
            else:
                payload = quantize_tensor_q4_k_m(arr)

            descriptors.append((t_name, ggml_shape, ggml_type, data_offset))
            raw_payloads.append(payload)

            data_offset += len(payload)
            rem = data_offset % GGUF_ALIGNMENT
            if rem != 0:
                data_offset += GGUF_ALIGNMENT - rem

        for t_name, ggml_shape, t_type, offset in descriptors:
            tb = t_name.encode("utf-8")
            f.write(struct.pack("<Q", len(tb)) + tb)
            f.write(struct.pack("<I", len(ggml_shape)))
            f.writelines(struct.pack("<Q", dim) for dim in ggml_shape)
            f.write(struct.pack("<I", t_type))
            f.write(struct.pack("<Q", offset))

        header_pos = f.tell()
        pad = (GGUF_ALIGNMENT - (header_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        f.write(b"\x00" * pad)

        for payload in raw_payloads:
            f.write(payload)
            payload_pad = (GGUF_ALIGNMENT - (len(payload) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
            f.write(b"\x00" * payload_pad)


def _write_gguf_file(
    file_path: Path,
    model_name: str,
    tensors: Mapping[str, TensorInput] | None = None,
    metadata: Mapping[str, GGUFMetadataValue] | None = None,
    out_type: str = "q4_0",
) -> None:
    """Serialize tensors and metadata into a valid GGUF v3 binary format.

    Args:
        file_path: Destination path for the `.gguf` file.
        model_name: Name or architecture identifier for model metadata.
        tensors: Optional dictionary mapping tensor names to arrays.
        metadata: Optional key-value metadata dictionary.
        out_type: Quantization format identifier (default: `'q4_0'`).

    """
    if tensors is None:
        tensor_data: TensorInput = torch.zeros((16, 16), dtype=torch.float32) if torch is not None else bytes(1024)
        tensors = {"token_embd.weight": tensor_data}

    meta: Mapping[str, GGUFMetadataValue] = metadata or {
        "general.architecture": "gemma4",
        "general.name": model_name,
    }
    write_gguf_v3(file_path, metadata=meta, tensors=tensors, out_type=out_type)


def _read_gguf_typed_value(
    f: io.BufferedReader, v_type: int
) -> GGUFMetadataScalar | list[GGUFMetadataScalar | GGUFMetadataArray]:
    """Read a single GGUF v3 metadata value of `v_type` using exact spec byte widths and signedness.

    Args:
        f: Open binary buffered reader positioned at the value payload.
        v_type: Integer GGUF metadata value type code (`0..12`).

    Returns:
        Decoded scalar or nested array metadata value.

    Raises:
        ValueError: If `v_type` is not a valid GGUF v3 metadata value type (`0..12`).

    """
    if v_type == GGUF_TYPE_UINT8:
        return int(struct.unpack("<B", f.read(1))[0])
    if v_type == GGUF_TYPE_INT8:
        return int(struct.unpack("<b", f.read(1))[0])
    if v_type == GGUF_TYPE_UINT16:
        return int(struct.unpack("<H", f.read(2))[0])
    if v_type == GGUF_TYPE_INT16:
        return int(struct.unpack("<h", f.read(2))[0])
    if v_type == GGUF_TYPE_UINT32:
        return int(struct.unpack("<I", f.read(4))[0])
    if v_type == GGUF_TYPE_INT32:
        return int(struct.unpack("<i", f.read(4))[0])
    if v_type == GGUF_TYPE_FLOAT32:
        return float(struct.unpack("<f", f.read(4))[0])
    if v_type == GGUF_TYPE_BOOL:
        return int(struct.unpack("<B", f.read(1))[0]) != 0
    if v_type == GGUF_TYPE_STRING:
        v_len = int(struct.unpack("<Q", f.read(8))[0])
        return f.read(v_len).decode("utf-8", errors="replace")
    if v_type == GGUF_TYPE_ARRAY:
        arr_type = int(struct.unpack("<I", f.read(4))[0])
        arr_len = int(struct.unpack("<Q", f.read(8))[0])
        return [_read_gguf_typed_value(f, arr_type) for _ in range(arr_len)]
    if v_type == GGUF_TYPE_UINT64:
        return int(struct.unpack("<Q", f.read(8))[0])
    if v_type == GGUF_TYPE_INT64:
        return int(struct.unpack("<q", f.read(8))[0])
    if v_type == GGUF_TYPE_FLOAT64:
        return float(struct.unpack("<d", f.read(8))[0])
    raise ValueError(f"Unsupported GGUF metadata value type: {v_type}")


def validate_gguf_file(file_path: str | Path) -> GGUFHeaderInfo:
    """Validate a GGUF binary file, inspecting magic bytes, header, and tensor descriptors.

    Args:
        file_path: Path to the GGUF file to inspect.

    Returns:
        A `GGUFHeaderInfo` dictionary containing parsed metadata, version, and tensor descriptors.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If magic bytes, metadata value types, or tensor offsets are corrupted.

    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found at {path}")

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Corrupt GGUF magic header: expected b'GGUF', got {magic!r}")

        version = int(struct.unpack("<I", f.read(4))[0])
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count = int(struct.unpack("<Q", f.read(8))[0])
        kv_count = int(struct.unpack("<Q", f.read(8))[0])

        metadata: dict[str, GGUFMetadataValue] = {}
        for _ in range(kv_count):
            k_len = int(struct.unpack("<Q", f.read(8))[0])
            k = f.read(k_len).decode("utf-8", errors="replace")
            v_type = int(struct.unpack("<I", f.read(4))[0])
            metadata[k] = _read_gguf_typed_value(f, v_type)

        tensors: list[GGUFTensorDescriptor] = []
        for _ in range(tensor_count):
            t_len = int(struct.unpack("<Q", f.read(8))[0])
            t_name = f.read(t_len).decode("utf-8", errors="replace")
            n_dims = int(struct.unpack("<I", f.read(4))[0])
            dims = [int(struct.unpack("<Q", f.read(8))[0]) for _ in range(n_dims)]
            t_type = int(struct.unpack("<I", f.read(4))[0])
            offset = int(struct.unpack("<Q", f.read(8))[0])
            if offset % GGUF_ALIGNMENT != 0:
                raise ValueError(f"Unaligned GGUF tensor offset {offset} for '{t_name}' (must be a multiple of 32)")
            tensors.append({"name": t_name, "shape": dims, "type": t_type, "offset": offset})

        header_end = f.tell()
        file_size = path.stat().st_size
        data_start = header_end + ((GGUF_ALIGNMENT - (header_end % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT)
        if file_size > header_end:
            for td in tensors:
                if data_start + td["offset"] > file_size:
                    raise ValueError(f"GGUF tensor '{td['name']}' offset {td['offset']} exceeds file size {file_size}")

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata": metadata,
        "tensors": tensors,
    }


__all__ = [
    "GGML_TYPE_F16",
    "GGML_TYPE_F32",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q4_K",
    "GGML_TYPE_Q8_0",
    "GGUF_ALIGNMENT",
    "GGUF_TYPE_ARRAY",
    "GGUF_TYPE_BOOL",
    "GGUF_TYPE_FLOAT32",
    "GGUF_TYPE_FLOAT64",
    "GGUF_TYPE_INT8",
    "GGUF_TYPE_INT16",
    "GGUF_TYPE_INT32",
    "GGUF_TYPE_INT64",
    "GGUF_TYPE_STRING",
    "GGUF_TYPE_UINT8",
    "GGUF_TYPE_UINT16",
    "GGUF_TYPE_UINT32",
    "GGUF_TYPE_UINT64",
    "GGUFHeaderInfo",
    "GGUFMetadataArray",
    "GGUFMetadataScalar",
    "GGUFMetadataValue",
    "GGUFTensorDescriptor",
    "NestedFloatSeq",
    "TensorInput",
    "_StateDictModelProtocol",
    "_flatten_values",
    "_to_contiguous_float32_1d",
    "_write_gguf_file",
    "dequantize_tensor_q4_k_m",
    "extract_pytorch_state_dict",
    "quantize_tensor_f16",
    "quantize_tensor_q4_0",
    "quantize_tensor_q4_k_m",
    "quantize_tensor_q8_0",
    "validate_gguf_file",
    "write_gguf_v3",
]
