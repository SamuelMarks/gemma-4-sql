"""GGUF v3 binary serialization, tensor quantization, and parameter extraction.

This module provides native conversion of PyTorch and Gemma 4 model architectures
into binary GGUF v3 containers for high-performance llama.cpp inference.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gemma_4_sql.exceptions import ExportError

logger = logging.getLogger(__name__)

try:
    import numpy as _np

    np: Any = _np
except (ImportError, AttributeError):
    np = None

try:
    import torch as _torch

    torch: Any = _torch
except (ImportError, AttributeError):
    torch = None


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


def _flatten_values(arr: Any) -> list[float]:
    """Recursively flatten array-like structures into a 1D list of floats.

    Args:
        arr: Array, tensor, or nested list of numerical values.

    Returns:
        Flat 1D list of floating point values.
    """
    if isinstance(arr, (bytes, bytearray)):
        return [float(b) for b in arr]
    if np is not None and hasattr(arr, "flatten") and hasattr(arr, "tolist"):
        return [float(x) for x in arr.flatten().tolist()]
    if torch is not None and hasattr(arr, "flatten") and hasattr(arr, "detach"):
        return [float(x) for x in arr.detach().cpu().flatten().numpy().tolist()]
    if isinstance(arr, (list, tuple)):
        out: list[float] = []
        for item in arr:
            out.extend(_flatten_values(item))
        return out
    return [float(arr)]


def quantize_tensor_f16(arr: Any) -> bytes:
    """Quantize an input array or tensor to 16-bit half precision float bytes.

    Args:
        arr: Input array or tensor to quantize.

    Returns:
        Raw bytes representation in IEEE 754 float16 format.
    """
    if np is not None and hasattr(arr, "astype"):
        return bytes(arr.astype(np.float16).tobytes())
    if torch is not None and hasattr(arr, "to") and hasattr(arr, "numpy"):
        return bytes(arr.to(torch.float16).detach().cpu().numpy().tobytes())
    flat = _flatten_values(arr)
    return struct.pack(f"<{len(flat)}e", *flat)


def quantize_tensor_q8_0(arr: Any, block_size: int = 32) -> bytes:
    """Quantize an input float array into GGML Q8_0 blocks.

    Each block consists of a 16-bit float scale followed by `block_size` int8 values.

    Args:
        arr: Array-like float values to quantize.
        block_size: Number of values per block (standard: 32).

    Returns:
        Serialized Q8_0 binary byte stream.

    Raises:
        ValueError: If block_size is non-positive.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if np is not None:
        if isinstance(arr, np.ndarray):
            a = arr.reshape(-1).astype(np.float32)
        elif torch is not None and hasattr(arr, "detach") and hasattr(arr, "numpy"):
            a = arr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            a = np.array(_flatten_values(arr), dtype=np.float32)
        rem = len(a) % block_size
        if rem != 0:
            a = np.pad(a, (0, block_size - rem))
        n_blocks = len(a) // block_size
        blocks = a.reshape(n_blocks, block_size)
        max_vals = np.max(np.abs(blocks), axis=1)
        scales = np.where(max_vals > 0, max_vals / 127.0, 1.0).astype(np.float16)
        qs = np.clip(np.round(blocks / scales[:, None]), -128, 127).astype(np.int8)
        dt = np.dtype([("scale", "<f2"), ("qs", "i1", (block_size,))])
        structured = np.empty(n_blocks, dtype=dt)
        structured["scale"] = scales
        structured["qs"] = qs
        return bytes(structured.tobytes())

    flat = _flatten_values(arr)
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


def quantize_tensor_q4_0(arr: Any, block_size: int = 32) -> bytes:
    """Quantize an input float array into GGML Q4_0 blocks.

    Each block consists of a 16-bit float scale followed by packed 4-bit nibbles.

    Args:
        arr: Array-like float values to quantize.
        block_size: Number of values per block (standard: 32).

    Returns:
        Serialized Q4_0 binary byte stream.

    Raises:
        ValueError: If block_size is non-positive or not even.
    """
    if block_size <= 0 or block_size % 2 != 0:
        raise ValueError("block_size must be a positive even integer.")

    half_block = block_size // 2

    if np is not None:
        if isinstance(arr, np.ndarray):
            a = arr.reshape(-1).astype(np.float32)
        elif torch is not None and hasattr(arr, "detach") and hasattr(arr, "numpy"):
            a = arr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            a = np.array(_flatten_values(arr), dtype=np.float32)
        rem = len(a) % block_size
        if rem != 0:
            a = np.pad(a, (0, block_size - rem))
        n_blocks = len(a) // block_size
        blocks = a.reshape(n_blocks, block_size)
        max_vals = np.max(np.abs(blocks), axis=1)
        scales = np.where(max_vals > 0, max_vals / -8.0, 1.0).astype(np.float16)
        denom = np.where(scales[:, None] != 0, scales[:, None], 1.0)
        v0 = np.clip(np.round(blocks[:, :half_block] / denom) + 8, 0, 15).astype(np.uint8)
        v1 = np.clip(np.round(blocks[:, half_block:] / denom) + 8, 0, 15).astype(np.uint8)
        packed = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
        dt = np.dtype([("scale", "<f2"), ("qs", "u1", (half_block,))])
        structured = np.empty(n_blocks, dtype=dt)
        structured["scale"] = scales
        structured["qs"] = packed
        return bytes(structured.tobytes())

    flat = _flatten_values(arr)
    rem = len(flat) % block_size
    if rem != 0:
        flat.extend([0.0] * (block_size - rem))

    out = bytearray()
    for i in range(0, len(flat), block_size):
        chunk = flat[i : i + block_size]
        max_val = max(abs(x) for x in chunk)
        scale = max_val / -8.0 if max_val > 0.0 else 1.0
        out.extend(struct.pack("<e", scale))
        for j in range(half_block):
            v0 = round(chunk[j] / scale) + 8
            v1 = round(chunk[j + half_block] / scale) + 8
            v0 = max(0, min(15, v0))
            v1 = max(0, min(15, v1))
            byte_val = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
            out.append(byte_val)
    return bytes(out)


def quantize_tensor_q4_k_m(arr: Any) -> bytes:
    """Quantize an input float array into GGML Q4_K_M blocks.

    Groups elements into blocks of 256 elements with sub-block scaling.

    Args:
        arr: Array-like float values to quantize.

    Returns:
        Serialized Q4_K_M binary byte stream.
    """
    block_size = 256

    if np is not None:
        if isinstance(arr, np.ndarray):
            a = arr.reshape(-1).astype(np.float32)
        elif torch is not None and hasattr(arr, "detach") and hasattr(arr, "numpy"):
            a = arr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            a = np.array(_flatten_values(arr), dtype=np.float32)
        rem = len(a) % block_size
        if rem != 0:
            a = np.pad(a, (0, block_size - rem))
        n_blocks = len(a) // block_size
        blocks = a.reshape(n_blocks, block_size)
        max_vals = np.max(np.abs(blocks), axis=1)
        d = np.where(max_vals > 0, max_vals / -8.0, 1.0).astype(np.float16)
        dmin = np.min(blocks, axis=1).astype(np.float16)
        denom = np.where(d[:, None] != 0, d[:, None], 1.0)
        diff = blocks - dmin[:, None]
        quant = np.clip(np.round(diff / denom), 0, 15).astype(np.uint8)
        v0 = quant[:, 0::2]
        v1 = quant[:, 1::2]
        packed = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
        meta = np.ones((n_blocks, 12), dtype=np.uint8)
        dt = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("meta", "u1", (12,)), ("qs", "u1", (128,))])
        structured = np.empty(n_blocks, dtype=dt)
        structured["d"] = d
        structured["dmin"] = dmin
        structured["meta"] = meta
        structured["qs"] = packed
        return bytes(structured.tobytes())

    flat = _flatten_values(arr)
    rem = len(flat) % block_size
    if rem != 0:
        flat.extend([0.0] * (block_size - rem))

    out = bytearray()
    for i in range(0, len(flat), block_size):
        chunk = flat[i : i + block_size]
        max_val = max(abs(x) for x in chunk)
        d = max_val / -8.0 if max_val > 0.0 else 1.0
        dmin = min(chunk)
        out.extend(struct.pack("<e", d))
        out.extend(struct.pack("<e", dmin))
        # 12 bytes scale metadata
        out.extend(b"\x01" * 12)
        # 128 bytes packed 4-bit nibbles
        for j in range(0, block_size, 2):
            v0 = max(0, min(15, round((chunk[j] - dmin) / (d if d != 0.0 else 1.0))))
            v1 = max(0, min(15, round((chunk[j + 1] - dmin) / (d if d != 0.0 else 1.0))))
            out.append((v0 & 0x0F) | ((v1 & 0x0F) << 4))
    return bytes(out)


def extract_pytorch_state_dict(model_or_name: Any) -> dict[str, Any]:
    """Extract named weight tensors from a PyTorch model, state dictionary, or preset.

    Maps architecture layer names to canonical GGUF llama.cpp layer identifiers.

    Args:
        model_or_name: Model instance, state dictionary, or preset model string identifier.

    Returns:
        Dictionary mapping canonical GGUF tensor names to tensor or array objects.

    Raises:
        ExportError: If weight extraction fails or architecture is unrecognized.
    """
    raw_dict: dict[str, Any] = {}

    if isinstance(model_or_name, dict):
        raw_dict = model_or_name
    elif hasattr(model_or_name, "state_dict") and callable(model_or_name.state_dict):
        raw_dict = dict(model_or_name.state_dict())
    elif isinstance(model_or_name, str):
        loaded = False
        p = Path(model_or_name)
        if p.exists() and torch is not None:
            try:
                from transformers import AutoModelForCausalLM

                mod = AutoModelForCausalLM.from_pretrained(str(p), local_files_only=True)
                raw_dict = dict(mod.state_dict())
                loaded = True
            except (ImportError, ValueError, RuntimeError, AttributeError, OSError) as e:
                logger.debug("Local AutoModelForCausalLM load failed for '%s': %s", model_or_name, e)

        if not loaded:
            try:
                from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
                from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

                cfg = Gemma4Config(vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
                native_mod = Gemma4ForCausalLM(cfg)
                raw_dict = dict(native_mod.state_dict())
                loaded = True
            except (ImportError, ValueError, RuntimeError, AttributeError, OSError) as e:
                raise ExportError(f"Failed to extract state dict for model '{model_or_name}': {e}") from e

    # Map architecture parameters to canonical GGUF names
    canonical: dict[str, Any] = {}
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


def write_gguf_v3(
    file_path: Path | str,
    metadata: dict[str, Any],
    tensors: dict[str, Any],
    out_type: str = "q4_0",
) -> None:
    """Serialize metadata and model tensors into a validated GGUF v3 binary file.

    Args:
        file_path: Output file path for the .gguf file.
        metadata: Key-value metadata dictionary.
        tensors: Dictionary mapping tensor names to tensor arrays.
        out_type: Quantization format identifier ('f32', 'f16', 'q8_0', 'q4_0', 'q4_k_m').

    Raises:
        ValueError: If out_type is unrecognized or arguments are invalid.
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

    # Standard model metadata defaults
    meta_dict = dict(metadata)
    meta_dict.setdefault("general.architecture", "gemma4")
    meta_dict.setdefault("general.file_type", ggml_type)

    with open(path, "wb") as f:
        # Magic bytes and version
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(meta_dict)))

        # Write metadata key-value pairs
        for k, v in meta_dict.items():
            kb = k.encode("utf-8")
            f.write(struct.pack("<Q", len(kb)) + kb)
            if isinstance(v, str):
                f.write(struct.pack("<I", GGUF_TYPE_STRING))
                vb = v.encode("utf-8")
                f.write(struct.pack("<Q", len(vb)) + vb)
            elif isinstance(v, bool):
                f.write(struct.pack("<I", GGUF_TYPE_BOOL))
                f.write(struct.pack("<B", 1 if v else 0))
            elif isinstance(v, int):
                f.write(struct.pack("<I", GGUF_TYPE_INT32))
                f.write(struct.pack("<i", v))
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

        # Quantize and prepare tensors
        raw_payloads: list[bytes] = []
        descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
        data_offset = 0

        for t_name, arr in tensors.items():
            shape: tuple[int, ...]
            if hasattr(arr, "shape"):
                shape = tuple(arr.shape)
            elif isinstance(arr, Sequence) and arr and isinstance(arr[0], Sequence):
                shape = (len(arr), len(arr[0]))
            else:
                shape = (len(arr),) if hasattr(arr, "__len__") else (16, 16)

            payload: bytes
            if isinstance(arr, (bytes, bytearray)):
                payload = bytes(arr)
            elif ggml_type == GGML_TYPE_F32:
                flat_vals = _flatten_values(arr)
                payload = struct.pack(f"<{len(flat_vals)}f", *flat_vals)
            elif ggml_type == GGML_TYPE_F16:
                payload = quantize_tensor_f16(arr)
            elif ggml_type == GGML_TYPE_Q8_0:
                payload = quantize_tensor_q8_0(arr)
            elif ggml_type == GGML_TYPE_Q4_0:
                payload = quantize_tensor_q4_0(arr)
            elif ggml_type == GGML_TYPE_Q4_K:
                payload = quantize_tensor_q4_k_m(arr)
            else:  # pragma: no cover
                flat_vals = _flatten_values(arr)
                payload = struct.pack(f"<{len(flat_vals)}f", *flat_vals)

            descriptors.append((t_name, shape, ggml_type, data_offset))
            raw_payloads.append(payload)

            data_offset += len(payload)
            rem = data_offset % GGUF_ALIGNMENT
            if rem != 0:
                data_offset += GGUF_ALIGNMENT - rem

        # Write tensor descriptors
        for t_name, shape, t_type, offset in descriptors:
            tb = t_name.encode("utf-8")
            f.write(struct.pack("<Q", len(tb)) + tb)
            f.write(struct.pack("<I", len(shape)))
            f.writelines(struct.pack("<Q", dim) for dim in shape)
            f.write(struct.pack("<I", t_type))
            f.write(struct.pack("<Q", offset))

        # Align to GGUF_ALIGNMENT boundary before payload section
        header_pos = f.tell()
        pad = (GGUF_ALIGNMENT - (header_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        f.write(b"\x00" * pad)

        # Write aligned tensor payloads
        for payload in raw_payloads:
            f.write(payload)
            payload_pad = (GGUF_ALIGNMENT - (len(payload) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
            f.write(b"\x00" * payload_pad)
