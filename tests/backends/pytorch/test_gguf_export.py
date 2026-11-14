"""Comprehensive tests for PyTorch GGUF v3 quantization, reversed GGML shapes, unsigned metadata, and export."""

from __future__ import annotations

from tests.conftest import requires_torch

pytestmark = requires_torch

import math
import struct
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from gemma_4_sql.backends.pytorch.gguf import (
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_FLOAT64,
    GGUF_TYPE_INT8,
    GGUF_TYPE_INT16,
    GGUF_TYPE_UINT8,
    GGUF_TYPE_UINT16,
    GGUF_TYPE_UINT32,
    GGUFMetadataValue,
    TensorInput,
    _flatten_values,
    _to_contiguous_float32_1d,
    dequantize_tensor_q4_k_m,
    extract_pytorch_state_dict,
    quantize_tensor_f16,
    quantize_tensor_q4_0,
    quantize_tensor_q4_k_m,
    quantize_tensor_q8_0,
    write_gguf_v3,
)
from gemma_4_sql.backends.pytorch.inference import _MODEL_CACHE
from gemma_4_sql.backends.pytorch.quantize import _export_gguf, validate_gguf_file
from gemma_4_sql.exceptions import ExportError, QuantizationError
from tests.utils.tensor_helpers import compute_snr
from tests.utils.tensor_helpers import torch_save_safetensors as save_file


def test_to_contiguous_float32_1d_exact_shapes_1d_2d_3d_scalar() -> None:
    """Verify _to_contiguous_float32_1d preserves 1D/2D/3D+ shapes and scalar (1,) without fabricating (16, 16)."""
    vals_scalar, shape_scalar = _to_contiguous_float32_1d(42.5)
    assert vals_scalar == [42.5]
    assert shape_scalar == (1,)

    vals_empty, shape_empty = _to_contiguous_float32_1d([])
    assert vals_empty == []
    assert shape_empty == (0,)

    vals_3d, shape_3d = _to_contiguous_float32_1d([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ])
    assert shape_3d == (2, 2, 3)
    assert len(vals_3d) == 12

    t0 = torch.tensor(3.5)
    vals_t0, shape_t0 = _to_contiguous_float32_1d(t0)
    assert vals_t0 == [3.5]
    assert shape_t0 == (1,)

    class _ToListWithoutFlatten:
        """Minimal `ToListArrayProtocol` without `.flatten()` to exercise `ToListArrayProtocol` conversion."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Shape tuple `(2,)`."""
            return (2,)

        def tolist(self) -> list[int]:
            """Return `[1, 2]`."""
            return [1, 2]

    inst_tl = _ToListWithoutFlatten()
    assert inst_tl.shape == (2,)
    vals_tl, shape_tl = _to_contiguous_float32_1d(inst_tl)
    assert vals_tl == [1.0, 2.0]
    assert shape_tl == (2,)


def test_ggml_reversed_shapes_and_all_gguf_v3_metadata_types(tmp_path: Path) -> None:
    """Verify GGUF v3 reverses shapes into GGML (ne[0]=cols, ne[1]=rows) and parses all metadata types."""
    gguf_file = tmp_path / "spec_roundtrip.gguf"
    w_2d = torch.zeros((4, 32), dtype=torch.float32)
    w_3d = torch.zeros((2, 4, 8), dtype=torch.float32)
    metadata: dict[str, GGUFMetadataValue] = {
        "uint32_large": 3_000_000_000,
        "uint64_large": 9_000_000_000_000_000_000,
        "int64_neg": -5_000_000_000,
    }
    write_gguf_v3(gguf_file, metadata=metadata, tensors={"w2": w_2d, "w3": w_3d}, out_type="f32")
    info = validate_gguf_file(gguf_file)
    by_name = {td["name"]: td["shape"] for td in info["tensors"]}
    assert by_name["w2"] == [32, 4]
    assert by_name["w3"] == [8, 4, 2]
    assert info["metadata"]["uint32_large"] == 3_000_000_000
    assert info["metadata"]["uint64_large"] == 9_000_000_000_000_000_000
    assert info["metadata"]["int64_neg"] == -5_000_000_000

    # Also test binary header with UINT8, INT8, UINT16, INT16, UINT32, UINT64, INT64, FLOAT64, and nested ARRAY
    manual_gguf = tmp_path / "all_meta_types.gguf"
    with open(manual_gguf, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 7))
        f.write(struct.pack("<Q", 2) + b"u8" + struct.pack("<I", GGUF_TYPE_UINT8) + struct.pack("<B", 250))
        f.write(struct.pack("<Q", 2) + b"i8" + struct.pack("<I", GGUF_TYPE_INT8) + struct.pack("<b", -120))
        f.write(struct.pack("<Q", 3) + b"u16" + struct.pack("<I", GGUF_TYPE_UINT16) + struct.pack("<H", 60000))
        f.write(struct.pack("<Q", 3) + b"i16" + struct.pack("<I", GGUF_TYPE_INT16) + struct.pack("<h", -30000))
        f.write(struct.pack("<Q", 3) + b"u32" + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", 4000000000))
        f.write(struct.pack("<Q", 3) + b"f64" + struct.pack("<I", GGUF_TYPE_FLOAT64) + struct.pack("<d", 1.25))
        f.write(
            struct.pack("<Q", 3)
            + b"arr"
            + struct.pack("<I", GGUF_TYPE_ARRAY)
            + struct.pack("<I", GGUF_TYPE_UINT16)
            + struct.pack("<Q", 2)
            + struct.pack("<HH", 10, 20)
        )
    parsed = validate_gguf_file(manual_gguf)
    assert parsed["metadata"]["u8"] == 250
    assert parsed["metadata"]["i8"] == -120
    assert parsed["metadata"]["u16"] == 60000
    assert parsed["metadata"]["i16"] == -30000
    assert parsed["metadata"]["u32"] == 4000000000
    assert parsed["metadata"]["f64"] == 1.25
    assert parsed["metadata"]["arr"] == [10, 20]

    bad_meta = tmp_path / "bad_meta.gguf"
    with open(bad_meta, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 3) + b"bad" + struct.pack("<I", 99))
    with pytest.raises(ValueError, match="Unsupported GGUF metadata value type: 99"):
        validate_gguf_file(bad_meta)


def test_flatten_values_torch() -> None:
    """Test _flatten_values with a torch tensor."""
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert _flatten_values(t) == [1.0, 2.0, 3.0, 4.0]


def test_quantize_tensor_f16() -> None:
    """Test F16 quantization."""
    data = [1.0, -2.5, 0.0, math.pi]
    res = quantize_tensor_f16(data)
    assert len(res) == len(data) * 2
    unpacked = struct.unpack(f"<{len(data)}e", res)
    assert abs(unpacked[0] - 1.0) < 1e-3
    assert abs(unpacked[1] - (-2.5)) < 1e-3


def test_quantize_tensor_q8_0() -> None:
    """Test Q8_0 quantization format and block structure."""
    data = [float(i) for i in range(64)]
    res = quantize_tensor_q8_0(data, block_size=32)
    assert len(res) == 68
    with pytest.raises(ValueError, match="block_size must be positive"):
        quantize_tensor_q8_0(data, block_size=0)


def test_quantize_tensor_q4_0() -> None:
    """Test Q4_0 quantization format and packed nibbles."""
    data = [float(i) for i in range(64)]
    res = quantize_tensor_q4_0(data, block_size=32)
    assert len(res) == 36
    with pytest.raises(ValueError, match="positive even integer"):
        quantize_tensor_q4_0(data, block_size=31)
    with pytest.raises(ValueError, match="positive even integer"):
        quantize_tensor_q4_0(data, block_size=-4)


def test_quantize_tensor_q4_k_m() -> None:
    """Test Q4_K_M quantization format and dequantization reconstruction accuracy."""
    data = [math.sin(i * 0.1) * 5.0 - 1.2 for i in range(256)]
    res = quantize_tensor_q4_k_m(data)
    assert len(res) == 144
    assert res[4:16] != b"\x00" * 12

    recon = dequantize_tensor_q4_k_m(res)
    assert len(recon) == 256
    assert compute_snr(data, recon) > 15.0

    with pytest.raises(ValueError, match="multiple of 144"):
        dequantize_tensor_q4_k_m(b"\x00" * 10)


def test_extract_pytorch_state_dict_from_dict() -> None:
    """Test extracting state dict from a dictionary with canonical mapping."""
    raw = {
        "model.embed_tokens.weight": [1.0, 2.0],
        "model.layers.0.self_attn.q_proj.weight": [[1.0]],
        "model.layers.0.self_attn.k_proj.weight": [[1.0]],
        "model.layers.0.self_attn.v_proj.weight": [[1.0]],
        "model.layers.0.self_attn.o_proj.weight": [[1.0]],
        "model.layers.0.mlp.gate_proj.weight": [[1.0]],
        "model.layers.0.mlp.up_proj.weight": [[1.0]],
        "model.layers.0.mlp.down_proj.weight": [[1.0]],
        "model.layers.0.input_layernorm.weight": [1.0],
        "model.layers.0.post_attention_layernorm.weight": [1.0],
        "model.norm.weight": [1.0],
        "lm_head.weight": [1.0],
    }
    extracted = extract_pytorch_state_dict(raw)
    assert "token_embd.weight" in extracted
    assert "blk.0.attn_q.weight" in extracted
    assert "blk.0.attn_k.weight" in extracted
    assert "blk.0.attn_v.weight" in extracted
    assert "blk.0.attn_output.weight" in extracted
    assert "blk.0.ffn_gate.weight" in extracted
    assert "blk.0.ffn_up.weight" in extracted
    assert "blk.0.ffn_down.weight" in extracted
    assert "blk.0.attn_norm.weight" in extracted
    assert "blk.0.ffn_norm.weight" in extracted
    assert "output_norm.weight" in extracted
    assert "output.weight" in extracted


def test_extract_pytorch_state_dict_from_model_and_cache_and_safetensors(tmp_path: Path) -> None:
    """Test extracting state dict from nn.Module, _MODEL_CACHE, and local safetensors."""

    class DummyModel(nn.Module):
        """Test stub for `DummyModel`."""

        def __init__(self) -> None:
            """Initialize test stub instance."""
            super().__init__()
            self.model = nn.ModuleDict({"embed_tokens": nn.Embedding(2, 4)})

    real_model = DummyModel()
    extracted = extract_pytorch_state_dict(real_model)
    assert "token_embd.weight" in extracted

    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    tiny_cached = Gemma4ForCausalLM(Gemma4Config.gemma4_tiny())
    _MODEL_CACHE.put_model("cached-gguf-test", (tiny_cached, None))
    extracted_cached = extract_pytorch_state_dict("cached-gguf-test")
    assert "token_embd.weight" in extracted_cached
    _MODEL_CACHE.pop("cached-gguf-test", None)

    st_dir = tmp_path / "st_ckpt"
    st_dir.mkdir()
    save_file({"model.embed_tokens.weight": torch.ones((2, 4), dtype=torch.float32)}, str(st_dir / "model.safetensors"))
    extracted_st = extract_pytorch_state_dict(str(st_dir))
    assert "token_embd.weight" in extracted_st

    from unittest.mock import patch

    with (
        patch.dict(sys.modules, {"safetensors": None}),
        pytest.raises(ExportError, match="allow_dummy_fallback is False"),
    ):
        extract_pytorch_state_dict(str(st_dir))

    with (
        patch.dict(sys.modules, {"safetensors.torch": None, "transformers": None}),
        pytest.raises(ExportError, match="allow_dummy_fallback is False"),
    ):
        extract_pytorch_state_dict(str(st_dir))

    corrupt_st = tmp_path / "corrupt.safetensors"
    corrupt_st.write_bytes(b"corrupt")
    with pytest.raises(ExportError, match="allow_dummy_fallback is False"):
        extract_pytorch_state_dict(str(corrupt_st))


def test_extract_pytorch_state_dict_empty_and_string_preset() -> None:
    """Test extract_pytorch_state_dict raises ExportError on empty dict or missing model without dummy fallback."""
    with pytest.raises(ExportError, match="Extracted parameter dictionary is empty"):
        extract_pytorch_state_dict({})

    with pytest.raises(ExportError, match="allow_dummy_fallback is False"):
        extract_pytorch_state_dict("gemma4-preset")

    extracted = extract_pytorch_state_dict("gemma4-preset", allow_dummy_fallback=True)
    assert "token_embd.weight" in extracted

    extracted_tiny = extract_pytorch_state_dict("gemma-4-tiny", allow_dummy_fallback=True)
    assert "token_embd.weight" in extracted_tiny


def test_write_and_validate_gguf_v3(tmp_path: Path) -> None:
    """Test writing and validating a complete GGUF v3 file with multiple tensor types."""
    gguf_file = tmp_path / "model.gguf"
    metadata: dict[str, GGUFMetadataValue] = {
        "general.architecture": "gemma4",
        "general.name": "test-model",
        "model.context_length": 4096,
        "model.is_fine_tuned": True,
        "model.learning_rate": 0.0001,
        "tokenizer.tokens": ["<pad>", "<eos>", "SELECT"],
    }
    tensors: dict[str, TensorInput] = {
        "token_embd.weight": [[0.1, 0.2], [0.3, 0.4]],
        "blk.0.attn_q.weight": [float(i) for i in range(32)],
    }

    write_gguf_v3(gguf_file, metadata=metadata, tensors=tensors, out_type="q4_0")
    assert gguf_file.exists()

    info = validate_gguf_file(gguf_file)
    assert info["version"] == 3
    assert info["tensor_count"] == 2
    assert info["metadata"]["general.architecture"] == "gemma4"
    assert info["metadata"]["general.name"] == "test-model"
    assert info["metadata"]["model.context_length"] == 4096
    assert info["metadata"]["model.is_fine_tuned"] is True
    lr_val = info["metadata"]["model.learning_rate"]
    assert isinstance(lr_val, float)
    assert abs(lr_val - 0.0001) < 1e-4

    tensor_names = [t["name"] for t in info["tensors"]]
    assert "token_embd.weight" in tensor_names
    assert "blk.0.attn_q.weight" in tensor_names


def test_write_gguf_unsupported_out_type_and_corrupt_validation(tmp_path: Path) -> None:
    """Test write_gguf_v3 and validate_gguf_file error checking."""
    with pytest.raises(ValueError, match="Unsupported GGUF quantization format"):
        write_gguf_v3(tmp_path / "bad.gguf", {}, {}, out_type="invalid_type")

    with pytest.raises(FileNotFoundError):
        validate_gguf_file(tmp_path / "missing.gguf")

    corrupt_magic = tmp_path / "corrupt_magic.gguf"
    corrupt_magic.write_bytes(b"BAD!12345678")
    with pytest.raises(ValueError, match="Corrupt GGUF magic header"):
        validate_gguf_file(corrupt_magic)

    corrupt_ver = tmp_path / "corrupt_ver.gguf"
    corrupt_ver.write_bytes(b"GGUF" + struct.pack("<I", 99))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        validate_gguf_file(corrupt_ver)

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(b"GGUF\x00")
    with pytest.raises(struct.error):
        validate_gguf_file(truncated)


def test_export_gguf_strict_and_dummy_fallback(tmp_path: Path) -> None:
    """Test _export_gguf rejects missing models by default and exports when allow_dummy_fallback=True."""
    with pytest.raises(QuantizationError, match="allow_dummy_fallback is False"):
        _export_gguf("nonexistent-gemma-preset", export_path=str(tmp_path), out_type="q4_0")

    reduction, status = _export_gguf(
        "test-gemma",
        export_path=str(tmp_path),
        out_type="q4_0",
        allow_dummy_fallback=True,
    )
    assert status == "quantized_gguf"
    assert reduction == pytest.approx(0.6)
    exported_file = tmp_path / "test-gemma.gguf"
    assert exported_file.exists()
    assert validate_gguf_file(exported_file)["version"] == 3

    target_file = tmp_path / "custom_output.gguf"
    _, status2 = _export_gguf(
        "test-gemma",
        export_path=str(target_file),
        out_type="q8_0",
        allow_dummy_fallback=True,
    )
    assert status2 == "quantized_gguf"
    assert target_file.exists()


def test_flatten_values_edge_cases() -> None:
    """Test _flatten_values on bytes, bytearrays, torch tensors, and custom flattenable protocol."""
    import gemma_4_sql.backends.pytorch.gguf as gguf_mod

    class _CustomFlattenable:
        """Minimal implementation of `_FlattenableArrayProtocol`."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Provide array shape."""
            return (2, 2)

        def flatten(self) -> _CustomFlattenable:
            """Return flattened self."""
            return self

        def tolist(self) -> list[float]:
            """Return flat float list."""
            return [1.0, 2.0, 3.0, 4.0]

    assert _flatten_values(b"\x01\x02") == [1.0, 2.0]
    assert _flatten_values(bytearray([3, 4])) == [3.0, 4.0]
    assert _flatten_values(42.5) == [42.5]
    assert _flatten_values(torch.tensor([[1.0, 2.0], [3.0, 4.0]])) == [1.0, 2.0, 3.0, 4.0]
    assert gguf_mod._flatten_values(_CustomFlattenable()) == [1.0, 2.0, 3.0, 4.0]


def test_quantize_fallbacks_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize fallback implementations when torch is None."""
    import gemma_4_sql.backends.pytorch.gguf as gguf_mod

    monkeypatch.setattr(gguf_mod, "torch", None)
    assert len(quantize_tensor_f16([1.0, 2.0, -1.0, 0.5])) == 8
    data = [float(i) for i in range(35)]
    assert len(gguf_mod.quantize_tensor_q8_0(data, block_size=32)) == 68
    assert len(gguf_mod.quantize_tensor_q8_0([1.0] * 32, block_size=32)) == 34
    assert len(gguf_mod.quantize_tensor_q4_0(data, block_size=32)) == 36
    assert len(gguf_mod.quantize_tensor_q4_0([1.0] * 32, block_size=32)) == 18
    data_k = [float(i % 5) for i in range(260)]
    assert len(gguf_mod.quantize_tensor_q4_k_m(data_k)) == 288
    assert len(gguf_mod.quantize_tensor_q4_k_m([1.0] * 256)) == 144


def test_write_gguf_all_quant_types_and_metadata_payloads(tmp_path: Path) -> None:
    """Test writing GGUF files with F32, F16, Q8_0, Q4_0, Q4_K_M, bytes payloads, and custom metadata."""
    metadata: dict[str, GGUFMetadataValue] = {
        "float_val": math.pi,
        "list_val": ["a", "b", "c"],
        "tuple_val": (1, 2),
        "int_val": 42,
        "bool_val": False,
        "str_val": "hello",
        "custom_obj": Path("custom/path"),
    }
    tensors: dict[str, TensorInput] = {
        "raw_tensor": b"\x00" * 32,
        "2d_tensor": [[1.0] * 32] * 8,
        "1d_tensor": [0.5] * 256,
    }
    for fmt in ("f32", "f16", "q8_0", "q4_0", "q4_k_m"):
        out_f = tmp_path / f"model_{fmt}.gguf"
        write_gguf_v3(out_f, metadata=metadata, tensors=tensors, out_type=fmt)
        assert validate_gguf_file(out_f)["tensor_count"] == 3

    f16_scalar = tmp_path / "scalar_f16.gguf"
    write_gguf_v3(f16_scalar, metadata={"k": "v"}, tensors={"scalar_tensor": 42.0}, out_type="f16")
    info_scalar = validate_gguf_file(f16_scalar)
    assert info_scalar["tensors"][0]["shape"] == [1]


def test_extract_state_dict_local_hf_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extract_pytorch_state_dict local HF directory loading, unrecognized types, and fallback failures."""
    local_dir = tmp_path / "local_hf_model"
    local_dir.mkdir()

    class HFModel(nn.Module):
        """Test stub for `HFModel`."""

        def __init__(self) -> None:
            """Initialize test stub instance."""
            super().__init__()
            self.embed_tokens = nn.Embedding(2, 4)
            self.layers = nn.ModuleDict({"notanint": nn.Linear(4, 4)})

    class _FakeAutoLoader:
        """Typed `AutoModelForCausalLM` double."""

        @staticmethod
        def from_pretrained(_path: str, *, local_files_only: bool = False) -> HFModel:
            """Return `HFModel`."""
            _ = local_files_only
            return HFModel()

    monkeypatch.setattr("transformers.AutoModelForCausalLM", _FakeAutoLoader)
    res = extract_pytorch_state_dict(str(local_dir))
    assert "token_embd.weight" in res

    with pytest.raises(ExportError, match="Extracted parameter dictionary is empty"):
        extract_pytorch_state_dict(12345)

    extracted = extract_pytorch_state_dict({"all_layers.weight": [1.0], "embed_tokens.weight": [1.0]})
    assert "all_layers.weight" in extracted

    empty_dir = tmp_path / "empty_weights"
    empty_dir.mkdir()
    monkeypatch.undo()
    with pytest.raises(ExportError, match="allow_dummy_fallback is False"):
        extract_pytorch_state_dict(str(empty_dir), allow_dummy_fallback=False)

    monkeypatch.setitem(sys.modules, "gemma_4_sql.backends.pytorch.gemma4.modeling", None)
    with pytest.raises(ExportError, match="Failed to extract state dict"):
        extract_pytorch_state_dict(str(empty_dir), allow_dummy_fallback=True)


def test_gguf_torch_unaligned_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test unaligned torch tensor quantization and optional torch import reload."""
    import importlib

    import gemma_4_sql.backends.pytorch.gguf as gguf_mod

    assert len(quantize_tensor_q8_0(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))) > 0
    assert len(quantize_tensor_q4_0(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))) > 0
    assert len(quantize_tensor_q4_k_m(torch.tensor([1.0] * 10, dtype=torch.float32))) > 0
    assert len(quantize_tensor_q4_k_m(torch.tensor([1.0] * 256, dtype=torch.float32))) == 144

    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(gguf_mod)
    assert gguf_mod.torch is None
    monkeypatch.undo()
    importlib.reload(gguf_mod)
