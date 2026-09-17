"""Comprehensive tests for PyTorch GGUF v3 quantization, extraction, and export."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch
except (ImportError, RuntimeError):
    torch = None

import math

from gemma_4_sql.backends.pytorch.gguf import (
    extract_pytorch_state_dict,
    quantize_tensor_f16,
    quantize_tensor_q4_0,
    quantize_tensor_q4_k_m,
    quantize_tensor_q8_0,
    write_gguf_v3,
)
from gemma_4_sql.backends.pytorch.quantize import _export_gguf, validate_gguf_file
from gemma_4_sql.exceptions import ExportError


def test_quantize_tensor_f16() -> None:
    """Test F16 quantization."""
    data = [1.0, -2.5, 0.0, math.pi]
    res = quantize_tensor_f16(data)
    assert len(res) == len(data) * 2  # 2 bytes per float16
    unpacked = struct.unpack(f"<{len(data)}e", res)
    assert abs(unpacked[0] - 1.0) < 1e-3
    assert abs(unpacked[1] - (-2.5)) < 1e-3


def test_quantize_tensor_q8_0() -> None:
    """Test Q8_0 quantization format and block structure."""
    data = [float(i) for i in range(64)]
    res = quantize_tensor_q8_0(data, block_size=32)
    # 2 blocks of 32: each block is 2 bytes scale + 32 bytes int8 = 34 bytes * 2 = 68 bytes
    assert len(res) == 68

    with pytest.raises(ValueError, match="block_size must be positive"):
        quantize_tensor_q8_0(data, block_size=0)


def test_quantize_tensor_q4_0() -> None:
    """Test Q4_0 quantization format and packed nibbles."""
    data = [float(i) for i in range(64)]
    res = quantize_tensor_q4_0(data, block_size=32)
    # 2 blocks of 32: each block is 2 bytes scale + 16 bytes nibbles = 18 bytes * 2 = 36 bytes
    assert len(res) == 36

    with pytest.raises(ValueError, match="positive even integer"):
        quantize_tensor_q4_0(data, block_size=31)

    with pytest.raises(ValueError, match="positive even integer"):
        quantize_tensor_q4_0(data, block_size=-4)


def test_quantize_tensor_q4_k_m() -> None:
    """Test Q4_K_M quantization format."""
    data = [float(i % 10) for i in range(256)]
    res = quantize_tensor_q4_k_m(data)
    # 256 block: 2 scale + 2 dmin + 12 meta + 128 data = 144 bytes
    assert len(res) == 144


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


def test_extract_pytorch_state_dict_from_model() -> None:
    """Test extracting state dict from a model object with state_dict() method."""
    mock_model = MagicMock()
    mock_model.state_dict.return_value = {"model.embed_tokens.weight": [1.0]}
    extracted = extract_pytorch_state_dict(mock_model)
    assert "token_embd.weight" in extracted


def test_extract_pytorch_state_dict_empty_raises() -> None:
    """Test extract_pytorch_state_dict raises ExportError on empty dictionary."""
    with pytest.raises(ExportError, match="Extracted parameter dictionary is empty"):
        extract_pytorch_state_dict({})


def test_extract_pytorch_state_dict_string_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test string preset extraction fallback."""
    # When transformers fails, it falls back to Gemma4 native modeling
    extracted = extract_pytorch_state_dict("gemma4-preset")
    assert "token_embd.weight" in extracted


def test_write_and_validate_gguf_v3(tmp_path: Path) -> None:
    """Test writing and validating a complete GGUF v3 file with multiple tensor types."""
    gguf_file = tmp_path / "model.gguf"
    metadata = {
        "general.architecture": "gemma4",
        "general.name": "test-model",
        "model.context_length": 4096,
        "model.is_fine_tuned": True,
        "model.learning_rate": 0.0001,
        "tokenizer.tokens": ["<pad>", "<eos>", "SELECT"],
    }
    tensors = {
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
    assert info["metadata"]["model.is_fine_tuned"] == 1
    assert abs(info["metadata"]["model.learning_rate"] - 0.0001) < 1e-4

    # Verify tensor descriptors
    tensor_names = [t["name"] for t in info["tensors"]]
    assert "token_embd.weight" in tensor_names
    assert "blk.0.attn_q.weight" in tensor_names


def test_write_gguf_unsupported_out_type(tmp_path: Path) -> None:
    """Test write_gguf_v3 raises ValueError on unknown out_type."""
    with pytest.raises(ValueError, match="Unsupported GGUF quantization format"):
        write_gguf_v3(tmp_path / "bad.gguf", {}, {}, out_type="invalid_type")


def test_validate_gguf_file_corrupt(tmp_path: Path) -> None:
    """Test validate_gguf_file error checking."""
    missing = tmp_path / "missing.gguf"
    with pytest.raises(FileNotFoundError):
        validate_gguf_file(missing)

    corrupt_magic = tmp_path / "corrupt_magic.gguf"
    corrupt_magic.write_bytes(b"BAD!12345678")
    with pytest.raises(ValueError, match="Corrupt GGUF magic header"):
        validate_gguf_file(corrupt_magic)

    corrupt_ver = tmp_path / "corrupt_ver.gguf"
    corrupt_ver.write_bytes(b"GGUF" + struct.pack("<I", 99))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        validate_gguf_file(corrupt_ver)


def test_export_gguf_end_to_end(tmp_path: Path) -> None:
    """Test _export_gguf end-to-end creates and validates file."""
    reduction, status = _export_gguf("test-gemma", export_path=str(tmp_path), out_type="q4_0")
    assert status == "quantized_gguf"
    assert reduction == pytest.approx(0.6)

    exported_file = tmp_path / "test-gemma.gguf"
    assert exported_file.exists()
    info = validate_gguf_file(exported_file)
    assert info["version"] == 3


def test_flatten_values_edge_cases() -> None:
    """Test _flatten_values on bytes, bytearrays, torch-like, numpy-like, and scalar objects."""
    import numpy as np

    from gemma_4_sql.backends.pytorch.gguf import _flatten_values

    assert _flatten_values(b"\x01\x02") == [1.0, 2.0]
    assert _flatten_values(bytearray([3, 4])) == [3.0, 4.0]
    assert _flatten_values(42.5) == [42.5]
    assert _flatten_values(np.array([[1.0, 2.0], [3.0, 4.0]])) == [1.0, 2.0, 3.0, 4.0]

    class MockTorchTensor:
        """Mock torch tensor for flatten test."""

        def detach(self) -> MockTorchTensor:
            """Return self."""
            return self

        def cpu(self) -> MockTorchTensor:
            """Return self."""
            return self

        def flatten(self) -> MockTorchTensor:
            """Return self."""
            return self

        def numpy(self) -> np.ndarray:
            """Return numpy array."""
            return np.array([5.0, 6.0])

    assert _flatten_values(MockTorchTensor()) == [5.0, 6.0]


def test_quantize_fallbacks_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize fallback implementations when numpy is None."""
    import gemma_4_sql.backends.pytorch.gguf as gguf_mod

    monkeypatch.setattr(gguf_mod, "np", None)

    # Q8_0 fallback
    data = [float(i) for i in range(35)]  # Needs padding
    res_q8 = gguf_mod.quantize_tensor_q8_0(data, block_size=32)
    assert len(res_q8) == 68

    # Q4_0 fallback
    res_q4 = gguf_mod.quantize_tensor_q4_0(data, block_size=32)
    assert len(res_q4) == 36

    # Q4_K_M fallback
    data_k = [float(i % 5) for i in range(260)]  # Needs padding
    res_qk = gguf_mod.quantize_tensor_q4_k_m(data_k)
    assert len(res_qk) == 288


def test_write_gguf_all_quant_types(tmp_path: Path) -> None:
    """Test writing GGUF files with F16, Q8_0, and Q4_K_M formats."""
    metadata = {
        "general.architecture": "gemma4",
        "model.context_length": 2048,
    }
    tensors = {
        "token_embd.weight": [float(i) for i in range(256)],
    }

    # F16
    f16_file = tmp_path / "model_f16.gguf"
    write_gguf_v3(f16_file, metadata=metadata, tensors=tensors, out_type="f16")
    assert f16_file.exists()
    assert validate_gguf_file(f16_file)["tensor_count"] == 1

    # Q8_0
    q8_file = tmp_path / "model_q8.gguf"
    write_gguf_v3(q8_file, metadata=metadata, tensors=tensors, out_type="q8_0")
    assert q8_file.exists()
    assert validate_gguf_file(q8_file)["tensor_count"] == 1

    # Q4_K_M
    qk_file = tmp_path / "model_qk.gguf"
    write_gguf_v3(qk_file, metadata=metadata, tensors=tensors, out_type="q4_k_m")
    assert qk_file.exists()
    assert validate_gguf_file(qk_file)["tensor_count"] == 1


def test_export_gguf_explicit_filepath(tmp_path: Path) -> None:
    """Test _export_gguf when export_path is an explicit .gguf file path."""
    target_file = tmp_path / "custom_output.gguf"
    _reduction, status = _export_gguf("test-gemma", export_path=str(target_file), out_type="q8_0")
    assert status == "quantized_gguf"
    assert target_file.exists()


def test_extract_state_dict_local_file_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extract_pytorch_state_dict error when local file load fails and fallback fails.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import sys

    local_dir = tmp_path / "dummy_weights"
    local_dir.mkdir()

    monkeypatch.setitem(sys.modules, "gemma_4_sql.backends.pytorch.gemma4.modeling", None)
    with pytest.raises(ExportError, match="Failed to extract state dict"):
        extract_pytorch_state_dict(str(local_dir))


def test_gguf_tensor_quantize_with_numpy_and_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test tensor quantization with numpy arrays, torch tensors, and fallback when numpy is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import numpy as np

    import gemma_4_sql.backends.pytorch.gguf as gguf_mod

    # Numpy and Torch inputs
    np_arr = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    assert len(quantize_tensor_f16(np_arr)) == 8

    if torch is not None:
        monkeypatch.setattr(gguf_mod, "torch", torch)
        t_f16 = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
        assert len(quantize_tensor_f16(t_f16)) == 8

        t_64 = torch.tensor([float(i) for i in range(64)], dtype=torch.float32)
        assert len(quantize_tensor_q8_0(t_64, block_size=32)) == 68
        assert len(quantize_tensor_q4_0(t_64, block_size=32)) == 36

        t_256 = torch.tensor([float(i % 10) for i in range(256)], dtype=torch.float32)
        assert len(quantize_tensor_q4_k_m(t_256)) == 144

    # Pure Python fallback when np is None
    monkeypatch.setattr(gguf_mod, "np", None)
    py_data_64 = [float(i) for i in range(64)]
    assert len(quantize_tensor_q8_0(py_data_64, block_size=32)) == 68
    assert len(quantize_tensor_q4_0(py_data_64, block_size=32)) == 36

    py_data_256 = [float(i % 10) for i in range(256)]
    assert len(quantize_tensor_q4_k_m(py_data_256)) == 144


def test_gguf_write_metadata_types_and_payloads(tmp_path: Path) -> None:
    """Test write_gguf_v3 with float metadata, array metadata, and raw bytes tensors.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    target_file = tmp_path / "metadata_test.gguf"
    metadata: dict[str, object] = {
        "float_val": math.pi,
        "list_val": ["a", "b", "c"],
        "tuple_val": (1, 2),
        "int_val": 42,
        "bool_val": True,
        "str_val": "hello",
        "custom_obj": object(),
    }
    tensors: dict[str, object] = {
        "raw_tensor": b"\x00" * 32,
        "2d_tensor": [[1.0] * 32] * 8,
        "1d_tensor": [0.5] * 256,
    }
    write_gguf_v3(target_file, metadata=metadata, tensors=tensors, out_type="q4_k_m")
    assert target_file.exists()
    assert validate_gguf_file(target_file)["tensor_count"] == 3

    # Test out_type="f16" and tensor without __len__ (scalar)
    f16_file = tmp_path / "f16_test.gguf"
    write_gguf_v3(f16_file, metadata={"k": "v"}, tensors={"scalar_tensor": 42.0}, out_type="f16")
    assert f16_file.exists()


def test_extract_pytorch_state_dict_callable_object() -> None:
    """Test extract_pytorch_state_dict from an object with a state_dict callable.

    Returns:
        None.
    """

    class MockModel:
        """Mock model with state_dict callable."""

        def state_dict(self) -> dict[str, list[float]]:
            """Return mock state dict."""
            return {"model.embed_tokens.weight": [1.0, 2.0]}

    extracted = extract_pytorch_state_dict(MockModel())
    assert "token_embd.weight" in extracted


def test_extract_state_dict_local_hf_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extract_pytorch_state_dict from a local model directory using transformers.

    Args:
        tmp_path: Temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    local_dir = tmp_path / "local_hf_model"
    local_dir.mkdir()

    class MockMod:
        """Mock model."""

        def state_dict(self) -> dict[str, list[float]]:
            """Return state dict."""
            return {
                "embed_tokens.weight": [1.0],
                "layers.notanint.weight": [1.0],
            }

    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = MockMod()
    monkeypatch.setattr("transformers.AutoModelForCausalLM", mock_auto)
    res = extract_pytorch_state_dict(str(local_dir))
    assert "token_embd.weight" in res


def test_gguf_branches_and_edge_cases(tmp_path: Path) -> None:
    """Test GGUF quantization and export edge cases and formats.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    # 1. Line 120: quantize_tensor_q8_0 with np.ndarray
    import numpy as np

    res_q8_np = quantize_tensor_q8_0(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert len(res_q8_np) > 0

    # Plain list fallback
    res_q8 = quantize_tensor_q8_0([1.0, 2.0, 3.0, 4.0])
    assert len(res_q8) > 0

    # 2. Branch 308->334: extract_pytorch_state_dict with unrecognized type raises ExportError
    with pytest.raises(ExportError, match="Extracted parameter dictionary is empty"):
        extract_pytorch_state_dict(12345)

    # 3. Lines 361-362: parameter containing 'layers.' where 'layers' is not in parts triggers ValueError
    extracted = extract_pytorch_state_dict({"all_layers.weight": [1.0], "embed_tokens.weight": [1.0]})
    assert "all_layers.weight" in extracted
    assert "token_embd.weight" in extracted

    # 4. Line 397 and lines 477-478: write_gguf_v3 with out_type="q4_0"
    p_q4_0 = tmp_path / "model_q4_0.gguf"
    write_gguf_v3(p_q4_0, {}, {"token_embd.weight": [1.0] * 32}, out_type="q4_0")
    assert p_q4_0.exists()

    # 5. Lines 468-469: write_gguf_v3 with out_type="f32"
    p_f32 = tmp_path / "model_f32.gguf"
    write_gguf_v3(p_f32, {}, {"token_embd.weight": [1.0] * 32}, out_type="f32")
    assert p_f32.exists()

    # 6. Lines 479-480: write_gguf_v3 with out_type="q4_k_m"
    p_q4k = tmp_path / "model_q4k.gguf"
    write_gguf_v3(p_q4k, {}, {"token_embd.weight": [1.0] * 256}, out_type="q4_k_m")
    assert p_q4k.exists()


def test_gguf_export_corrupted_file_validation(tmp_path: Path) -> None:
    """Test validate_gguf_file on empty or corrupted files."""
    import struct

    corrupted = tmp_path / "corrupted.gguf"
    corrupted.write_bytes(b"BAD_MAGIC_HEADER")
    with pytest.raises(ValueError, match="Corrupt GGUF magic header"):
        validate_gguf_file(corrupted)

    empty_f = tmp_path / "empty.gguf"
    empty_f.write_bytes(b"")
    with pytest.raises(ValueError, match="Corrupt GGUF magic header"):
        validate_gguf_file(empty_f)

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(b"GGUF\x00")
    with pytest.raises((ValueError, struct.error)):
        validate_gguf_file(truncated)
