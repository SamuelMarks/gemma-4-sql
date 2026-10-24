"""Behavioral and unit tests for PyTorch model quantization and GGUF export."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_pytorch_quantize_imports() -> None:
    """Verify pytorch.quantize gracefully sets optional imports to None when unavailable."""
    import importlib

    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch.dict(sys.modules, {"torch": None, "numpy": None, "transformers": None}):
        importlib.reload(quantize_module)
        assert quantize_module.torch is None
        assert quantize_module.np is None
        assert quantize_module.BitsAndBytesConfig is None
        assert quantize_module.AutoModelForCausalLM is None
        assert quantize_module.AutoTokenizer is None
    importlib.reload(quantize_module)


def test_validate_gguf_file(tmp_path: Path) -> None:
    """Verify validate_gguf_file rejects missing files, corrupt magic bytes, and unsupported GGUF versions."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"

    with pytest.raises(FileNotFoundError):
        quantize_module.validate_gguf_file(file_path)

    with open(file_path, "wb") as f:
        f.write(b"BADH")
    with pytest.raises(ValueError, match="Corrupt GGUF magic"):
        quantize_module.validate_gguf_file(file_path)

    with open(file_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 1))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        quantize_module.validate_gguf_file(file_path)


def test_validate_gguf_file_valid(tmp_path: Path) -> None:
    """Verify validate_gguf_file parses all GGUF v3 metadata types and tensor headers."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"

    with open(file_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 11))

        # KV 1: String
        f.write(struct.pack("<Q", 2) + b"k1" + struct.pack("<I", 8) + struct.pack("<Q", 2) + b"v1")
        # KV 2: Bool
        f.write(struct.pack("<Q", 2) + b"k2" + struct.pack("<I", 7) + struct.pack("<B", 1))
        # KV 3: Int32
        f.write(struct.pack("<Q", 2) + b"k3" + struct.pack("<I", 4) + struct.pack("<i", 42))
        # KV 4: Float32
        f.write(struct.pack("<Q", 2) + b"k4" + struct.pack("<I", 6) + struct.pack("<f", math.pi))
        # KV 5: Int64
        f.write(struct.pack("<Q", 2) + b"k5" + struct.pack("<I", 10) + struct.pack("<q", 100))
        # KV 6: Float64
        f.write(struct.pack("<Q", 2) + b"k6" + struct.pack("<I", 12) + struct.pack("<d", 2.0))
        # KV 7: Array of int32
        f.write(struct.pack("<Q", 2) + b"k7" + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", 1) + struct.pack("<i", 5))
        # KV 8: Array of strings
        f.write(struct.pack("<Q", 2) + b"k8" + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 1) + struct.pack("<Q", 1) + b"a")
        # KV 9: Array of float32
        f.write(struct.pack("<Q", 2) + b"k9" + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 1) + struct.pack("<f", 1.0))
        # KV 10: Array of other
        f.write(struct.pack("<Q", 3) + b"k10" + struct.pack("<I", 9) + struct.pack("<I", 99) + struct.pack("<Q", 1))
        # KV 11: Unknown type
        f.write(struct.pack("<Q", 3) + b"k11" + struct.pack("<I", 99))

        # Tensor descriptor
        f.write(struct.pack("<Q", 2) + b"t1" + struct.pack("<I", 1) + struct.pack("<Q", 10) + struct.pack("<I", 1) + struct.pack("<Q", 0))

    res = quantize_module.validate_gguf_file(file_path)
    assert res["version"] == 3
    assert res["tensor_count"] == 1
    meta: Any = res["metadata"]
    assert meta["k1"] == "v1"
    assert meta["k2"] is True
    assert meta["k3"] == 42
    assert "tensors" in res


def test_write_gguf_file(tmp_path: Path) -> None:
    """Verify _write_gguf_file writes a valid GGUF v3 binary container with quantized tensors."""
    import torch

    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"
    quantize_module._write_gguf_file(file_path, "m")
    parsed = quantize_module.validate_gguf_file(file_path)
    assert parsed["version"] == 3

    file_path_custom = tmp_path / "custom.gguf"
    quantize_module._write_gguf_file(
        file_path_custom,
        "m",
        {"blk.0.attn_q.weight": torch.ones((4, 8), dtype=torch.float32)},
        {"general.architecture": "gemma4"},
    )
    parsed_custom = quantize_module.validate_gguf_file(file_path_custom)
    assert parsed_custom["tensor_count"] == 1


def test_apply_awq_quantization(tmp_path: Path) -> None:
    """Verify _apply_awq_quantization runs AutoAWQ quantization and handles missing awq package."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    mock_awq = MagicMock()
    mock_awq.AutoAWQForCausalLM = MagicMock()
    mock_tf = MagicMock()

    with patch.dict(sys.modules, {"awq": mock_awq, "transformers": mock_tf}):
        res = quantize_module._apply_awq_quantization("m", str(tmp_path), ["data"])
        assert res[1] == "quantized_awq"

        res2 = quantize_module._apply_awq_quantization("m", str(tmp_path / "sub"), None)
        assert res2[1] == "quantized_awq"

    with patch.dict(sys.modules, {"awq": None}), pytest.raises(DependencyMissingError):
        quantize_module._apply_awq_quantization("m")


def test_apply_gptq_quantization(tmp_path: Path) -> None:
    """Verify _apply_gptq_quantization runs Optimum GPTQQuantizer and handles missing optimum package."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    mock_opt = MagicMock()
    mock_opt.gptq.GPTQQuantizer = MagicMock()
    mock_tf = MagicMock()

    with patch.dict(sys.modules, {"optimum": mock_opt, "transformers": mock_tf}):
        res = quantize_module._apply_gptq_quantization("m", str(tmp_path))
        assert res[1] == "quantized_gptq"

        res2 = quantize_module._apply_gptq_quantization("m", str(tmp_path / "sub"))
        assert res2[1] == "quantized_gptq"

    with patch.dict(sys.modules, {"optimum": None}), pytest.raises(DependencyMissingError):
        quantize_module._apply_gptq_quantization("m")


def test_export_gguf(tmp_path: Path) -> None:
    """Verify _export_gguf creates and validates real GGUF files and reuses existing files."""
    import torch

    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch(
        "gemma_4_sql.backends.pytorch.gguf.extract_pytorch_state_dict",
        return_value={"token_embd.weight": torch.randn(4, 32, dtype=torch.float32)},
    ) as mock_extract:
        res = quantize_module._export_gguf("m", str(tmp_path))
        assert res[1] == "quantized_gguf"
        assert (tmp_path / "m.gguf").exists()

        # Re-exporting when file already exists skips extraction
        mock_extract.reset_mock()
        quantize_module._export_gguf("m", str(tmp_path))
        mock_extract.assert_not_called()

        # Fallback when state_dict extraction raises ValueError
        mock_extract.side_effect = ValueError("error")
        res_fb = quantize_module._export_gguf("m2", str(tmp_path))
        assert res_fb[1] == "quantized_gguf"
        assert (tmp_path / "m2.gguf").exists()


def test_quantize_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_model runs end-to-end across GGUF, INT8, INT4, AWQ, and GPTQ."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    tiny_model = Gemma4ForCausalLM(cfg)

    # 1. End-to-end GGUF quantization against tiny PyTorch model
    gguf_dir = tmp_path / "gguf_out"
    with patch(
        "gemma_4_sql.backends.pytorch.gguf.extract_pytorch_state_dict",
        return_value={k: v for k, v in tiny_model.state_dict().items() if v.ndim == 2},
    ):
        res_gguf = quantize_module.quantize_model("tiny-gemma4", "gguf", export_path=str(gguf_dir), out_type="q8_0")
        assert res_gguf["status"] == "quantized_gguf"
        assert res_gguf["memory_reduction_factor"] == 0.6
        gguf_file = gguf_dir / "tiny-gemma4.gguf"
        assert gguf_file.exists()
        validated = quantize_module.validate_gguf_file(gguf_file)
        assert validated["version"] == 3
        assert int(validated["tensor_count"]) > 0

    # 2. End-to-end INT8 and INT4 quantization on tiny PyTorch model
    from gemma_4_sql.type_hints import get_quantized_artifact

    monkeypatch.setattr(quantize_module, "BitsAndBytesConfig", lambda **kw: kw)
    res_int8 = quantize_module.quantize_model("tiny-gemma4", "int8", model=tiny_model)
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == 0.5
    art_int8 = get_quantized_artifact(tiny_model)
    assert art_int8 is not None
    assert art_int8.is_quantized is True
    assert art_int8.method == "int8"

    tiny_model_int4 = Gemma4ForCausalLM(cfg)
    res_int4 = quantize_module.quantize_model("tiny-gemma4", "int4", model=tiny_model_int4)
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == 0.75
    art_int4 = get_quantized_artifact(tiny_model_int4)
    assert art_int4 is not None
    assert art_int4.is_quantized is True
    assert art_int4.method == "int4"

    # 3. End-to-end AWQ and GPTQ dispatch through quantize_model
    mock_awq = MagicMock()
    mock_awq.AutoAWQForCausalLM = MagicMock()
    mock_tf = MagicMock()
    with patch.dict(sys.modules, {"awq": mock_awq, "transformers": mock_tf}):
        res_awq = quantize_module.quantize_model(
            "tiny-gemma4",
            "awq",
            export_path=str(tmp_path / "awq_out"),
            calib_data=["SELECT 1;"],
        )
        assert res_awq["status"] == "quantized_awq"

    mock_opt = MagicMock()
    mock_opt.gptq.GPTQQuantizer = MagicMock()
    with patch.dict(sys.modules, {"optimum": mock_opt, "transformers": mock_tf}):
        res_gptq = quantize_module.quantize_model(
            "tiny-gemma4",
            "gptq",
            export_path=str(tmp_path / "gptq_out"),
        )
        assert res_gptq["status"] == "quantized_gptq"

    monkeypatch.setattr(quantize_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("m", "int8")
