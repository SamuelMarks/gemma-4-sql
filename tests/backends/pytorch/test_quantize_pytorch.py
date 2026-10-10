"""Tests for PyTorch quantize."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_pytorch_quantize_imports():
    """Test pytorch quantize imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "numpy": None, "transformers": None}):
        import gemma_4_sql.backends.pytorch.quantize as quantize_module

        importlib.reload(quantize_module)
        assert quantize_module.torch is None
        assert quantize_module.np is None
        assert quantize_module.BitsAndBytesConfig is None
        assert quantize_module.AutoModelForCausalLM is None
        assert quantize_module.AutoTokenizer is None
    importlib.reload(quantize_module)


def test_validate_gguf_file(tmp_path):
    """Test validate_gguf_file."""
    import struct

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
        f.write(struct.pack("<I", 1))  # Version 1
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        quantize_module.validate_gguf_file(file_path)


def test_validate_gguf_file_valid(tmp_path):
    """Test validate_gguf_file valid."""
    import struct

    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    file_path = tmp_path / "test.gguf"

    with open(file_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # Version 3
        f.write(struct.pack("<Q", 1))  # 1 tensor
        f.write(struct.pack("<Q", 11))  # 11 KV pairs

        # KV 1: String
        f.write(struct.pack("<Q", 2))
        f.write(b"k1")
        f.write(struct.pack("<I", 8))  # String type
        f.write(struct.pack("<Q", 2))
        f.write(b"v1")

        # KV 2: Bool
        f.write(struct.pack("<Q", 2))
        f.write(b"k2")
        f.write(struct.pack("<I", 7))
        f.write(struct.pack("<B", 1))

        # KV 3: Int32
        f.write(struct.pack("<Q", 2))
        f.write(b"k3")
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<i", 42))

        # KV 4: Float32
        f.write(struct.pack("<Q", 2))
        f.write(b"k4")
        f.write(struct.pack("<I", 6))
        f.write(struct.pack("<f", 3.14))

        # KV 5: Int64
        f.write(struct.pack("<Q", 2))
        f.write(b"k5")
        f.write(struct.pack("<I", 10))
        f.write(struct.pack("<q", 100))

        # KV 6: Float64
        f.write(struct.pack("<Q", 2))
        f.write(b"k6")
        f.write(struct.pack("<I", 12))
        f.write(struct.pack("<d", 2.0))

        # KV 7: Array of int32
        f.write(struct.pack("<Q", 2))
        f.write(b"k7")
        f.write(struct.pack("<I", 9))  # Array type
        f.write(struct.pack("<I", 4))  # Int32 elements
        f.write(struct.pack("<Q", 1))  # 1 element
        f.write(struct.pack("<i", 5))

        # KV 8: Array of strings
        f.write(struct.pack("<Q", 2))
        f.write(b"k8")
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))
        f.write(b"a")

        # KV 9: Array of float32
        f.write(struct.pack("<Q", 2))
        f.write(b"k9")
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 6))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<f", 1.0))

        # KV 10: Array of other
        f.write(struct.pack("<Q", 3))
        f.write(b"k10")
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 99))
        f.write(struct.pack("<Q", 1))

        # KV 11: Unknown type
        f.write(struct.pack("<Q", 3))
        f.write(b"k11")
        f.write(struct.pack("<I", 99))

        # Tensor
        f.write(struct.pack("<Q", 2))
        f.write(b"t1")
        f.write(struct.pack("<I", 1))  # 1 dim
        f.write(struct.pack("<Q", 10))  # dim=10
        f.write(struct.pack("<I", 1))  # type=1
        f.write(struct.pack("<Q", 0))  # offset=0

    res = quantize_module.validate_gguf_file(file_path)
    assert res["version"] == 3
    assert res["tensor_count"] == 1
    assert res["metadata"]["k1"] == "v1"
    assert res["metadata"]["k2"] is True
    assert res["metadata"]["k3"] == 42
    assert "tensors" in res


def test_write_gguf_file(tmp_path):
    """Test _write_gguf_file."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch("gemma_4_sql.backends.pytorch.gguf.write_gguf_v3") as mock_write:
        file_path = tmp_path / "test.gguf"
        quantize_module._write_gguf_file(file_path, "m")
        mock_write.assert_called_once()

        # with tensors and meta
        quantize_module._write_gguf_file(file_path, "m", {"t": 1}, {"m": 1})


def test_apply_awq_quantization(tmp_path):
    """Test _apply_awq_quantization."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch("builtins.__import__") as mock_import:
        mock_awq = MagicMock()
        mock_awq.AutoAWQForCausalLM = MagicMock()
        mock_import.return_value = mock_awq

        quantize_module.AutoTokenizer = MagicMock()

        res = quantize_module._apply_awq_quantization("m", str(tmp_path), ["data"])
        assert res[1] == "quantized_awq"

        res2 = quantize_module._apply_awq_quantization("m", None, None)
        assert res2[1] == "quantized_awq"

        mock_import.side_effect = ImportError("error")
        with pytest.raises(DependencyMissingError):
            quantize_module._apply_awq_quantization("m")


def test_apply_gptq_quantization(tmp_path):
    """Test _apply_gptq_quantization."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch("builtins.__import__") as mock_import:
        mock_opt = MagicMock()
        mock_opt.gptq.GPTQQuantizer = MagicMock()
        mock_import.return_value = mock_opt

        quantize_module.AutoTokenizer = MagicMock()
        quantize_module.AutoModelForCausalLM = MagicMock()
        quantize_module.torch = MagicMock()

        res = quantize_module._apply_gptq_quantization("m", str(tmp_path))
        assert res[1] == "quantized_gptq"

        res2 = quantize_module._apply_gptq_quantization("m", None)
        assert res2[1] == "quantized_gptq"

        mock_import.side_effect = ImportError("error")
        with pytest.raises(DependencyMissingError):
            quantize_module._apply_gptq_quantization("m")


def test_export_gguf(tmp_path):
    """Test _export_gguf."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    with patch("gemma_4_sql.backends.pytorch.gguf.extract_pytorch_state_dict") as mock_extract:
        mock_extract.return_value = {"t": 1}
        with patch("gemma_4_sql.backends.pytorch.quantize._write_gguf_file"):
            with patch("gemma_4_sql.backends.pytorch.quantize.validate_gguf_file"):
                res = quantize_module._export_gguf("m", str(tmp_path))
                assert res[1] == "quantized_gguf"

                # Check file exists branch skip extract
                (tmp_path / "m.gguf").touch()
                mock_extract.reset_mock()
                quantize_module._export_gguf("m", str(tmp_path))
                mock_extract.assert_not_called()

                # Check error
                (tmp_path / "m2.gguf").unlink(missing_ok=True)
                mock_extract.side_effect = ValueError("error")
                quantize_module._export_gguf("m2", str(tmp_path))


def test_quantize_model():
    """Test quantize_model."""
    import gemma_4_sql.backends.pytorch.quantize as quantize_module

    quantize_module.torch = MagicMock()
    quantize_module.BitsAndBytesConfig = MagicMock()
    quantize_module.AutoModelForCausalLM = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.quantize.quantize_model_wrapper") as mock_wrapper:
        # GGUF
        quantize_module.quantize_model("m", "gguf")
        fn_gguf = mock_wrapper.call_args[1]["apply_fn"]
        with patch("gemma_4_sql.backends.pytorch.quantize._export_gguf") as mock_exp:
            mock_exp.return_value = (0.5, "ok")
            assert fn_gguf() == (0.5, "ok")

        # AWQ
        quantize_module.quantize_model("m", "awq", calib_data="d")
        fn_awq = mock_wrapper.call_args[1]["apply_fn"]
        with patch("gemma_4_sql.backends.pytorch.quantize._apply_awq_quantization") as mock_awq:
            mock_awq.return_value = (0.5, "ok")
            assert fn_awq() == (0.5, "ok")

        # GPTQ
        quantize_module.quantize_model("m", "gptq")
        fn_gptq = mock_wrapper.call_args[1]["apply_fn"]
        with patch("gemma_4_sql.backends.pytorch.quantize._apply_gptq_quantization") as mock_gptq:
            mock_gptq.return_value = (0.5, "ok")
            assert fn_gptq() == (0.5, "ok")

        # int8
        quantize_module.quantize_model("m", "int8")
        fn_int8 = mock_wrapper.call_args[1]["apply_fn"]
        with patch("gemma_4_sql.backends.pytorch.quantize.apply_bits_and_bytes_quantization") as mock_bnb:
            mock_bnb.return_value = (0.5, "ok")
            assert fn_int8() == (0.5, "ok")

    quantize_module.torch = None
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("m", "int8")
