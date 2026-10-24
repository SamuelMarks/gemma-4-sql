"""Unit tests for multimodal image, WAV audio, and prompt processing utilities."""

from __future__ import annotations

import builtins
import struct
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.common_multimodal as mod


def test_missing_np_and_pil_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Verify load_image_bytes, process_audio, and _parse_wav_samples handle missing numpy/PIL and non-16-bit WAVs."""
    assert len(synthetic_image_bytes) > 0
    assert len(synthetic_wav_bytes) > 0
    orig_import: Any = builtins.__import__

    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        monkeypatch.setattr(mod, "Image", None)

        def mock_import_no_pil(name: str, *args: Any, **kwargs: Any) -> Any:
            """Simulate missing PIL."""
            if name == "PIL":
                raise ImportError("simulated missing import")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import_no_pil):
            mock_image_input = MagicMock()
            mock_image_input.save = MagicMock()
            with pytest.raises(ValueError):
                mod.load_image_bytes(mock_image_input)
            assert mock_debug.call_args_list[0][0][0] == "Failed to import PIL: %s"

    with patch("gemma_4_sql.backends.common_multimodal.logger.debug"):
        monkeypatch.setattr(mod, "Image", None)
        mock_image_input = MagicMock()
        mock_image_input.save = MagicMock()
        with pytest.raises(ValueError, match="Unsupported image input type"):
            mod.load_image_bytes(mock_image_input)

    with patch("gemma_4_sql.backends.common_multimodal.logger.debug"):
        monkeypatch.setattr(mod, "np", None)

        def mock_import_no_np(name: str, *args: Any, **kwargs: Any) -> Any:
            """Simulate missing numpy."""
            if name == "numpy":
                raise ImportError("simulated missing np")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import_no_np):
            wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            res = mod.process_audio(wav_header)
            assert len(res["audio_values"]) > 0

            class DummyAudio:
                """Unsupported audio input stub."""

            dummy_aud: Any = DummyAudio()
            with pytest.raises(ValueError):
                mod.process_audio(dummy_aud)

    bad_wav_8bit = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x08\x00data\x02\x00\x00\x00\x01\x01"
    res_8bit = mod._parse_wav_samples(bad_wav_8bit)
    assert len(res_8bit[0]) > 0

    bad_wav_24bit = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x18\x00data\x02\x00\x00\x00\x01\x01"
    res_24bit = mod._parse_wav_samples(bad_wav_24bit)
    assert len(res_24bit[0]) > 0


def test_process_audio_numpy_and_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify process_audio handles numpy arrays, lazy numpy imports, and missing numpy fallback."""
    import numpy as np_real

    audio_input = np_real.array([[1.0, 2.0], [3.0, 4.0]])
    res = mod.process_audio(audio_input)
    assert len(res["audio_values"]) > 0
    assert len(res["spectrogram"]) > 0

    monkeypatch.setattr(mod, "np", None)
    audio_1d = np_real.array([1.0, 2.0, 3.0])
    res_lazy = mod.process_audio(audio_1d)
    assert len(res_lazy["audio_values"]) > 0

    monkeypatch.setattr(mod, "np", None)
    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res_no_np = mod.process_audio(wav_header)
    assert len(res_no_np["spectrogram"]) == res_no_np["num_frames"]

    # Also test lazy numpy import failure logging inside process_audio spectrogram computation
    orig_import: Any = builtins.__import__

    def mock_import_fail_np(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise ImportError when importing numpy."""
        if name == "numpy":
            raise ImportError("Simulated")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_fail_np)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.setattr(mod, "np", None)
    mock_logger = MagicMock()
    monkeypatch.setattr(mod, "logger", mock_logger)
    monkeypatch.setattr(mod, "load_audio_bytes", lambda _x: b"")
    monkeypatch.setattr(mod, "_parse_wav_samples", lambda _x: ([0.0], 16000))
    mod.process_audio(b"fake")
    assert any("Failed to import numpy" in str(c) for c in mock_logger.debug.call_args_list)


def test_parse_wav_samples_chunk_variants() -> None:
    """Verify _parse_wav_samples handles missing data chunks, JUNK chunks, truncated chunks, and valid PCM data."""
    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)

    res_no_data, _ = mod._parse_wav_samples(header + fmt_chunk)
    assert len(res_no_data) > 0

    junk = b"JUNK" + struct.pack("<I", 4) + b"1234"
    res_junk, _ = mod._parse_wav_samples(header + junk)
    assert len(res_junk) > 0

    res_padded, _ = mod._parse_wav_samples(header + junk + b"123")
    assert len(res_padded) > 0

    res_trunc, _ = mod._parse_wav_samples(header + b"JUNK1")
    assert len(res_trunc) > 0

    res_empty_wave, _ = mod._parse_wav_samples(header)
    assert len(res_empty_wave) > 0

    data_chunk = b"data" + struct.pack("<I", 2) + b"\x00\x00"
    res_valid, _ = mod._parse_wav_samples(header + fmt_chunk + data_chunk)
    assert len(res_valid) > 0

    data_chunk_oversized = b"data" + struct.pack("<I", 1000)
    res_over, _ = mod._parse_wav_samples(header + fmt_chunk + data_chunk_oversized)
    assert len(res_over) > 0


def test_format_multimodal_prompt_and_image_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify format_multimodal_prompt, base64 image loading, and PIL image processing."""
    with pytest.raises(ValueError):
        mod.format_multimodal_prompt("")

    assert mod.format_multimodal_prompt("hello", has_image=True, has_audio=True).get("modality") == "multimodal"
    assert mod.format_multimodal_prompt("hello", has_image=True).get("modality") == "vision"
    assert mod.format_multimodal_prompt("hello", has_audio=True).get("modality") == "audio"
    assert mod.format_multimodal_prompt("hello").get("modality") == "text"

    assert mod.load_image_bytes("data:image/png;base64,iVBORw0KGgo=") == bytes.fromhex("89504e470d0a1a0a")

    class SavableImage:
        """Stub PIL-like image with save method."""

        def save(self, buf: Any, format: str = "PNG") -> None:
            """Write dummy PNG bytes."""
            buf.write(b"saved_png_bytes")

    savable_img: Any = SavableImage()
    monkeypatch.setattr(mod, "Image", object())
    assert mod.load_image_bytes(savable_img) == b"saved_png_bytes"
    monkeypatch.setattr(mod, "Image", None)
    monkeypatch.setitem(sys.modules, "PIL", MagicMock(Image=object()))
    assert mod.load_image_bytes(savable_img) == b"saved_png_bytes"

    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        mod.process_image(b"", target_size=(0, 0))
    with pytest.raises(ValueError, match="target dimensions"):
        mod.process_image(b"", target_size=(224, 224), patch_size=15)

    from tests.conftest import MockPILImage

    monkeypatch.setattr(mod, "Image", MockPILImage)
    res_img = mod.process_image(b"fakebytes", target_size=(224, 224), patch_size=14)
    assert res_img["num_patches"] == 256
    assert res_img["shape"] == (3, 224, 224)

    with (
        patch.object(mod, "process_image", return_value={"pixel_values": [1.0]}),
        patch.object(mod, "process_audio", return_value={"audio_values": [2.0]}),
    ):
        p_out, pix_out, aud_out = mod.prepare_multimodal_inputs(
            "SELECT 1",
            {"image_path": "img.png", "audio_path": "aud.wav"},
            lambda arr: ("tensor", arr),
        )
        assert "<image>" in p_out
        assert "<audio>" in p_out
        assert pix_out == ("tensor", [1.0])
        assert aud_out == ("tensor", [2.0])

        p_text, pix_none, aud_none = mod.prepare_multimodal_inputs("SELECT 1", {}, lambda arr: arr)
        assert p_text == "SELECT 1"
        assert pix_none is None
        assert aud_none is None

        p_pre, pix_pre, aud_pre = mod.prepare_multimodal_inputs(
            "SELECT 1",
            {"image_path": "img.png", "audio_path": "aud.wav", "pixel_values": "pre_pix", "audio_values": "pre_aud"},
            lambda arr: arr,
        )
        assert "<image>" in p_pre
        assert pix_pre == "pre_pix"
        assert aud_pre == "pre_aud"

    class MockFailingPILImage:
        """Stub PIL Image that raises ValueError on open."""

        @staticmethod
        def open(_buf: Any) -> None:
            """Raise ValueError."""
            raise ValueError("PIL error")

    monkeypatch.setattr(mod, "Image", MockFailingPILImage)
    with pytest.raises(ValueError, match="Corrupted or unreadable image:"):
        mod.process_image(b"fakebytes", target_size=(224, 224), patch_size=14)

    monkeypatch.setattr(mod, "np", None)
    with pytest.raises(ValueError, match="Corrupted or unreadable image:"):
        mod.process_image(b"bad", target_size=(224, 224), patch_size=14)
