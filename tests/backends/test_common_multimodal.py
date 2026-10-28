"""Unit tests for multimodal image, WAV audio, and prompt processing utilities."""

from __future__ import annotations

import io
import struct
import sys
from collections.abc import Callable
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.common_multimodal as mod
from gemma_4_sql.exceptions import MultimodalProcessingError


class _DummyPILSentinel:
    """Non-None sentinel for `mod.Image` without an `open` method."""


def test_missing_jnp_and_pil_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_image_bytes: bytes,
    synthetic_wav_bytes: bytes,
) -> None:
    """Verify load_image_bytes, process_audio, and _parse_wav_samples handle missing jax.numpy/PIL and multi-bit WAVs."""
    assert len(synthetic_image_bytes) > 0
    assert len(synthetic_wav_bytes) > 0

    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        monkeypatch.setattr(mod, "Image", None)
        monkeypatch.setattr(mod, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
        mock_image_input = MagicMock()
        mock_image_input.save = MagicMock()
        with pytest.raises(MultimodalProcessingError):
            mod.load_image_bytes(mock_image_input)
        assert mock_debug.call_args_list[0][0][0] == "Failed to import PIL: %s"

    with patch("gemma_4_sql.backends.common_multimodal.logger.debug"):
        monkeypatch.setattr(mod, "jnp", None)
        monkeypatch.setattr(mod, "optional_import", lambda _name: None)  # pyright: ignore[reportUnknownLambdaType]
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        res = mod.process_audio(wav_header)
        assert len(res["audio_values"]) > 0

        class DummyAudio:
            """Unsupported audio input stub."""

        untyped_process_audio: Callable[..., mod.ProcessedAudioDict] = mod.process_audio
        with pytest.raises(MultimodalProcessingError):
            untyped_process_audio(DummyAudio())  # ty: ignore[invalid-argument-type]

    wav_8bit = b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x44\xac\x00\x00\x01\x00\x08\x00data\x02\x00\x00\x00\x80\xff"
    res_8bit = mod._parse_wav_samples(wav_8bit)
    assert len(res_8bit[0]) == 2
    assert res_8bit[0][0] == 0.0

    wav_24bit = b"RIFF\x2a\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\xcc\x04\x02\x00\x03\x00\x18\x00data\x06\x00\x00\x00\x00\x00\x40\x00\x00\xc0"
    res_24bit = mod._parse_wav_samples(wav_24bit)
    assert len(res_24bit[0]) == 2
    assert res_24bit[0][0] == pytest.approx(0.5, rel=1e-4)
    assert res_24bit[0][1] == pytest.approx(-0.5, rel=1e-4)

    wav_32bit_int = b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x20\x00" + b"data\x04\x00\x00\x00" + struct.pack("<i", 1073741824)
    res_32int = mod._parse_wav_samples(wav_32bit_int)
    assert res_32int[0][0] == pytest.approx(0.5, rel=1e-4)

    wav_32bit_float = b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x03\x00\x01\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x20\x00" + b"data\x04\x00\x00\x00" + struct.pack("<f", 0.25)
    res_32float = mod._parse_wav_samples(wav_32bit_float)
    assert res_32float[0][0] == pytest.approx(0.25, rel=1e-5)

    wav_invalid_bits = b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x44\xac\x00\x00\x01\x00\x0c\x00data\x02\x00\x00\x00\x00\x00"
    with pytest.raises(MultimodalProcessingError, match="Unsupported WAV bits_per_sample: 12"):
        mod._parse_wav_samples(wav_invalid_bits)


def test_process_audio_jax_and_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify process_audio handles jax arrays, lazy jax.numpy imports, and missing jax.numpy fallback."""
    import jax.numpy as jnp_real

    audio_input = jnp_real.array([[1.0, 2.0], [3.0, 4.0]])
    res = mod.process_audio(audio_input)
    assert len(res["audio_values"]) > 0
    assert len(res["spectrogram"]) > 0

    short_tail = jnp_real.ones(500, dtype=jnp_real.float32) * 0.1
    monkeypatch.setattr(mod, "jnp", jnp_real)
    res_tail = mod.process_audio(short_tail, sample_rate=16000, n_mels=16, frame_length_ms=25, frame_shift_ms=10)
    assert res_tail["num_frames"] == 1

    res_small_mel = mod.process_audio(jnp_real.ones(600, dtype=jnp_real.float32), sample_rate=8000, n_mels=64, frame_length_ms=10, frame_shift_ms=5)
    assert res_small_mel["num_frames"] > 1

    monkeypatch.setattr(mod, "jnp", None)
    audio_1d = jnp_real.array([1.0, 2.0, 3.0])
    res_lazy = mod.process_audio(audio_1d)
    assert len(res_lazy["audio_values"]) > 0

    monkeypatch.setattr(mod, "jnp", None)
    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res_no_jnp = mod.process_audio(wav_header)
    assert len(res_no_jnp["spectrogram"]) == res_no_jnp["num_frames"]

    monkeypatch.setattr(mod, "optional_import", lambda _name: None)  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(mod, "jnp", None)
    mock_logger = MagicMock()
    monkeypatch.setattr(mod, "logger", mock_logger)
    monkeypatch.setattr(mod, "load_audio_bytes", lambda _x: b"raw")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(mod, "_parse_wav_samples", lambda _x: ([0.0], 16000))  # pyright: ignore[reportUnknownLambdaType]
    mod.process_audio(b"fake")
    assert any("Failed to import jax.numpy" in str(c) for c in mock_logger.debug.call_args_list)


def test_parse_wav_samples_chunk_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _parse_wav_samples handles missing data chunks, JUNK chunks, truncated headers, and valid PCM data."""
    header = b"RIFF" + struct.pack("<I", 36) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    junk = b"JUNK" + struct.pack("<I", 4) + b"1234"

    res_no_data, _ = mod._parse_wav_samples(header + fmt_chunk + junk)
    assert len(res_no_data) > 0

    with pytest.raises(MultimodalProcessingError, match="Audio byte payload cannot be empty"):
        mod._parse_wav_samples(b"")

    with pytest.raises(MultimodalProcessingError, match="Truncated or malformed RIFF/WAVE header"):
        mod._parse_wav_samples(header)

    data_chunk = b"data" + struct.pack("<I", 2) + b"\x00\x00"
    res_valid, _ = mod._parse_wav_samples(header + fmt_chunk + data_chunk)
    assert len(res_valid) > 0

    data_chunk_oversized = b"data" + struct.pack("<I", 1000)
    res_over, _ = mod._parse_wav_samples(header + fmt_chunk + data_chunk_oversized)
    assert len(res_over) > 0

    orig_unpack = struct.unpack
    call_count = 0

    def _unpack_fail_subchunk(fmt: str, buffer: bytes) -> tuple[int | float | bytes, ...]:
        """Raise struct.error on the 5th unpack call (subchunk size)."""
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise struct.error("subchunk error")
        return orig_unpack(fmt, buffer)

    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr("struct.unpack", _unpack_fail_subchunk)
        with pytest.raises(MultimodalProcessingError, match="Corrupted WAV subchunk header"):
            mod._parse_wav_samples(header + fmt_chunk + data_chunk)


def test_format_multimodal_prompt_and_image_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify format_multimodal_prompt (including tokenizer alignment), base64 image loading, and PIL image processing."""
    with pytest.raises(MultimodalProcessingError):
        mod.format_multimodal_prompt("")

    assert mod.format_multimodal_prompt("hello", has_image=True, has_audio=True).get("modality") == "multimodal"
    assert mod.format_multimodal_prompt("hello", has_image=True).get("modality") == "vision"
    assert mod.format_multimodal_prompt("hello", has_audio=True).get("modality") == "audio"
    assert mod.format_multimodal_prompt("hello").get("modality") == "text"

    from gemma_4_sql.tokenization import SQLTokenizer

    tok = SQLTokenizer()
    tok_res = mod.format_multimodal_prompt("SELECT 1", has_image=True, has_audio=True, tokenizer=tok)
    assert tok_res.get("image_token_mask") is not None
    assert any(tok_res.get("image_token_mask") or [])
    assert any(tok_res.get("audio_token_mask") or [])

    assert mod.load_image_bytes("data:image/png;base64,iVBORw0KGgo=") == bytes.fromhex("89504e470d0a1a0a")

    class SavableImage:
        """Stub PIL-like image with save method."""

        def save(self, fp: io.BytesIO, format: str = "PNG") -> None:
            """Write dummy PNG bytes."""
            _ = format
            fp.write(b"saved_png_bytes")

    savable_img = SavableImage()
    monkeypatch.setattr(mod, "Image", _DummyPILSentinel())
    assert mod.load_image_bytes(savable_img) == b"saved_png_bytes"
    monkeypatch.setattr(mod, "Image", None)
    monkeypatch.setitem(sys.modules, "PIL", MagicMock(Image=_DummyPILSentinel()))
    assert mod.load_image_bytes(savable_img) == b"saved_png_bytes"

    with pytest.raises(MultimodalProcessingError, match="target_size dimensions must be positive"):
        mod.process_image(b"", target_size=(0, 0))
    with pytest.raises(MultimodalProcessingError, match="target dimensions"):
        mod.process_image(b"", target_size=(224, 224), patch_size=15)

    from tests.conftest import MockPILImage

    monkeypatch.setattr(mod, "Image", MockPILImage)
    res_img = mod.process_image(b"fakebytes", target_size=(224, 224), patch_size=14)
    assert res_img["num_patches"] == 256
    assert res_img["shape"] == (3, 224, 224)

    import jax.numpy as jnp_real

    dummy_arr = jnp_real.zeros((3, 14, 14), dtype=jnp_real.float32)
    dummy_aud_arr = jnp_real.zeros(160, dtype=jnp_real.float32)
    with (
        patch.object(
            mod,
            "process_image",
            return_value={"pixel_values": dummy_arr, "patches": dummy_arr, "num_patches": 1, "shape": (3, 14, 14)},
        ),
        patch.object(
            mod,
            "process_audio",
            return_value={"audio_values": dummy_aud_arr, "spectrogram": dummy_aud_arr, "num_frames": 1, "sample_rate": 16000},
        ),
    ):
        p_out, pix_out, aud_out = mod.prepare_multimodal_inputs(
            "SELECT 1",
            {"image_path": "img.png", "audio_path": "aud.wav"},
            lambda arr: arr,
        )
        assert "<image>" in p_out
        assert "<audio>" in p_out
        assert pix_out is dummy_arr
        assert aud_out is dummy_aud_arr

        p_text, pix_none, aud_none = mod.prepare_multimodal_inputs("SELECT 1", {}, lambda arr: arr)
        assert p_text == "SELECT 1"
        assert pix_none is None
        assert aud_none is None

        p_pre, pix_pre, aud_pre = mod.prepare_multimodal_inputs(
            "SELECT 1",
            {"image_path": "img.png", "audio_path": "aud.wav", "pixel_values": dummy_arr, "audio_values": dummy_aud_arr},
            lambda arr: arr,
        )
        assert "<image>" in p_pre
        assert pix_pre is dummy_arr
        assert aud_pre is dummy_aud_arr

    class MockFailingPILImage:
        """Stub PIL Image that raises ValueError on open."""

        @staticmethod
        def open(_buf: io.BytesIO) -> NoReturn:
            """Raise ValueError."""
            raise ValueError("PIL error")

    monkeypatch.setattr(mod, "Image", MockFailingPILImage)
    with pytest.raises(MultimodalProcessingError, match="Corrupted or unreadable image:"):
        mod.process_image(b"fakebytes", target_size=(224, 224), patch_size=14)

    monkeypatch.setattr(mod, "Image", _DummyPILSentinel())
    with pytest.raises(MultimodalProcessingError, match=r"PIL\.Image\.open is not callable"):
        mod.process_image(b"fakebytes", target_size=(224, 224), patch_size=14)

    monkeypatch.setattr(mod, "jnp", None)
    with pytest.raises(MultimodalProcessingError, match="Corrupted or unreadable image:"):
        mod.process_image(b"bad", target_size=(224, 224), patch_size=14)
