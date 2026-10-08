"""Tests for common multimodal utilities."""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.common_multimodal as mod
from gemma_4_sql.backends.common_multimodal import (
    _parse_wav_samples,
    format_multimodal_prompt,
    load_audio_bytes,
    load_image_bytes,
    process_audio,
    process_image,
)


def test_load_image_bytes_none():
    """Docstring for test_load_image_bytes_none."""
    with pytest.raises(ValueError, match="image_input cannot be None."):
        load_image_bytes(None)


def test_load_image_bytes_bytes():
    """Docstring for test_load_image_bytes_bytes."""
    assert load_image_bytes(b"123") == b"123"


def test_load_image_bytes_path(tmp_path):
    """Docstring for test_load_image_bytes_path."""
    p = tmp_path / "img.png"
    p.write_bytes(b"fake_image")
    assert load_image_bytes(p) == b"fake_image"
    assert load_image_bytes(str(p)) == b"fake_image"


def test_load_image_bytes_path_not_found():
    """Docstring for test_load_image_bytes_path_not_found."""
    with pytest.raises(FileNotFoundError):
        load_image_bytes("nonexistent_path.png")


def test_load_image_bytes_base64():
    """Docstring for test_load_image_bytes_base64."""
    long_bytes = b"h" * 300
    long_b64 = base64.b64encode(long_bytes).decode("utf-8")
    assert load_image_bytes(long_b64) == long_bytes
    # Also test an invalid long base64 string
    with pytest.raises(FileNotFoundError):
        load_image_bytes("*" * 300)

    # also test PIL fallback failure
    class BadImg:
        """Docstring for BadImg."""

        def save(self, buf, format):
            """Docstring for save."""
            raise OSError("err")

    with patch("gemma_4_sql.backends.common_multimodal.Image", MagicMock()), pytest.raises(OSError):
        load_image_bytes(BadImg())


def test_load_image_bytes_data_uri():
    """Docstring for test_load_image_bytes_data_uri."""
    b64 = base64.b64encode(b"hello").decode("utf-8")
    assert load_image_bytes(f"data:image/png;base64,{b64}") == b"hello"
    assert load_image_bytes(f"data:{b64}") == b"hello"


def test_load_image_bytes_pil_image():
    """Docstring for test_load_image_bytes_pil_image."""
    mock_img = MagicMock()
    mock_img.save = MagicMock(side_effect=lambda buf, format: buf.write(b"pil_data"))
    assert load_image_bytes(mock_img) == b"pil_data"


def test_load_image_bytes_unsupported():
    """Docstring for test_load_image_bytes_unsupported."""
    with pytest.raises(ValueError, match="Unsupported image input type:"):
        load_image_bytes(123)


def test_process_image_invalid_sizes():
    """Docstring for test_process_image_invalid_sizes."""
    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        process_image(b"fake", target_size=(-1, 10))
    with pytest.raises(ValueError, match="must be divisible by patch_size"):
        process_image(b"fake", target_size=(224, 224), patch_size=15)


def test_process_image_no_np_fallback():
    """Docstring for test_process_image_no_np_fallback."""
    with patch("gemma_4_sql.backends.common_multimodal.np", None), patch("gemma_4_sql.backends.common_multimodal.Image", None):
        res = process_image(b"fake", target_size=(14, 14), patch_size=14)
        assert res["num_patches"] == 1
        assert len(res["pixel_values"]) == 3
        assert len(res["pixel_values"][0]) == 14
        assert len(res["patches"]) == 1
        assert res["shape"] == (3, 14, 14)


def test_process_image_with_np_fallback():
    """Docstring for test_process_image_with_np_fallback."""
    import numpy as np

    with patch("gemma_4_sql.backends.common_multimodal.np", np), patch("gemma_4_sql.backends.common_multimodal.Image", None):
        res = process_image(b"fake", target_size=(14, 14), patch_size=14)
        assert res["num_patches"] == 1
        assert res["pixel_values"].shape == (3, 14, 14)
        assert res["patches"].shape == (1, 3 * 14 * 14)


def test_process_image_with_pil():
    """Docstring for test_process_image_with_pil."""
    import numpy as np
    from PIL import Image

    img = Image.new("RGB", (28, 28), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with patch("gemma_4_sql.backends.common_multimodal.np", np), patch("gemma_4_sql.backends.common_multimodal.Image", Image):
        res = process_image(img_bytes, target_size=(28, 28), patch_size=14)
        assert res["num_patches"] == 4
        assert res["pixel_values"].shape == (3, 28, 28)


def test_load_audio_bytes_none():
    """Docstring for test_load_audio_bytes_none."""
    with pytest.raises(ValueError):
        load_audio_bytes(None)


def test_load_audio_bytes_bytes():
    """Docstring for test_load_audio_bytes_bytes."""
    assert load_audio_bytes(b"audio") == b"audio"


def test_load_audio_bytes_path(tmp_path):
    """Docstring for test_load_audio_bytes_path."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"audio_content")
    assert load_audio_bytes(p) == b"audio_content"
    assert load_audio_bytes(str(p)) == b"audio_content"


def test_load_audio_bytes_path_not_found():
    """Docstring for test_load_audio_bytes_path_not_found."""
    with pytest.raises(FileNotFoundError):
        load_audio_bytes("nonexistent_path.wav")


def test_load_audio_bytes_data_uri():
    """Docstring for test_load_audio_bytes_data_uri."""
    b64 = base64.b64encode(b"audio").decode("utf-8")
    assert load_audio_bytes(f"data:audio/wav;base64,{b64}") == b"audio"
    assert load_audio_bytes(f"data:{b64}") == b"audio"


def test_load_audio_bytes_unsupported():
    """Docstring for test_load_audio_bytes_unsupported."""
    with pytest.raises(ValueError, match="Unsupported audio input type:"):
        load_audio_bytes(123)


def test_parse_wav_samples_fallback():
    """Docstring for test_parse_wav_samples_fallback."""
    # Provide bad bytes, it should fallback to synthetic
    res, rate = _parse_wav_samples(b"fake_wav")
    assert rate == 16000
    assert len(res) > 0


def test_parse_wav_samples_valid():
    """Docstring for test_parse_wav_samples_valid."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    wav += b"data" + struct.pack("<I", 4) + struct.pack("<h", 0) + struct.pack("<h", 32767)
    res, rate = _parse_wav_samples(wav)
    assert rate == 16000
    assert len(res) == 2


def test_parse_wav_samples_valid_stereo():
    """Docstring for test_parse_wav_samples_valid_stereo."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 2) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 64000) + struct.pack("<H", 4) + struct.pack("<H", 16)
    wav += b"data" + struct.pack("<I", 4) + struct.pack("<h", 0) + struct.pack("<h", 32767)
    res, rate = _parse_wav_samples(wav)
    assert rate == 16000
    assert len(res) == 1


def test_process_audio_invalid():
    """Docstring for test_process_audio_invalid."""
    with pytest.raises(ValueError):
        process_audio(b"fake", sample_rate=0)
    with pytest.raises(ValueError):
        process_audio(b"fake", n_mels=0)


def test_process_audio_list():
    """Docstring for test_process_audio_list."""
    import numpy as np

    with patch("gemma_4_sql.backends.common_multimodal.np", np):
        res = process_audio([0.1, 0.2, 0.3], sample_rate=16000)
        assert res["sample_rate"] == 16000
        assert res["num_frames"] == 1
        assert "audio_values" in res


def test_process_audio_resample():
    """Docstring for test_process_audio_resample."""
    import numpy as np

    with patch("gemma_4_sql.backends.common_multimodal.np", np):
        # 8000 Hz list input to target 16000 Hz
        process_audio([0.0, 1.0], sample_rate=16000)
        with patch("gemma_4_sql.backends.common_multimodal._parse_wav_samples", return_value=([0.0, 1.0], 8000)):
            res2 = process_audio(b"fake", sample_rate=16000)
            assert res2["sample_rate"] == 16000


def test_process_audio_np():
    """Docstring for test_process_audio_np."""
    import numpy as np

    mock_input = np.array([0.0] * 16000, dtype=np.float32)
    with patch("gemma_4_sql.backends.common_multimodal.np", np):
        res = process_audio(mock_input, sample_rate=16000)
        assert res["sample_rate"] == 16000
        assert res["spectrogram"].shape[1] == 80


def test_process_audio_no_np():
    """Docstring for test_process_audio_no_np."""
    with patch("gemma_4_sql.backends.common_multimodal.np", None):
        res = process_audio([0.0] * 16000, sample_rate=16000)
        assert res["sample_rate"] == 16000
        assert len(res["spectrogram"]) > 0


def test_format_multimodal_prompt():
    """Docstring for test_format_multimodal_prompt."""
    with pytest.raises(ValueError):
        format_multimodal_prompt("")

    res = format_multimodal_prompt("hello")
    assert res["prompt"] == "hello"
    assert res["modality"] == "text"

    res = format_multimodal_prompt("hello <image>", has_image=True)
    assert res["prompt"] == "hello <image>"
    assert res["modality"] == "vision"
    assert res["image_token_mask"] == [False, True]

    res = format_multimodal_prompt("hello", has_audio=True)
    assert res["prompt"] == "<audio> hello"
    assert res["modality"] == "audio"
    assert res["audio_token_mask"] == [True, False]

    res = format_multimodal_prompt("hello", has_image=True, has_audio=True)
    assert res["modality"] == "multimodal"


def test_process_image_pil_parse_error():
    """Docstring for test_process_image_pil_parse_error."""
    from PIL import Image

    with patch("gemma_4_sql.backends.common_multimodal.np", None), patch("gemma_4_sql.backends.common_multimodal.Image", Image):
        # b"fake" is not a valid image, will cause PIL to raise UnidentifiedImageError (OSError)
        res = process_image(b"fake", target_size=(14, 14), patch_size=14)
        assert res["num_patches"] == 1
        assert len(res["pixel_values"]) == 3
        assert len(res["pixel_values"][0]) == 14


def test_load_audio_bytes_path_os_error():
    """Docstring for test_load_audio_bytes_path_os_error."""
    # To cause an OSError in Path(...).is_file(), we can mock it
    with patch("gemma_4_sql.backends.common_multimodal.Path.is_file", side_effect=OSError), pytest.raises(FileNotFoundError):
        load_audio_bytes("fake_audio_path.wav")


def test_parse_wav_samples_valid_mono():
    """Docstring for test_parse_wav_samples_valid_mono."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    wav += b"data" + struct.pack("<I", 4) + struct.pack("<h", 32767) + struct.pack("<h", 32767)
    res, rate = _parse_wav_samples(wav)
    assert rate == 16000
    assert len(res) == 2


def test_parse_wav_samples_struct_error():
    """Docstring for test_parse_wav_samples_struct_error."""
    import struct

    # Malformed chunk to cause struct error
    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    wav += b"data" + struct.pack("<I", 3) + b"123"  # odd number of bytes will fail struct.unpack for shorts
    _res, rate = _parse_wav_samples(wav)
    assert rate == 16000


def test_parse_wav_samples_not_data_chunk():
    """Docstring for test_parse_wav_samples_not_data_chunk."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    wav += b"junk" + struct.pack("<I", 4) + b"1234"
    wav += b"data" + struct.pack("<I", 2) + struct.pack("<h", 32767)
    res, _rate = _parse_wav_samples(wav)
    assert len(res) == 1


def test_parse_wav_samples_8bit():
    """Docstring for test_parse_wav_samples_8bit."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 16000) + struct.pack("<H", 1) + struct.pack("<H", 8)
    wav += b"data" + struct.pack("<I", 2) + struct.pack("<B", 255) + struct.pack("<B", 0)
    res, _rate = _parse_wav_samples(wav)
    # the code says if bits_per_sample == 16, so 8-bit should break the loop and fallback
    assert len(res) == 46  # length of the header since it falls back to parsing raw bytes


def test_parse_wav_samples_empty():
    """Docstring for test_parse_wav_samples_empty."""
    res, rate = _parse_wav_samples(b"")
    assert res == [0.0] * 1600
    assert rate == 16000


def test_parse_wav_samples_no_data():
    """Docstring for test_parse_wav_samples_no_data."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    # add a big chunk that is not "data"
    wav += b"junk" + struct.pack("<I", 100) + b"x" * 100
    _res, rate = _parse_wav_samples(wav)
    assert rate == 16000


def test_parse_wav_samples_struct_err_real():
    """Docstring for test_parse_wav_samples_struct_err_real."""
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    # data chunk says size is 100 but we only provide 2 bytes
    wav += b"data" + struct.pack("<I", 100) + b"xy"
    _res, rate = _parse_wav_samples(wav)
    assert rate == 16000


def test_process_image_no_np_fallback_direct():
    """Docstring for test_process_image_no_np_fallback_direct."""
    mod.np = None
    mod.Image = None
    res = mod.process_image(b"fake", target_size=(14, 14), patch_size=14)
    assert res["num_patches"] == 1
    assert len(res["pixel_values"]) == 3


def test_parse_wav_samples_struct_err_mock():
    """Docstring for test_parse_wav_samples_struct_err_mock."""
    import struct
    from unittest.mock import patch

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 16)
    wav += b"data" + struct.pack("<I", 2) + b"xy"
    with patch("gemma_4_sql.backends.common_multimodal.struct.unpack", side_effect=struct.error("mock error")):
        _res, rate = _parse_wav_samples(wav)
        assert rate == 16000


def test_process_image_no_np_fallback_direct_2():
    """Docstring for test_process_image_no_np_fallback_direct_2."""
    import gemma_4_sql.backends.common_multimodal as mod

    old_np = mod.np
    old_img = mod.Image
    mod.np = None
    mod.Image = None
    try:
        res = mod.process_image(b"fake", target_size=(14, 14), patch_size=14)
        assert res["num_patches"] == 1
    finally:
        mod.np = old_np
        mod.Image = old_img


def test_parse_wav_break_coverage():
    """Docstring for test_parse_wav_break_coverage."""
    # To hit line 259 "break", bits_per_sample must NOT be 16.
    import struct

    wav = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
    wav += struct.pack("<H", 1) + struct.pack("<H", 1) + struct.pack("<I", 16000)
    wav += struct.pack("<I", 32000) + struct.pack("<H", 2) + struct.pack("<H", 8)  # 8 bit
    wav += b"data" + struct.pack("<I", 2) + b"xy"
    _res, rate = _parse_wav_samples(wav)
    assert rate == 16000


def test_process_image_pil_valid_np_none():
    """Docstring for test_process_image_pil_valid_np_none."""
    from PIL import Image

    import gemma_4_sql.backends.common_multimodal as mod

    img = Image.new("RGB", (28, 28), color="red")
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    old_np = mod.np
    mod.np = None
    try:
        res = mod.process_image(img_bytes, target_size=(14, 14), patch_size=14)
        assert res["num_patches"] == 1
    finally:
        mod.np = old_np
