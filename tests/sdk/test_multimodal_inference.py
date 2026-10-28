"""Unit and integration tests for multimodal inference, preprocessing, agent loops, and serving."""

from __future__ import annotations

import base64
import struct
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from gemma_4_sql.backends.common_multimodal import (
    format_multimodal_prompt,
    load_audio_bytes,
    load_image_bytes,
    process_audio,
)
from gemma_4_sql.backends.common_serve import GenerateRequest, create_common_app
from gemma_4_sql.sdk.agent import AgentContext, run_agentic_loop
from gemma_4_sql.sdk.evaluation import evaluate
from gemma_4_sql.sdk.inference import generate_sql
from gemma_4_sql.type_hints import ETLConfig, InferenceResultDict, JSONDict, JSONValue


def _create_synthetic_wav(duration_s: float = 0.1, sample_rate: int = 16000, stereo: bool = False) -> bytes:
    """Create a synthetic 16-bit PCM RIFF WAVE byte stream.

    Args:
        duration_s: Duration in seconds.
        sample_rate: Samples per second.
        stereo: Whether to produce two interleaved channels.

    Returns:
        Binary bytes of a valid RIFF WAVE file.

    """
    num_samples = int(duration_s * sample_rate)
    num_channels = 2 if stereo else 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align

    header = bytearray(b"RIFF")
    header.extend(struct.pack("<I", 36 + data_size))
    header.extend(b"WAVEfmt ")
    header.extend(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))

    # Single sine-like wave
    pcm_data = bytearray()
    for i in range(num_samples):
        val = int(16000 * 0.5 * (1 if (i % 20) < 10 else -1))
        pcm_data.extend(struct.pack("<h", val))
        if stereo:
            pcm_data.extend(struct.pack("<h", val))

    return bytes(header + pcm_data)


def test_load_image_bytes(tmp_path: Path) -> None:
    """Test loading image bytes across paths, base64 strings, and raw bytes."""
    raw = bytes.fromhex("89504e470d0a1a0a0000000d49484452")
    assert load_image_bytes(raw) == raw

    # File path
    img_file = tmp_path / "test.png"
    img_file.write_bytes(raw)
    assert load_image_bytes(img_file) == raw
    assert load_image_bytes(str(img_file)) == raw

    # Base64 data URI
    b64_str = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"
    assert load_image_bytes(data_uri) == raw

    # Long raw base64 string
    long_b64 = base64.b64encode(raw * 20).decode("utf-8")
    assert len(load_image_bytes(long_b64)) > 0

    # Error conditions
    with pytest.raises(ValueError, match="image_input cannot be None"):
        load_image_bytes(None)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image_bytes(tmp_path / "nonexistent.png")

    untyped_load_image: Callable[..., bytes] = load_image_bytes
    with pytest.raises(ValueError, match="Unsupported image input type"):
        untyped_load_image(12345)  # ty: ignore[invalid-argument-type]


def test_process_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test image resizing, normalization, and patch extraction."""
    import gemma_4_sql.backends.common_multimodal as cm
    from tests.conftest import MockPILImage

    monkeypatch.setattr(cm, "Image", MockPILImage)
    valid_png = b"fakebytes"

    res = cm.process_image(valid_png, target_size=(224, 224), patch_size=14)
    assert "pixel_values" in res
    assert "patches" in res
    assert res["num_patches"] == 256
    assert res["shape"] == (3, 224, 224)

    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        cm.process_image(b"fake", target_size=(0, 224))

    with pytest.raises(ValueError, match="target dimensions"):
        cm.process_image(b"fake", target_size=(224, 224), patch_size=15)


def test_load_audio_bytes(tmp_path: Path) -> None:
    """Test loading audio bytes across paths, base64 strings, and raw bytes."""
    wav = _create_synthetic_wav(0.05)
    assert load_audio_bytes(wav) == wav

    # File path
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(wav)
    assert load_audio_bytes(audio_file) == wav
    assert load_audio_bytes(str(audio_file)) == wav

    # Base64 data URI
    b64_str = base64.b64encode(wav).decode("utf-8")
    data_uri = f"data:audio/wav;base64,{b64_str}"
    assert load_audio_bytes(data_uri) == wav

    # Errors
    with pytest.raises(ValueError, match="audio_input cannot be None"):
        load_audio_bytes(None)

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio_bytes(tmp_path / "nonexistent.wav")

    untyped_load_audio: Callable[..., bytes] = load_audio_bytes
    with pytest.raises(ValueError, match="Unsupported audio input type"):
        untyped_load_audio(99999)  # ty: ignore[invalid-argument-type]


def test_process_audio() -> None:
    """Test processing audio waveforms and log-mel spectrogram extraction."""
    mono_wav = _create_synthetic_wav(0.1, sample_rate=16000, stereo=False)
    res_mono = process_audio(mono_wav, sample_rate=16000, n_mels=80)
    assert "audio_values" in res_mono
    assert "spectrogram" in res_mono
    assert res_mono["sample_rate"] == 16000
    assert res_mono["num_frames"] > 0

    # Stereo WAV test with resampling from 8000 to 16000
    stereo_wav = _create_synthetic_wav(0.1, sample_rate=8000, stereo=True)
    res_stereo = process_audio(stereo_wav, sample_rate=16000, n_mels=80)
    assert res_stereo["sample_rate"] == 16000

    # Raw float list
    res_list = process_audio([0.1, -0.2, 0.3, 0.4] * 100)
    assert res_list["num_frames"] >= 1

    # Validation errors
    with pytest.raises(ValueError, match="must be positive integers"):
        process_audio(mono_wav, sample_rate=0)


def test_parse_wav_no_data_chunk() -> None:
    """Test WAV parsing fallback when data chunk is absent."""
    wav_no_data = bytearray(b"RIFF")
    wav_no_data.extend(struct.pack("<I", 36))
    wav_no_data.extend(b"WAVEfmt ")
    wav_no_data.extend(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
    wav_no_data.extend(b"JUNK")
    wav_no_data.extend(struct.pack("<I", 4))
    wav_no_data.extend(b"test")
    res = process_audio(bytes(wav_no_data))
    assert res["sample_rate"] == 16000
    assert len(res["audio_values"]) > 0


def test_format_multimodal_prompt() -> None:
    """Test formatting prompt with multimodal placeholder tokens and masks."""
    # Text only
    text_res = format_multimodal_prompt("Show all active users")
    assert text_res.get("prompt") == "Show all active users"
    assert text_res.get("modality") == "text"
    assert text_res.get("image_token_mask") is None
    assert text_res.get("audio_token_mask") is None

    # Vision only
    vis_res = format_multimodal_prompt("Count departments", has_image=True)
    assert vis_res["prompt"].startswith("<image>")
    assert vis_res["modality"] == "vision"
    vis_mask = vis_res["image_token_mask"]
    assert vis_mask is not None
    assert vis_mask[0] is True
    assert vis_mask[1] is False

    # Audio only
    aud_res = format_multimodal_prompt("Calculate sum", has_audio=True)
    assert aud_res["prompt"].startswith("<audio>")
    assert aud_res["modality"] == "audio"
    aud_mask = aud_res["audio_token_mask"]
    assert aud_mask is not None
    assert aud_mask[0] is True

    # Multimodal (vision + audio)
    mm_res = format_multimodal_prompt("Find customer with id 10", has_image=True, has_audio=True)
    assert "<image>" in mm_res["prompt"]
    assert "<audio>" in mm_res["prompt"]
    assert mm_res["modality"] == "multimodal"

    # Already has tokens in prompt
    existing_res = format_multimodal_prompt("<image> Query schema", has_image=True)
    assert existing_res["prompt"].count("<image>") == 1

    with pytest.raises(ValueError, match=r"prompt cannot be empty\."):
        format_multimodal_prompt("")


def test_sdk_inference_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SDK generate and generate_sql functions with multimodal arguments."""
    captured: JSONDict = {}

    class MockBackend:
        """Mock backend implementation for inference."""

        @staticmethod
        def generate_sql(model_name: str, prompt: str, **kwargs: JSONValue) -> InferenceResultDict:
            """Capture arguments."""
            captured["model_name"] = model_name
            captured["prompt"] = prompt
            captured.update(kwargs)
            return {
                "backend": "pytorch",
                "model": model_name,
                "prompt": prompt,
                "sql": "SELECT 1",
                "beam_width": 3,
                "confidence_score": 0.99,
                "status": "success",
            }

    monkeypatch.setattr("gemma_4_sql.sdk.inference.get_backend", lambda _b: MockBackend())  # pyright: ignore[reportUnknownLambdaType]

    res = generate_sql(
        "test-model",
        "Select users",
        backend="pytorch",
        image_path="/path/to/diagram.png",
        audio_path="/path/to/voice.wav",
        modality="multimodal",
    )
    assert res["sql"] == "SELECT 1"
    assert captured["image_path"] == "/path/to/diagram.png"
    assert captured["audio_path"] == "/path/to/voice.wav"
    assert captured["modality"] == "multimodal"


def test_sdk_agent_multimodal_preservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that self-correction agent loop preserves multimodal context across retry attempts."""
    attempts_seen: list[str] = []

    class MockFailingThenSuccessBackend:
        """Mock backend returning error on first attempt and success on second."""

        @staticmethod
        def generate_sql(model_name: str, prompt: str, **kwargs: JSONValue) -> InferenceResultDict:
            """Record prompt and kwargs."""
            attempts_seen.append(prompt)
            assert kwargs.get("image_path") == "schema.png"
            assert kwargs.get("modality") == "vision"
            if len(attempts_seen) == 1:
                return {
                    "backend": "jax",
                    "model": model_name,
                    "prompt": prompt,
                    "sql": "SELECT invalid_col FROM users",
                    "beam_width": 3,
                    "confidence_score": 0.9,
                    "status": "success",
                }
            return {
                "backend": "jax",
                "model": model_name,
                "prompt": prompt,
                "sql": "SELECT id FROM users",
                "beam_width": 3,
                "confidence_score": 0.9,
                "status": "success",
            }

    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: MockFailingThenSuccessBackend())  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(
        db_path=":memory:",
        ddl="CREATE TABLE users (id INT);",
        max_retries=3,
        image_path="schema.png",
        modality="vision",
    )
    result = run_agentic_loop("test-model", "List users", backend="jax", context=ctx)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result["attempts"] == 2
    assert "<image>" in attempts_seen[0]
    assert "<image>" in attempts_seen[1]
    assert "Previous attempt failed" in attempts_seen[1]


def test_sdk_evaluation_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluation pipeline forwarding multimodal options to backend."""

    class MockEvalBackend:
        """Mock evaluation backend."""

        @staticmethod
        def build_dataloader(
            config: ETLConfig,
        ) -> dict[str, JSONValue | list[dict[str, list[list[int]] | list[list[float]]]]]:
            """Return mock dataloader with sample batch."""
            return {
                "backend": "jax",
                "dataset": config.dataset_name,
                "split": config.split,
                "batch_size": config.batch_size,
                "distributed": config.distributed,
                "status": "loaded",
                "loader": [{"inputs": [[1, 2]], "targets": [[3, 4]], "pixel_values": [[0.0]]}],
            }

        @staticmethod
        def generate_sql(model_name: str, prompt: str, **kwargs: JSONValue) -> InferenceResultDict:
            """Return mock generation."""
            assert "pixel_values" in kwargs or "image_path" in kwargs
            assert kwargs.get("modality") == "vision"
            return {
                "backend": "jax",
                "model": model_name,
                "prompt": prompt,
                "sql": "SELECT * FROM t",
                "beam_width": 3,
                "confidence_score": 0.95,
                "status": "success",
            }

    monkeypatch.setattr("gemma_4_sql.sdk.evaluation.get_backend", lambda _b: MockEvalBackend())  # pyright: ignore[reportUnknownLambdaType]

    res = evaluate(
        "test-model",
        "spider_erd",
        backend="jax",
        image_path="/path/to/erd.png",
        modality="vision",
    )
    assert res["status"] == "completed"
    assert res.get("modality") == "vision"
    assert res.get("image_path") == "/path/to/erd.png"


def test_common_serve_multimodal() -> None:
    """Test FastAPI continuous batching server with multimodal attachments."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = create_common_app(
        backend_name="pytorch",
        model_name="test-model",
        generate_logic=lambda prompt: f"SELECT * FROM test WHERE p='{prompt}'",  # pyright: ignore[reportUnknownLambdaType]
    )

    req_payload: JSONDict = {
        "prompt": "Find all records",
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "modality": "vision",
    }
    validated = GenerateRequest.from_dict(req_payload)
    assert validated.modality == "vision"
    assert validated.image_base64 is not None

    assert isinstance(app, FastAPI)
    with TestClient(app) as client:
        resp = client.post("/generate", json=req_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "sql" in body
        assert body["modality"] == "vision"
        assert "<image>" in body["sql"]


def test_common_multimodal_pure_python_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test image and audio processing when jax.numpy and PIL are not available."""
    import gemma_4_sql.backends.common_multimodal as cm

    orig_jnp = cm.jnp
    orig_image = cm.Image

    monkeypatch.setattr(cm, "jnp", None)
    monkeypatch.setattr(cm, "Image", None)
    monkeypatch.setattr(cm, "optional_import", lambda _name: None)  # pyright: ignore[reportUnknownLambdaType]

    # Process image with pure python fallback
    raw_img = b"fake_image_bytes"
    with pytest.raises(ValueError, match="Corrupted or unreadable image:"):
        cm.process_image(raw_img, target_size=(28, 28), patch_size=14)

    # Process audio with pure python fallback
    raw_audio = b"fake_audio_bytes" * 50
    res_audio = cm.process_audio(raw_audio, sample_rate=16000, n_mels=10)
    assert "audio_values" in res_audio
    assert "spectrogram" in res_audio
    assert len(res_audio["spectrogram"][0]) == 10

    monkeypatch.setattr(cm, "jnp", orig_jnp)
    monkeypatch.setattr(cm, "Image", orig_image)


def test_multimodal_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test edge cases in loading bytes and corrupted WAV headers."""
    import gemma_4_sql.backends.common_multimodal as cm

    # Invalid base64 in long string
    invalid_b64 = "!!!" * 100
    with pytest.raises(FileNotFoundError):
        cm.load_image_bytes(invalid_b64)

    # Data URI without comma
    data_raw = b"imagedata"
    b64_val = base64.b64encode(data_raw).decode("utf-8")
    assert cm.load_image_bytes(f"data:{b64_val}") == data_raw
    assert cm.load_audio_bytes(f"data:{b64_val}") == data_raw

    # Audio input as tuple
    res_tuple = cm.process_audio((0.1, -0.1, 0.2))
    assert res_tuple["num_frames"] >= 1

    # WAV header with unsupported bits per sample (e.g. 24 bit)
    valid_wav = bytearray(_create_synthetic_wav(0.02))
    valid_wav[34:36] = struct.pack("<H", 24)
    samples, _rate = cm._parse_wav_samples(bytes(valid_wav))
    assert len(samples) > 0

    # WAV with intermediate JUNK chunk before data
    junk_chunk = b"JUNK\x04\x00\x00\x001234"
    wav_with_junk = bytes(valid_wav[:36] + junk_chunk + valid_wav[36:])
    samples_j, _ = cm._parse_wav_samples(wav_with_junk)
    assert len(samples_j) > 0

    # Image available with MockPILImage and jnp None vs available
    from tests.conftest import MockPILImage

    png_header = bytes.fromhex("89504e470d0a1a0a")
    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr(cm, "Image", MockPILImage)
        res_img_with_pil = cm.process_image(png_header, target_size=(14, 14), patch_size=14)
        assert res_img_with_pil["num_patches"] == 1
        monkeypatch_ctx.setattr(cm, "jnp", None)
        with pytest.raises(ValueError, match="Corrupted or unreadable image:"):
            cm.process_image(png_header, target_size=(14, 14), patch_size=14)

    # Corrupted WAV header triggering struct.error
    corrupt_wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40
    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr("struct.unpack", lambda *_a, **_k: (_ for _ in ()).throw(struct.error("Corrupted WAV unpack")))  # pyright: ignore[reportUnknownLambdaType]
        with pytest.raises(ValueError, match="Corrupted WAV fmt chunk header"):
            cm._parse_wav_samples(corrupt_wav)

    # PIL open raising OSError handled gracefully
    class _MockFailingPIL:
        """Mock PIL Image raising OSError on open."""

        @staticmethod
        def open(_buf: bytes | str | Path) -> NoReturn:
            """Raise OSError to simulate corrupted image open."""
            raise OSError("Corrupted image")

    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr(cm, "Image", _MockFailingPIL)
        with pytest.raises(ValueError, match="Corrupted or unreadable image:"):
            cm.process_image(b"corrupted", target_size=(14, 14), patch_size=14)

    # Audio file not found
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        cm.load_audio_bytes(Path("non_existent_audio_path_xyz.wav"))

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        cm.load_audio_bytes(Path("x" * 300))

    # Mono WAV direct parse
    mono_direct = _create_synthetic_wav(0.02, stereo=False)
    samples_m, _ = cm._parse_wav_samples(mono_direct)
    assert len(samples_m) > 0

    # Stereo WAV direct parse
    stereo_direct = _create_synthetic_wav(0.02, stereo=True)
    samples_s, _ = cm._parse_wav_samples(stereo_direct)
    assert len(samples_s) > 0

    # Audio input as jax.Array
    import jax.numpy as jnp

    arr_audio = jnp.array([0.05, -0.05, 0.1, -0.1], dtype=jnp.float32)
    res_np = cm.process_audio(arr_audio)
    assert res_np["num_frames"] >= 1

    # WAV container with intermediate chunk before data chunk
    chunk_wav = b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00JUNK\x04\x00\x00\x001234data\x04\x00\x00\x00\x00\x00\x00\x00"
    samples_chunk, _ = cm._parse_wav_samples(chunk_wav)
    assert len(samples_chunk) > 0

    # Empty bytes raises MultimodalProcessingError (a ValueError subclass)
    with pytest.raises(ValueError, match="Audio byte payload cannot be empty"):
        cm._parse_wav_samples(b"")

    # Corrupt image bytes triggering PIL exception
    with pytest.raises(ValueError):
        cm.process_image(b"NOT_A_VALID_IMAGE_BYTES_PAYLOAD", target_size=(14, 14), patch_size=14)

    # Missing jnp raises MultimodalProcessingError
    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr(cm, "jnp", None)
        with pytest.raises(ValueError):
            cm.process_image(b"NOT_A_VALID_IMAGE", target_size=(14, 14), patch_size=14)

    # Truncated WAV header raises MultimodalProcessingError
    truncated_wav = b"RIFF\x20\x00\x00\x00WAVEfmt "
    with pytest.raises(ValueError, match="Truncated or malformed RIFF/WAVE header"):
        cm._parse_wav_samples(truncated_wav)
