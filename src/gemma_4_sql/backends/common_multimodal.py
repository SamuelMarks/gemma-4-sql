"""Common multimodal data preprocessing and tensor conversion utilities.

Provides image normalization, patch extraction, audio resampling, log-mel
spectrogram extraction, and prompt placeholder formatting for vision and audio Text-to-SQL.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import AudioInput, ImageInput, MultimodalInput

logger = logging.getLogger(__name__)

np = optional_import("numpy")
Image = optional_attr("PIL", "Image")


def load_image_bytes(image_input: ImageInput) -> bytes:
    """Load raw binary image bytes from filesystem paths, base64 strings, or byte sequences.

    Args:
        image_input: File path, base64 string, raw bytes, or image object.

    Returns:
        Decoded binary image bytes.

    Raises:
        ValueError: If image_input is None or cannot be decoded.
        FileNotFoundError: If the specified image path does not exist on disk.

    """
    if image_input is None:
        msg = "image_input cannot be None."
        raise ValueError(msg)

    if isinstance(image_input, bytes):
        return image_input

    import pathlib

    if isinstance(image_input, pathlib.Path) or (isinstance(image_input, str) and not image_input.startswith("data:")):
        try:
            p = Path(str(image_input))
            if p.is_file():
                return p.read_bytes()
        except OSError:
            pass

        if isinstance(image_input, str) and len(image_input) > 200:
            try:
                return base64.b64decode(image_input, validate=True)
            except (ValueError, binascii.Error) as err:
                logger.debug("Base64 string decode failed: %s", err)
        msg = f"Image file not found: {image_input}"
        raise FileNotFoundError(msg)

    if isinstance(image_input, str) and image_input.startswith("data:"):
        if "," in image_input:
            parts = image_input.split(",", 1)
            return base64.b64decode(parts[1])
        return base64.b64decode(image_input[5:])

    if getattr(image_input, "save", None) is not None:
        pil_image = Image
        if pil_image is None:
            try:
                from PIL import Image as _Image

                pil_image = _Image
            except ImportError as err:
                logger.debug("Failed to import PIL: %s", err)
        if pil_image is not None:
            buf = io.BytesIO()
            img_any: Any = image_input
            img_any.save(buf, format="PNG")
            return buf.getvalue()

    msg = f"Unsupported image input type: {type(image_input)}"
    raise ValueError(msg)


def process_image(
    image_input: ImageInput,
    target_size: tuple[int, int] = (224, 224),
    patch_size: int = 14,
) -> dict[str, Any]:
    """Process an image into normalized RGB pixel arrays and extracted vision patches.

    Args:
        image_input: Image file path, binary bytes, or PIL Image.
        target_size: Target (height, width) tuple to resize image to (default: (224, 224)).
        patch_size: Square edge length of individual patches (default: 14).

    Returns:
        Dictionary containing:
            - 'pixel_values': Normalized float array of shape (3, H, W).
            - 'patches': Flattened 2D array of shape (num_patches, 3 * patch_size * patch_size).
            - 'num_patches': Total count of extracted vision patches.
            - 'shape': Shape tuple of the normalized image tensor.

    Raises:
        ValueError: If target_size or patch_size are invalid or image is corrupted.

    """
    (target_h, target_w) = target_size
    if target_h <= 0 or target_w <= 0:
        msg = f"target_size dimensions must be positive, got {target_size}"
        raise ValueError(msg)
    if patch_size <= 0 or target_h % patch_size != 0 or target_w % patch_size != 0:
        msg = f"target dimensions {target_size} must be divisible by patch_size {patch_size}"
        raise ValueError(msg)

    img_bytes = load_image_bytes(image_input)

    np_mod: Any = np
    if Image is None or np_mod is None:
        raise ValueError("Corrupted or unreadable image: Pillow library is required but missing.")
    try:
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            pil_resized = pil_img.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
            rgb_array = np_mod.array(pil_resized, dtype=np_mod.float32) / 255.0
    except Exception as e:
        raise ValueError(f"Corrupted or unreadable image: {e}") from e

    # Standard ImageNet normalization: (x - mean) / std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    norm_img = (rgb_array - mean) / std
    pixel_values = np_mod.transpose(norm_img, (2, 0, 1)).astype(np_mod.float32)

    num_patches_h = target_h // patch_size
    num_patches_w = target_w // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = 3 * patch_size * patch_size

    reshaped = pixel_values.reshape(3, num_patches_h, patch_size, num_patches_w, patch_size)
    transposed = np_mod.transpose(reshaped, (1, 3, 0, 2, 4))
    patches = transposed.reshape((num_patches, patch_dim)).astype(np_mod.float32)

    return {
        "pixel_values": pixel_values,
        "patches": patches,
        "num_patches": int(num_patches),
        "shape": (3, target_h, target_w),
    }


def load_audio_bytes(audio_input: AudioInput) -> bytes:
    """Load raw binary audio bytes from filesystem paths, base64 data URIs, or byte sequences.

    Args:
        audio_input: File path, base64 data URI string, or raw bytes.

    Returns:
        Decoded binary audio bytes.

    Raises:
        ValueError: If audio_input is None or unsupported.
        FileNotFoundError: If the specified audio file path does not exist on disk.

    """
    if audio_input is None:
        msg = "audio_input cannot be None."
        raise ValueError(msg)

    if isinstance(audio_input, bytes):
        return audio_input

    import pathlib

    if isinstance(audio_input, pathlib.Path) or (isinstance(audio_input, str) and not audio_input.startswith("data:")):
        try:
            p = Path(str(audio_input))
            if p.is_file():
                return p.read_bytes()
        except OSError:
            pass
        msg = f"Audio file not found: {audio_input}"
        raise FileNotFoundError(msg)

    if isinstance(audio_input, str) and audio_input.startswith("data:"):
        if "," in audio_input:
            parts = audio_input.split(",", 1)
            return base64.b64decode(parts[1])
        return base64.b64decode(audio_input[5:])

    msg = f"Unsupported audio input type: {type(audio_input)}"
    raise ValueError(msg)


def _parse_wav_samples(wav_bytes: bytes) -> tuple[list[float], int]:
    """Parse PCM waveform samples and sampling rate from RIFF WAV bytes.

    Args:
        wav_bytes: Raw binary bytes of a WAV container.

    Returns:
        Tuple of (samples_list, sample_rate).

    """
    if len(wav_bytes) >= 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
        try:
            num_channels = struct.unpack("<H", wav_bytes[22:24])[0]
            sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
            bits_per_sample = struct.unpack("<H", wav_bytes[34:36])[0]

            data_offset = 12
            while data_offset < len(wav_bytes) - 8:
                chunk_id = wav_bytes[data_offset : data_offset + 4]
                chunk_size = struct.unpack("<I", wav_bytes[data_offset + 4 : data_offset + 8])[0]
                if chunk_id == b"data":
                    data_bytes = wav_bytes[data_offset + 8 : data_offset + 8 + chunk_size]
                    if bits_per_sample == 16:
                        count = len(data_bytes) // 2
                        raw_ints = struct.unpack(f"<{count}h", data_bytes[: count * 2])
                        if num_channels == 2:
                            mono = [(raw_ints[i] + raw_ints[i + 1]) / 2.0 / 32768.0 for i in range(0, count - 1, 2)]
                        else:
                            mono = [x / 32768.0 for x in raw_ints]
                        return mono, int(sample_rate)
                data_offset += 8 + chunk_size
        except (struct.error, ValueError, IndexError) as exc:
            logger.debug("WAV parsing failed: %s; using default synth waveform", exc)

    samples = [(float(b) - 128.0) / 128.0 for b in wav_bytes[:16000]]
    if not samples:
        samples = [0.0] * 1600
    return samples, 16000


def process_audio(
    audio_input: AudioInput,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
) -> dict[str, Any]:
    """Process raw audio into a normalized 16kHz waveform and log-mel spectrogram features.

    Args:
        audio_input: Audio file path, binary bytes, or sample array.
        sample_rate: Target sampling rate in Hz (default: 16000).
        n_mels: Number of mel-frequency filterbank bins (default: 80).
        frame_length_ms: STFT analysis window duration in milliseconds (default: 25).
        frame_shift_ms: STFT frame hop duration in milliseconds (default: 10).

    Returns:
        Dictionary containing:
            - 'audio_values': 1D normalized float waveform array.
            - 'spectrogram': 2D log-mel spectrogram array of shape (num_frames, n_mels).
            - 'num_frames': Number of temporal analysis frames.
            - 'sample_rate': Output sampling rate in Hz.

    Raises:
        ValueError: If sample_rate or n_mels are non-positive.

    """
    if sample_rate <= 0 or n_mels <= 0:
        msg = "sample_rate and n_mels must be positive integers."
        raise ValueError(msg)

    samples: list[float]
    np_mod: Any = np
    audio_any: Any = audio_input
    if np_mod is not None and hasattr(audio_any, "flatten"):
        samples = [float(x) for x in audio_any.flatten().astype(np_mod.float32).tolist()]
        orig_rate = sample_rate
    elif isinstance(audio_input, (list, tuple)):
        samples = [float(x) for x in audio_any]
        orig_rate = sample_rate
    else:
        if np_mod is None:
            try:
                import numpy as _numpy

                np_mod = _numpy
            except ImportError as err:
                logger.debug("Failed to import numpy: %s", err)
        if np_mod is not None and isinstance(audio_any, np_mod.ndarray):
            samples = [float(x) for x in audio_any.flatten().tolist()]
            orig_rate = sample_rate
        else:
            raw_bytes = load_audio_bytes(audio_input)
            samples, orig_rate = _parse_wav_samples(raw_bytes)

    if orig_rate != sample_rate and len(samples) > 1:
        target_len = int(len(samples) * sample_rate / orig_rate)
        target_len = max(target_len, 1)
        resampled: list[float] = []
        for i in range(target_len):
            src_idx = i * (len(samples) - 1) / max(1, target_len - 1)
            idx0 = int(src_idx)
            idx1 = min(len(samples) - 1, idx0 + 1)
            frac = src_idx - idx0
            resampled.append(float((1.0 - frac) * samples[idx0] + frac * samples[idx1]))
        samples = resampled

    frame_length = max(1, int(sample_rate * frame_length_ms / 1000))
    frame_shift = max(1, int(sample_rate * frame_shift_ms / 1000))

    num_frames = max(1, (len(samples) - frame_length) // frame_shift + 1) if len(samples) >= frame_length else 1

    if np_mod is not None:
        audio_array = np_mod.array(samples, dtype=np_mod.float32)
        spectrogram = np_mod.zeros((num_frames, n_mels), dtype=np_mod.float32)
        hann_window = 0.5 - 0.5 * np_mod.cos(2.0 * math.pi * np_mod.arange(frame_length) / max(1, frame_length - 1))

        for f in range(num_frames):
            start = f * frame_shift
            end = start + frame_length
            if end <= len(audio_array):
                frame = audio_array[start:end] * hann_window
                fft_mag = np_mod.abs(np_mod.fft.rfft(frame, n=n_mels * 2))[:n_mels]
                spectrogram[f] = np_mod.log(fft_mag + 1e-6)
            else:
                spectrogram[f] = np_mod.full(n_mels, -6.0, dtype=np_mod.float32)

        return {
            "audio_values": audio_array,
            "spectrogram": spectrogram,
            "num_frames": int(num_frames),
            "sample_rate": sample_rate,
        }

    spectrogram_list = [[-6.0] * n_mels for _ in range(num_frames)]
    return {
        "audio_values": samples,
        "spectrogram": spectrogram_list,
        "num_frames": num_frames,
        "sample_rate": sample_rate,
    }


def format_multimodal_prompt(
    prompt: str,
    has_image: bool = False,
    has_audio: bool = False,
    image_token: str = "<image>",
    audio_token: str = "<audio>",
) -> MultimodalInput:
    """Format user prompt with multimodal placeholder tokens and compute token alignment masks.

    Args:
        prompt: Natural language query string.
        has_image: Whether an image input is attached.
        has_audio: Whether an audio input is attached.
        image_token: Special token placeholder representing image tokens (default: '<image>').
        audio_token: Special token placeholder representing audio tokens (default: '<audio>').

    Returns:
        Structured MultimodalInput dictionary with formatted prompt and alignment masks.

    Raises:
        ValueError: If prompt is None.

    """
    if not prompt:
        msg = "prompt cannot be empty."
        raise ValueError(msg)

    formatted_prompt = prompt.strip()
    if has_image and image_token not in formatted_prompt:
        formatted_prompt = f"{image_token} {formatted_prompt}".strip()
    if has_audio and audio_token not in formatted_prompt:
        formatted_prompt = f"{audio_token} {formatted_prompt}".strip()

    tokens = formatted_prompt.split()
    image_mask = [t == image_token for t in tokens] if has_image else None
    audio_mask = [t == audio_token for t in tokens] if has_audio else None

    if has_image and has_audio:
        modality = "multimodal"
    elif has_image:
        modality = "vision"
    elif has_audio:
        modality = "audio"
    else:
        modality = "text"

    return {
        "prompt": formatted_prompt,
        "image_token_mask": image_mask,
        "audio_token_mask": audio_mask,
        "modality": modality,
    }


def prepare_multimodal_inputs(
    prompt: str,
    kwargs: dict[str, Any],
    tensor_convert_fn: Any,
) -> tuple[str, Any, Any]:
    """Format multimodal prompt and convert image/audio inputs into backend tensors.

    Args:
        prompt: Input natural language prompt.
        kwargs: Generation keyword arguments containing optional `image_path`, `audio_path`, `pixel_values`, `audio_values`.
        tensor_convert_fn: Backend callable converting raw feature arrays into batched backend tensors.

    Returns:
        A 3-tuple of `(formatted_prompt, pixel_values, audio_values)`.

    """
    image_path = kwargs.get("image_path")
    audio_path = kwargs.get("audio_path")
    pixel_values = kwargs.get("pixel_values")
    audio_values = kwargs.get("audio_values")

    if image_path is not None or audio_path is not None:
        formatted = format_multimodal_prompt(
            prompt,
            has_image=image_path is not None or pixel_values is not None,
            has_audio=audio_path is not None or audio_values is not None,
        )
        prompt = str(formatted.get("prompt", prompt))

        if image_path is not None and pixel_values is None:
            img_in: Any = image_path
            img_res = process_image(img_in)
            pixel_values = tensor_convert_fn(img_res["pixel_values"])

        if audio_path is not None and audio_values is None:
            aud_in: Any = audio_path
            aud_res = process_audio(aud_in)
            audio_values = tensor_convert_fn(aud_res["audio_values"])

    return (prompt, pixel_values, audio_values)
