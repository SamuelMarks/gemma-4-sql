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
import pathlib
import struct
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, Self, TypedDict, runtime_checkable

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import MultimodalProcessingError
from gemma_4_sql.type_hints import (
    AudioInput,
    BackendKwargValue,
    ImageInput,
    MlxArrayProtocol,
    MultimodalInput,
    NDArrayProtocol,
    TensorProtocol,
    TfTensorProtocol,
    TokenizerProtocol,
    ToListArrayProtocol,
    TorchTensorProtocol,
)

logger = logging.getLogger(__name__)

jnp = optional_import("jax.numpy")
Image = optional_attr("PIL", "Image")


@runtime_checkable
class SavablePILImageProtocol(Protocol):
    """Protocol for PIL-like image objects supporting `.save(buf, format=...)`."""

    def save(self, fp: io.BytesIO, format: str = "PNG") -> None:
        """Save encoded image bytes into an in-memory buffer.

        Args:
            fp: Destination `BytesIO` buffer.
            format: Image container format string.

        """
        ...


@runtime_checkable
class OpenedPILImageProtocol(Protocol):
    """Protocol for an opened PIL Image supporting context management, `convert`, `resize`, and `tobytes`."""

    def __enter__(self) -> Self:
        """Enter image context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit image context."""
        ...

    def convert(self, mode: str) -> Self:
        """Convert image color mode (e.g., `'RGB'`)."""
        ...

    def resize(self, size: tuple[int, int], resample: int = 2) -> Self:
        """Resize image to `(width, height)`."""
        ...

    def tobytes(self) -> bytes:
        """Return raw pixel bytes for the image."""
        ...


@runtime_checkable
class PILImageModuleProtocol(Protocol):
    """Protocol for the `PIL.Image` module exposing `.open(...)`."""

    def open(self, fp: io.BytesIO) -> OpenedPILImageProtocol:
        """Open an image from a `BytesIO` stream."""
        ...


@runtime_checkable
class PILResamplingHolderProtocol(Protocol):
    """Protocol for a `PIL.Image` module exposing `.Resampling.BILINEAR`."""

    @property
    def Resampling(self) -> PILResamplingEnumProtocol:
        """Provide the `Resampling` enum namespace."""
        ...


@runtime_checkable
class PILResamplingEnumProtocol(Protocol):
    """Protocol for `PIL.Image.Resampling` exposing `BILINEAR`."""

    @property
    def BILINEAR(self) -> int:
        """Provide the bilinear resampling constant."""
        ...


class ProcessedImageDict(TypedDict):
    """Structured dictionary returned by `process_image`."""

    pixel_values: NDArrayProtocol
    patches: NDArrayProtocol
    num_patches: int
    shape: tuple[int, int, int]


class ProcessedAudioDict(TypedDict):
    """Structured dictionary returned by `process_audio`."""

    audio_values: NDArrayProtocol | list[float]
    spectrogram: NDArrayProtocol | list[list[float]]
    num_frames: int
    sample_rate: int


def load_image_bytes(image_input: ImageInput | SavablePILImageProtocol) -> bytes:
    """Load raw binary image bytes from filesystem paths, base64 strings, or byte sequences.

    Args:
        image_input: File path, base64 string, raw bytes, or PIL image object.

    Returns:
        Decoded binary image bytes.

    Raises:
        MultimodalProcessingError: If `image_input` is None or cannot be decoded.
        FileNotFoundError: If the specified image path does not exist on disk.

    """
    if image_input is None:
        msg = "image_input cannot be None."
        raise MultimodalProcessingError(msg, operation="load_image_bytes")

    if isinstance(image_input, bytes):
        return image_input

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

    if isinstance(image_input, SavablePILImageProtocol):
        pil_image = Image if Image is not None else optional_attr("PIL", "Image")
        if pil_image is None:
            logger.debug("Failed to import PIL: %s", "PIL.Image is unavailable")
        else:
            buf = io.BytesIO()
            image_input.save(buf, format="PNG")
            return buf.getvalue()

    msg = f"Unsupported image input type: {type(image_input)}"
    raise MultimodalProcessingError(msg, operation="load_image_bytes")


def process_image(
    image_input: ImageInput | SavablePILImageProtocol,
    target_size: tuple[int, int] = (224, 224),
    patch_size: int = 14,
) -> ProcessedImageDict:
    """Process an image into normalized RGB pixel arrays and extracted vision patches.

    Args:
        image_input: Image file path, binary bytes, or PIL Image.
        target_size: Target (height, width) tuple to resize image to (default: (224, 224)).
        patch_size: Square edge length of individual patches (default: 14).

    Returns:
        `ProcessedImageDict` containing:
            - 'pixel_values': Normalized float array of shape (3, H, W).
            - 'patches': Flattened 2D array of shape (num_patches, 3 * patch_size * patch_size).
            - 'num_patches': Total count of extracted vision patches.
            - 'shape': Shape tuple of the normalized image tensor.

    Raises:
        MultimodalProcessingError: If `target_size` or `patch_size` are invalid or image is corrupted.

    """
    (target_h, target_w) = target_size
    if target_h <= 0 or target_w <= 0:
        msg = f"target_size dimensions must be positive, got {target_size}"
        raise MultimodalProcessingError(msg, operation="process_image")
    if patch_size <= 0 or target_h % patch_size != 0 or target_w % patch_size != 0:
        msg = f"target dimensions {target_size} must be divisible by patch_size {patch_size}"
        raise MultimodalProcessingError(msg, operation="process_image")

    img_bytes = load_image_bytes(image_input)

    if Image is None or jnp is None:
        raise MultimodalProcessingError(
            "Corrupted or unreadable image: Pillow and JAX libraries are required.",
            operation="process_image",
        )
    bilinear = int(Image.Resampling.BILINEAR) if isinstance(Image, PILResamplingHolderProtocol) else 2
    try:
        if not isinstance(Image, PILImageModuleProtocol) or not callable(Image.open):
            raise TypeError("PIL.Image.open is not callable")
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            pil_resized = pil_img.convert("RGB").resize((target_w, target_h), bilinear)
            raw_rgb = pil_resized.tobytes()
            rgb_array = jnp.frombuffer(raw_rgb, dtype=jnp.uint8).reshape((target_h, target_w, 3)).astype(jnp.float32) / 255.0
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError) as e:
        raise MultimodalProcessingError(
            f"Corrupted or unreadable image: {e}",
            operation="process_image",
            cause=e,
        ) from e

    # Standard ImageNet normalization: (x - mean) / std
    mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
    std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

    norm_img = (rgb_array - mean) / std
    pixel_values: NDArrayProtocol = jnp.transpose(norm_img, (2, 0, 1)).astype(jnp.float32)

    num_patches_h = target_h // patch_size
    num_patches_w = target_w // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = 3 * patch_size * patch_size

    reshaped = pixel_values.reshape(3, num_patches_h, patch_size, num_patches_w, patch_size)
    transposed = jnp.transpose(reshaped, (1, 3, 0, 2, 4))
    patches: NDArrayProtocol = transposed.reshape((num_patches, patch_dim)).astype(jnp.float32)

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
        MultimodalProcessingError: If `audio_input` is None or unsupported.
        FileNotFoundError: If the specified audio file path does not exist on disk.

    """
    if audio_input is None:
        msg = "audio_input cannot be None."
        raise MultimodalProcessingError(msg, operation="load_audio_bytes")

    if isinstance(audio_input, bytes):
        return audio_input

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
    raise MultimodalProcessingError(msg, operation="load_audio_bytes")


def _decode_pcm_bytes(data_bytes: bytes, bits_per_sample: int, audio_format: int, num_channels: int) -> list[float]:
    """Decode raw PCM or IEEE float WAV data chunk bytes into mono float samples in [-1.0, 1.0].

    Args:
        data_bytes: Raw sample bytes from the WAV `data` chunk.
        bits_per_sample: Bit depth per sample (8, 16, 24, or 32).
        audio_format: WAV format code (1 for PCM, 3 for IEEE float).
        num_channels: Number of interleaved audio channels.

    Returns:
        1D list of mono normalized float samples.

    Raises:
        MultimodalProcessingError: If `bits_per_sample` is unsupported.

    """
    channels = max(1, num_channels)
    raw_samples: list[float] = []
    if bits_per_sample == 8:
        raw_samples = [(float(b) - 128.0) / 128.0 for b in data_bytes]
    elif bits_per_sample == 16:
        count = len(data_bytes) // 2
        ints = struct.unpack(f"<{count}h", data_bytes[: count * 2])
        raw_samples = [x / 32768.0 for x in ints]
    elif bits_per_sample == 24:
        count = len(data_bytes) // 3
        for idx in range(count):
            b0, b1, b2 = data_bytes[idx * 3 : idx * 3 + 3]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val & 0x800000:
                val -= 0x1000000
            raw_samples.append(val / 8388608.0)
    elif bits_per_sample == 32:
        count = len(data_bytes) // 4
        if audio_format == 3:
            floats = struct.unpack(f"<{count}f", data_bytes[: count * 4])
            raw_samples = [float(x) for x in floats]
        else:
            ints32 = struct.unpack(f"<{count}i", data_bytes[: count * 4])
            raw_samples = [x / 2147483648.0 for x in ints32]
    else:
        raise MultimodalProcessingError(
            f"Unsupported WAV bits_per_sample: {bits_per_sample}. Expected 8, 16, 24, or 32.",
            operation="_parse_wav_samples",
        )

    if channels > 1 and len(raw_samples) >= channels:
        num_frames = len(raw_samples) // channels
        return [sum(raw_samples[i * channels : (i + 1) * channels]) / float(channels) for i in range(num_frames)]
    return raw_samples or [0.0]


def _parse_wav_samples(wav_bytes: bytes) -> tuple[list[float], int]:
    """Parse PCM waveform samples and sampling rate from RIFF WAV bytes.

    Supports 8-bit unsigned PCM, 16-bit signed PCM, 24-bit signed PCM, and 32-bit PCM/float WAV containers.

    Args:
        wav_bytes: Raw binary bytes of a WAV container.

    Returns:
        Tuple of `(samples_list, sample_rate)`.

    Raises:
        MultimodalProcessingError: If `wav_bytes` is empty, truncated, or has an invalid RIFF/WAVE header.

    """
    if not wav_bytes:
        raise MultimodalProcessingError("Audio byte payload cannot be empty.", operation="_parse_wav_samples")

    if wav_bytes[:4] == b"RIFF":
        if len(wav_bytes) < 44 or wav_bytes[8:12] != b"WAVE":
            raise MultimodalProcessingError("Truncated or malformed RIFF/WAVE header.", operation="_parse_wav_samples")
        try:
            audio_format = struct.unpack("<H", wav_bytes[20:22])[0]
            num_channels = struct.unpack("<H", wav_bytes[22:24])[0]
            sample_rate = int(struct.unpack("<I", wav_bytes[24:28])[0])
            bits_per_sample = struct.unpack("<H", wav_bytes[34:36])[0]
        except struct.error as exc:
            raise MultimodalProcessingError("Corrupted WAV fmt chunk header.", operation="_parse_wav_samples", cause=exc) from exc

        data_offset = 12
        while data_offset <= len(wav_bytes) - 8:
            chunk_id = wav_bytes[data_offset : data_offset + 4]
            try:
                chunk_size = struct.unpack("<I", wav_bytes[data_offset + 4 : data_offset + 8])[0]
            except struct.error as exc:
                raise MultimodalProcessingError("Corrupted WAV subchunk header.", operation="_parse_wav_samples", cause=exc) from exc
            if chunk_id == b"data":
                data_bytes = wav_bytes[data_offset + 8 : data_offset + 8 + chunk_size]
                mono = _decode_pcm_bytes(data_bytes, bits_per_sample, audio_format, num_channels)
                return (mono, max(1, sample_rate))
            data_offset += 8 + chunk_size

        return ([0.0], max(1, sample_rate))

    samples = [(float(b) - 128.0) / 128.0 for b in wav_bytes[:16000]]
    return (samples, 16000)


def _hz_to_mel(freq_hz: float) -> float:
    """Convert frequency in Hz to HTK Mel scale.

    Args:
        freq_hz: Frequency in Hz.

    Returns:
        Frequency in Mels.

    """
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert HTK Mel scale value back to frequency in Hz.

    Args:
        mel: Frequency in Mels.

    Returns:
        Frequency in Hz.

    """
    return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


def _build_mel_filterbank_weights(sample_rate: int, n_fft: int, n_mels: int) -> list[list[float]]:
    """Build triangular HTK Mel filterbank matrix of shape `(n_mels, n_fft // 2 + 1)`.

    Args:
        sample_rate: Audio sampling rate in Hz.
        n_fft: FFT size.
        n_mels: Number of Mel filterbank bins.

    Returns:
        2D list of triangular filter weights of shape `(n_mels, num_fft_bins)`.

    """
    num_fft_bins = n_fft // 2 + 1
    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = [mel_min + i * (mel_max - mel_min) / (n_mels + 1) for i in range(n_mels + 2)]
    hz_points = [_mel_to_hz(m) for m in mel_points]
    bin_points = [min(num_fft_bins - 1, max(0, math.floor((n_fft + 1) * h / sample_rate))) for h in hz_points]

    filters: list[list[float]] = []
    for m in range(1, n_mels + 1):
        row = [0.0] * num_fft_bins
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        if f_center == f_left:
            f_center = min(num_fft_bins - 1, f_left + 1)
        if f_right <= f_center:
            f_right = min(num_fft_bins - 1, f_center + 1)
        for k in range(f_left, f_center):
            row[k] = (k - f_left) / max(1.0, float(f_center - f_left))
        for k in range(f_center, f_right + 1):
            row[k] = (f_right - k) / max(1.0, float(f_right - f_center))
        filters.append(row)
    return filters


def process_audio(
    audio_input: AudioInput,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
) -> ProcessedAudioDict:
    """Process raw audio into a normalized 16kHz waveform and log-mel spectrogram features.

    Uses a Hann-windowed Short-Time Fourier Transform (STFT) and triangular HTK Mel filterbank.

    Args:
        audio_input: Audio file path, binary bytes, or sample array.
        sample_rate: Target sampling rate in Hz (default: 16000).
        n_mels: Number of mel-frequency filterbank bins (default: 80).
        frame_length_ms: STFT analysis window duration in milliseconds (default: 25).
        frame_shift_ms: STFT frame hop duration in milliseconds (default: 10).

    Returns:
        `ProcessedAudioDict` containing:
            - 'audio_values': 1D normalized float waveform array.
            - 'spectrogram': 2D log-mel spectrogram array of shape (num_frames, n_mels).
            - 'num_frames': Number of temporal analysis frames.
            - 'sample_rate': Output sampling rate in Hz.

    Raises:
        MultimodalProcessingError: If `sample_rate` or `n_mels` are non-positive or audio is invalid.

    """
    if sample_rate <= 0 or n_mels <= 0:
        msg = "sample_rate and n_mels must be positive integers."
        raise MultimodalProcessingError(msg, operation="process_audio")

    samples: list[float]
    jnp_mod = jnp
    if isinstance(audio_input, NDArrayProtocol):
        flat_vals = audio_input.flatten().tolist()
        samples = [float(x) for x in flat_vals if isinstance(x, (int, float))] if isinstance(flat_vals, list) else [float(flat_vals)]
        orig_rate = sample_rate
    elif isinstance(audio_input, (list, tuple)):
        samples = [float(x) for x in audio_input]
        orig_rate = sample_rate
    else:
        if jnp_mod is None:
            jnp_mod = optional_import("jax.numpy")
            if jnp_mod is None:
                logger.debug("Failed to import jax.numpy: %s", "jax.numpy is unavailable")
        raw_bytes = load_audio_bytes(audio_input)
        samples, orig_rate = _parse_wav_samples(raw_bytes)

    if orig_rate != sample_rate and len(samples) > 1:
        target_len = max(1, int(len(samples) * sample_rate / orig_rate))
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
    n_fft = max(frame_length, n_mels * 2)

    num_frames = max(1, (len(samples) - frame_length) // frame_shift + 1) if len(samples) >= frame_length else 1
    fb_weights = _build_mel_filterbank_weights(sample_rate, n_fft, n_mels)

    if jnp_mod is not None:
        audio_array: NDArrayProtocol = jnp_mod.array(samples, dtype=jnp_mod.float32)
        hann_window = 0.5 - 0.5 * jnp_mod.cos(2.0 * math.pi * jnp_mod.arange(frame_length) / max(1, frame_length - 1))
        fb_matrix = jnp_mod.array(fb_weights, dtype=jnp_mod.float32)

        spec_rows: list[NDArrayProtocol] = []
        for f in range(num_frames):
            start = f * frame_shift
            end = start + frame_length
            if end <= len(samples):
                frame = audio_array[start:end] * hann_window
            else:
                rem = max(0, len(samples) - start)
                pad = jnp_mod.zeros(frame_length - rem, dtype=jnp_mod.float32)
                frame = jnp_mod.concatenate([audio_array[start : start + rem], pad]) * hann_window
            power_spec = jnp_mod.abs(jnp_mod.fft.rfft(frame, n=n_fft)) ** 2
            mel_energy = jnp_mod.matmul(fb_matrix, power_spec)
            spec_rows.append(jnp_mod.log(mel_energy + 1e-6))

        spectrogram: NDArrayProtocol = jnp_mod.stack(spec_rows, axis=0).astype(jnp_mod.float32)
        return {
            "audio_values": audio_array,
            "spectrogram": spectrogram,
            "num_frames": int(num_frames),
            "sample_rate": sample_rate,
        }

    spectrogram_list: list[list[float]] = []
    hann_pure = [0.5 - 0.5 * math.cos(2.0 * math.pi * n / max(1, frame_length - 1)) for n in range(frame_length)]
    num_fft_bins = n_fft // 2 + 1
    for f in range(num_frames):
        start = f * frame_shift
        frame_vals = [(samples[start + n] if (start + n) < len(samples) else 0.0) * hann_pure[n] for n in range(frame_length)]
        power_bins = [0.0] * num_fft_bins
        stride = max(1, frame_length // 32)
        for k in range(num_fft_bins):
            re_sum = 0.0
            im_sum = 0.0
            omega = 2.0 * math.pi * k / n_fft
            for n in range(0, frame_length, stride):
                angle = omega * n
                re_sum += frame_vals[n] * math.cos(angle)
                im_sum -= frame_vals[n] * math.sin(angle)
            power_bins[k] = re_sum * re_sum + im_sum * im_sum
        mel_row = [math.log(sum(fb_weights[m][k] * power_bins[k] for k in range(num_fft_bins)) + 1e-6) for m in range(n_mels)]
        spectrogram_list.append(mel_row)

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
    *,
    tokenizer: TokenizerProtocol | None = None,
) -> MultimodalInput:
    """Format user prompt with multimodal placeholder tokens and compute token alignment masks.

    When a `tokenizer` is provided, token masks are aligned directly with the encoded subword/byte
    token sequence; otherwise, whitespace token segmentation is used.

    Args:
        prompt: Natural language query string.
        has_image: Whether an image input is attached.
        has_audio: Whether an audio input is attached.
        image_token: Special token placeholder representing image tokens (default: '<image>').
        audio_token: Special token placeholder representing audio tokens (default: '<audio>').
        tokenizer: Optional `TokenizerProtocol` instance for subword-accurate mask alignment.

    Returns:
        Structured `MultimodalInput` dictionary with formatted prompt and alignment masks.

    Raises:
        MultimodalProcessingError: If `prompt` is empty or None.

    """
    if not prompt:
        msg = "prompt cannot be empty."
        raise MultimodalProcessingError(msg, operation="format_multimodal_prompt")

    formatted_prompt = prompt.strip()
    if has_image and image_token not in formatted_prompt:
        formatted_prompt = f"{image_token} {formatted_prompt}".strip()
    if has_audio and audio_token not in formatted_prompt:
        formatted_prompt = f"{audio_token} {formatted_prompt}".strip()

    if tokenizer is not None:
        encoded_ids = tokenizer.encode(formatted_prompt)
        img_ids = tokenizer.encode(image_token) if has_image else []
        aud_ids = tokenizer.encode(audio_token) if has_audio else []
        image_mask = [tid in img_ids for tid in encoded_ids] if has_image else None
        audio_mask = [tid in aud_ids for tid in encoded_ids] if has_audio else None
    else:
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


def prepare_multimodal_inputs[TOut](
    prompt: str,
    kwargs: Mapping[str, BackendKwargValue],
    tensor_convert_fn: Callable[[NDArrayProtocol | Sequence[float]], TOut],
) -> tuple[str, TOut | None, TOut | None]:
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
    raw_px = kwargs.get("pixel_values")
    raw_au = kwargs.get("audio_values")
    pixel_values: TOut | None = (  # ty: ignore[invalid-assignment]
        raw_px  # pyright: ignore[reportAssignmentType]
        if isinstance(
            raw_px,
            (TensorProtocol, TorchTensorProtocol, TfTensorProtocol, MlxArrayProtocol, NDArrayProtocol, ToListArrayProtocol),
        )
        else None
    )
    audio_values: TOut | None = (  # ty: ignore[invalid-assignment]
        raw_au  # pyright: ignore[reportAssignmentType]
        if isinstance(
            raw_au,
            (TensorProtocol, TorchTensorProtocol, TfTensorProtocol, MlxArrayProtocol, NDArrayProtocol, ToListArrayProtocol),
        )
        else None
    )

    if image_path is not None or audio_path is not None:
        formatted = format_multimodal_prompt(
            prompt,
            has_image=image_path is not None or raw_px is not None,
            has_audio=audio_path is not None or raw_au is not None,
        )
        prompt = str(formatted.get("prompt", prompt))

        if image_path is not None and raw_px is None and isinstance(image_path, (str, Path, bytes, Sequence, NDArrayProtocol)):
            img_res = process_image(image_path)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            pixel_values = tensor_convert_fn(img_res["pixel_values"])

        if audio_path is not None and raw_au is None and isinstance(audio_path, (str, Path, bytes, Sequence, NDArrayProtocol)):
            aud_res = process_audio(audio_path)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            audio_values = tensor_convert_fn(aud_res["audio_values"])

    return (prompt, pixel_values, audio_values)


__all__ = [
    "OpenedPILImageProtocol",
    "PILImageModuleProtocol",
    "PILResamplingEnumProtocol",
    "PILResamplingHolderProtocol",
    "ProcessedAudioDict",
    "ProcessedImageDict",
    "SavablePILImageProtocol",
    "format_multimodal_prompt",
    "load_audio_bytes",
    "load_image_bytes",
    "prepare_multimodal_inputs",
    "process_audio",
    "process_image",
]
