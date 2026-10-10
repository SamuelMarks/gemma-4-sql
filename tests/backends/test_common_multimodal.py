"""Module docstring."""

from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.common_multimodal as mod


def test_top_level_import_failures(monkeypatch):
    """Docstring for test_top_level_import_failures."""
    import builtins
    import importlib
    import sys

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name in ["numpy", "PIL"]:
            raise ImportError(f"simulated missing {name}")
        return orig_import(name, *args, **kwargs)

    # Remove the module from sys.modules so we can cleanly reload it
    if "gemma_4_sql.backends.common_multimodal" in sys.modules:
        del sys.modules["gemma_4_sql.backends.common_multimodal"]

    with patch("builtins.__import__", side_effect=mock_import):
        import gemma_4_sql.backends.common_multimodal as mod_reloaded

        assert mod_reloaded.np is None
        assert mod_reloaded.Image is None

    # Reload again to restore
    importlib.reload(mod_reloaded)


def test_missing_np_import(monkeypatch):
    # Save original state
    """Docstring for test_missing_np_import."""
    orig_np = mod.np
    orig_Image = mod.Image

    # We want to test the fallback when Image or np are initially None
    # and importing them fails.

    # Test load_image_bytes when Image is None
    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        monkeypatch.setattr(mod, "Image", None)

        # We need to mock __import__ so `from PIL import Image as _Image` fails
        import builtins

        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Docstring for mock_import."""
            if name == "PIL":
                raise ImportError("simulated missing import")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            mock_image_input = MagicMock()
            mock_image_input.save = MagicMock()

            try:
                mod.load_image_bytes(mock_image_input)
            except ValueError:
                pass
            assert mock_debug.call_args_list[0][0][0] == "Failed to import PIL: %s"

    # Test load_image_bytes when Image is None but import succeeds! (covers line 88)
    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        monkeypatch.setattr(mod, "Image", None)

        mock_image_input = MagicMock()
        mock_image_input.save = MagicMock()

        # allow import to succeed
        res = mod.load_image_bytes(mock_image_input)
        assert isinstance(res, bytes)
        assert mod.Image is not None  # successfully imported

    # Test process_audio when np is None and audio_input is array-like
    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        monkeypatch.setattr(mod, "np", None)

        import builtins

        orig_import = builtins.__import__

        def mock_import_np(name, *args, **kwargs):
            """Docstring for mock_import_np."""
            if name == "numpy":
                raise ImportError("simulated missing np")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import_np):
            wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            res = mod.process_audio(wav_header)
            assert len(res["audio_values"]) > 0

            # Also test the else branch when audio_input is not str/bytes
            class DummyAudio:
                """Docstring for DummyAudio."""

                def read(self):
                    """Docstring for read."""
                    return wav_header

            try:
                mod.process_audio(DummyAudio())
            except ValueError:
                pass

    # Test _parse_wav_samples with bits_per_sample = 8 to hit break
    with patch("gemma_4_sql.backends.common_multimodal.logger.debug") as mock_debug:
        bad_wav_8bit = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x08\x00data\x02\x00\x00\x00\x01\x01"
        res = mod._parse_wav_samples(bad_wav_8bit)
        # breaks loop, falls back to synth
        assert len(res[0]) > 0

    # Test _parse_wav_samples where data chunk exists but bits_per_sample != 16
    bad_wav_24bit_data = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x18\x00data\x02\x00\x00\x00\x01\x01"
    res = mod._parse_wav_samples(bad_wav_24bit_data)
    assert len(res[0]) > 0

    # restore
    mod.np = orig_np
    mod.Image = orig_Image


def test_process_audio_np_ndarray(monkeypatch):
    """Docstring for test_process_audio_np_ndarray."""
    import numpy as np

    # Test process_audio with numpy array
    audio_input = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = mod.process_audio(audio_input)
    assert len(res["audio_values"]) > 0
    assert len(res["spectrogram"]) > 0


def test_process_audio_lazy_numpy(monkeypatch):
    """Docstring for test_process_audio_lazy_numpy."""
    import numpy as np_real

    monkeypatch.setattr(mod, "np", None)
    audio_input = np_real.array([1.0, 2.0, 3.0])
    res = mod.process_audio(audio_input)
    assert len(res["audio_values"]) > 0


def test_process_audio_no_np_fallback(monkeypatch):
    # Test process_audio returning fallback spectrogram
    """Docstring for test_process_audio_no_np_fallback."""
    monkeypatch.setattr(mod, "np", None)

    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res = mod.process_audio(wav_header)
    assert len(res["spectrogram"]) == res["num_frames"]
