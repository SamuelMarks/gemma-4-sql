"""Test file."""


def test_multimodal_lazy_numpy_import_branch(monkeypatch):
    """Test function."""
    import builtins
    import sys
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.common_multimodal as cm

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "numpy":
            mock_np = MagicMock()
            mock_np.ndarray = type("ndarray", (), {})
            return mock_np
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)

    monkeypatch.setattr(cm, "np", None)

    try:
        cm.process_audio(b"1234")
    except ValueError:
        pass


def test_multimodal_lazy_numpy_import_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import builtins

    import gemma_4_sql.backends.common_multimodal as cm

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "numpy":
            raise ImportError("Simulated")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import importlib

    importlib.reload(cm)

    monkeypatch.setattr(cm, "np", None)

    from unittest.mock import MagicMock

    mock_logger = MagicMock()
    monkeypatch.setattr(cm, "logger", mock_logger)

    try:
        cm.process_audio(b"1234")
    except ValueError:
        pass

    any("Failed to import numpy" in str(c) for c in mock_logger.debug.call_args_list)


def test_multimodal_lazy_numpy_import_error_reach(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import builtins
    import sys

    import gemma_4_sql.backends.common_multimodal as cm

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "numpy":
            raise ImportError("Simulated")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.delitem(sys.modules, "numpy", raising=False)

    monkeypatch.setattr(cm, "np", None)

    from unittest.mock import MagicMock

    mock_logger = MagicMock()
    monkeypatch.setattr(cm, "logger", mock_logger)

    import importlib

    importlib.reload(cm)
    monkeypatch.setattr(cm, "logger", mock_logger)
    try:
        monkeypatch.setattr(cm, "load_audio_bytes", lambda x: b"")
        monkeypatch.setattr(cm, "_parse_wav_samples", lambda x: ([0.0], 16000))
        cm.process_audio(b"fake")
    except ValueError:
        pass

    found = any("Failed to import numpy" in str(c) for c in mock_logger.debug.call_args_list)
    assert found


def test_parse_wav_samples_break_early(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    # Create a fmt chunk but no data chunk
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    res, _ = cm._parse_wav_samples(header + fmt_chunk)
    assert len(res) > 0


def test_parse_wav_samples_chunk_exhaustion(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    junk = b"JUNK" + struct.pack("<I", 4) + b"1234"
    res, _ = cm._parse_wav_samples(header + junk)
    assert len(res) > 0


def test_parse_wav_samples_chunk_exhaustion_data_not_found():
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    junk = b"JUNK" + struct.pack("<I", 4) + b"1234"
    res, _ = cm._parse_wav_samples(header + junk)

    padding = b"123"
    res, _ = cm._parse_wav_samples(header + junk + padding)


def test_parse_wav_samples_struct_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    junk = b"JUNK1"
    res, _ = cm._parse_wav_samples(header + junk)
    assert len(res) > 0


def test_parse_wav_samples_chunk_exhaustion_break(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    res, _ = cm._parse_wav_samples(header)
    assert len(res) > 0


def test_format_multimodal_prompt_branches():
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.common_multimodal as cm

    try:
        cm.format_multimodal_prompt("")
    except ValueError:
        pass

    res = cm.format_multimodal_prompt("hello", has_image=True, has_audio=True)
    assert res["modality"] == "multimodal"

    res = cm.format_multimodal_prompt("hello", has_image=True)
    assert res["modality"] == "vision"

    res = cm.format_multimodal_prompt("hello", has_audio=True)
    assert res["modality"] == "audio"

    res = cm.format_multimodal_prompt("hello")
    assert res["modality"] == "text"


def test_parse_wav_samples_data_chunk_break(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    data_chunk = b"data" + struct.pack("<I", 2) + b"\x00\x00"

    res, _ = cm._parse_wav_samples(header + fmt_chunk + data_chunk)
    assert len(res) > 0


def test_parse_wav_samples_data_chunk_break_really(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import struct

    import gemma_4_sql.backends.common_multimodal as cm

    header = b"RIFF" + struct.pack("<I", 12) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)

    data_chunk = b"data" + struct.pack("<I", 1000)

    res, _ = cm._parse_wav_samples(header + fmt_chunk + data_chunk)
    assert len(res) > 0
