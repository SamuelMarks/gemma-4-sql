"""Provide module docstring."""

import pytest


class MockTorch:
    """Provide class docstring."""

    def zeros(*_args: object, **_kwargs: object) -> int:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return cls()

    def state_dict(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"w": 1}


def test_export_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_export = __import__("gemma_4_sql.backends.pytorch.export", fromlist=[""])
    monkeypatch.setattr(m_export, "torch", MockTorch())
    monkeypatch.setattr(m_export, "save_file", lambda *_args, **_kwargs: None)
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "transformers.models.gemma4" and "Gemma4ForCausalLM" in fromlist:
            return type("M", (), {"Gemma4ForCausalLM": MockGemma4ForCausalLM})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    res = m_export.export_model("model", str(tmp_path))
    if res["status"] != "exported_with_safetensors":
        raise AssertionError


def test_export_model_error(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_export = __import__("gemma_4_sql.backends.pytorch.export", fromlist=[""])
    monkeypatch.setattr(m_export, "torch", MockTorch())
    monkeypatch.setattr(m_export, "save_file", lambda *_args, **_kwargs: None)
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.


        Raises:
            ValueError: Description.

        """
        if name == "transformers.models.gemma4" and "Gemma4ForCausalLM" in fromlist:
            msg = "err"
            raise ValueError(msg)
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    with pytest.raises(ValueError, match="Failed to load model model"):
        m_export.export_model("model", str(tmp_path))


def test_export_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_export = __import__("gemma_4_sql.backends.pytorch.export", fromlist=[""])
    monkeypatch.setattr(m_export, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch or safetensors missing"):
        m_export.export_model("model", str(tmp_path))
