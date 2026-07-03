"""Provide module docstring."""

import pytest


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    m_export = __import__("gemma_4_sql.backends.maxtext.export")
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_export)
    monkeypatch.undo()
    importlib.reload(m_export)


class MockJax:
    """Provide class docstring."""

    class MockRandom:
        """Provide class docstring."""

        def mock_prngkey(self) -> object:
            """Execute function."""
            return self

        PRNGKey = mock_prngkey

    random = MockRandom


class MockJnp:
    """Provide class docstring."""

    def zeros(*_args: object, **_kwargs: object) -> int:
        """Execute function."""
        return 1

    int32 = 1


class MockOcp:
    """Provide class docstring."""

    class PyTreeCheckpointer:
        """Provide class docstring."""

    class CheckpointManager:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

        def __enter__(self) -> object:
            """Execute function."""
            return self

        def __exit__(self, *args: object) -> object:
            """Execute function."""

        def save(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

    class CheckpointManagerOptions:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""


class MockGemma4Model:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def init(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return {"params": 1}


def test_export_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_export = __import__("gemma_4_sql.backends.maxtext.export")
    monkeypatch.setattr(m_export, "jax", MockJax())
    monkeypatch.setattr(m_export, "jnp", MockJnp())
    monkeypatch.setattr(m_export, "ocp", MockOcp())
    builtins = __import__("builtins")
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function."""
        if name == "maxtext.models.gemma4" and "Gemma4Model" in fromlist:
            return type("M", (), {"Gemma4Model": MockGemma4Model})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    res = m_export.export_model("model", str(tmp_path))
    if res["status"] != "exported_with_maxtext_orbax":
        raise AssertionError


def test_export_model_error(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_export = __import__("gemma_4_sql.backends.maxtext.export")
    monkeypatch.setattr(m_export, "jax", MockJax())
    monkeypatch.setattr(m_export, "jnp", MockJnp())
    monkeypatch.setattr(m_export, "ocp", MockOcp())
    builtins = __import__("builtins")
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function."""
        if name == "maxtext.models.gemma4" and "Gemma4Model" in fromlist:
            msg = "err"
            raise ValueError(msg)
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    res = m_export.export_model("model", str(tmp_path))
    if res["status"] != "exported_with_maxtext_orbax":
        raise AssertionError


def test_export_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_export = __import__("gemma_4_sql.backends.maxtext.export")
    monkeypatch.setattr(m_export, "jax", None)
    res = m_export.export_model("model", str(tmp_path))
    if res["status"] != "mock_exported":
        raise AssertionError
