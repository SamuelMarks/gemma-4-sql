"""Provide module docstring."""

from pathlib import Path

import pytest

import gemma_4_sql.backends.jax.export as export_jax


class MockJnp:
    """Provide class docstring."""

    def zeros(self, _shape: object) -> object:
        """Execute function."""
        return [0]


class MockOCP:
    """Provide class docstring."""

    class PyTreeCheckpointer:
        """Provide class docstring."""

        def save(self, path: object, weights: object) -> None:
            """Execute function."""


def test_export_jax_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Execute function."""
    monkeypatch.setattr(export_jax, "jax", None)
    monkeypatch.setattr(export_jax, "jnp", None)
    monkeypatch.setattr(export_jax, "ocp", None)
    path = str(tmp_path / "export")
    res = export_jax.export_model("model1", path)
    if res["status"] != "mock_exported":
        raise AssertionError
    if not (tmp_path / "export" / "mock_jax_model_model1.bin").exists():
        raise AssertionError


def test_export_jax_real_no_flax(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Execute function."""
    monkeypatch.setattr(export_jax, "jax", "mock")
    monkeypatch.setattr(export_jax, "jnp", MockJnp())
    monkeypatch.setattr(export_jax, "ocp", MockOCP())
    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "flax", None)
    path = str(tmp_path / "export_real")
    res = export_jax.export_model("model2", path)
    if res["status"] != "exported_with_orbax":
        raise AssertionError


class MockConfig:
    """Provide class docstring."""

    @staticmethod
    def gemma4_e2b() -> str:
        """Execute function."""
        return "config"


class MockModel:
    """Provide class docstring."""

    def __init__(self, config: object, rngs: object) -> None:
        """Execute function."""


class MockNNX:
    """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, seed: object) -> None:
            """Execute function."""

    @staticmethod
    def state(_model: object) -> object:
        """Execute function."""
        return {"w": [0]}


def test_export_jax_real_with_flax(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Execute function."""
    monkeypatch.setattr(export_jax, "jax", "mock")
    monkeypatch.setattr(export_jax, "jnp", MockJnp())
    monkeypatch.setattr(export_jax, "ocp", MockOCP())
    original_import = __builtins__["__import__"]

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function."""
        if name == "flax" and "nnx" in fromlist:

            class Module:
                """Provide class docstring."""

                nnx = MockNNX()

            return Module()
        if name == "gemma_4_sql.backends.jax.gemma4":
            if "Gemma4Config" in fromlist:

                class Module:
                    """Provide class docstring."""

                    Gemma4Config = MockConfig()

                return Module()
            if "Gemma4ForCausalLM" in fromlist:

                class Module:
                    """Provide class docstring."""

                    Gemma4ForCausalLM = MockModel

                return Module()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    path = str(tmp_path / "export_flax")
    res = export_jax.export_model("model3", path)
    if res["status"] != "exported_with_orbax":
        raise AssertionError


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(export_jax)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", None)
    importlib.reload(export_jax)
    monkeypatch.undo()
    importlib.reload(export_jax)
