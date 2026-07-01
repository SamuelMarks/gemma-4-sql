from pathlib import Path

import pytest

import gemma_4_sql.backends.jax.export as export_jax


class MockJnp:
    def zeros(self, shape):
        return [0]


class MockOCP:
    class PyTreeCheckpointer:
        def save(self, path, weights):
            pass


def test_export_jax_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_jax, "jax", None)
    monkeypatch.setattr(export_jax, "jnp", None)
    monkeypatch.setattr(export_jax, "ocp", None)

    path = str(tmp_path / "export")
    res = export_jax.export_model("model1", path)
    assert res["status"] == "mock_exported"
    assert (tmp_path / "export" / "mock_jax_model_model1.bin").exists()


def test_export_jax_real_no_flax(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_jax, "jax", "mock")
    monkeypatch.setattr(export_jax, "jnp", MockJnp())
    monkeypatch.setattr(export_jax, "ocp", MockOCP())

    # Hide flax to trigger ImportError block
    import sys

    monkeypatch.setitem(sys.modules, "flax", None)

    path = str(tmp_path / "export_real")
    res = export_jax.export_model("model2", path)
    assert res["status"] == "exported_with_orbax"


def test_export_jax_real_with_flax(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_jax, "jax", "mock")
    monkeypatch.setattr(export_jax, "jnp", MockJnp())
    monkeypatch.setattr(export_jax, "ocp", MockOCP())

    class MockConfig:
        @staticmethod
        def gemma4_e2b():
            return "config"

    class MockModel:
        def __init__(self, config, rngs):
            pass

    class MockNNX:
        class Rngs:
            def __init__(self, seed):
                pass

        @staticmethod
        def state(model):
            return {"w": [0]}

    # Instead of deep module mocking, just monkeypatch __import__ in this specific file if possible.
    # It's easier to mock flax and the specific module import.

    original_import = __builtins__["__import__"]

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "flax" and "nnx" in fromlist:

            class Module:
                nnx = MockNNX()

            return Module()
        if name == "gemma_4_sql.backends.jax.gemma4":
            if "Gemma4Config" in fromlist:

                class Module:
                    Gemma4Config = MockConfig()

                return Module()
            if "Gemma4ForCausalLM" in fromlist:

                class Module:
                    Gemma4ForCausalLM = MockModel

                return Module()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    path = str(tmp_path / "export_flax")
    res = export_jax.export_model("model3", path)
    assert res["status"] == "exported_with_orbax"


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    # Mock jax import failure
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(export_jax)

    # Mock ocp import failure
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", None)
    importlib.reload(export_jax)

    # Restore original to not break other tests
    monkeypatch.undo()
    importlib.reload(export_jax)
