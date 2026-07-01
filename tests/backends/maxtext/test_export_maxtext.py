import pytest


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.maxtext.export as m_export

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_export)
    monkeypatch.undo()
    importlib.reload(m_export)


def test_export_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.maxtext.export as m_export

    class MockJax:
        class random:
            def PRNGKey(x):
                return x

    class MockJnp:
        def zeros(*args, **kwargs):
            return 1

        int32 = 1

    class MockOcp:
        class PyTreeCheckpointer:
            pass

        class CheckpointManager:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def save(self, *args, **kwargs):
                pass

        class CheckpointManagerOptions:
            def __init__(self, *args, **kwargs):
                pass

    class MockGemma4Model:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, **kwargs):
            return {"params": 1}

    monkeypatch.setattr(m_export, "jax", MockJax())
    monkeypatch.setattr(m_export, "jnp", MockJnp())
    monkeypatch.setattr(m_export, "ocp", MockOcp())

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "maxtext.models.gemma4" and "Gemma4Model" in fromlist:
            return type("M", (), {"Gemma4Model": MockGemma4Model})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "exported_with_maxtext_orbax"


def test_export_model_error(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.maxtext.export as m_export

    class MockJax:
        class random:
            def PRNGKey(x):
                return x

    class MockJnp:
        def zeros(*args, **kwargs):
            return 1

        int32 = 1

    class MockOcp:
        class PyTreeCheckpointer:
            pass

        class CheckpointManager:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def save(self, *args, **kwargs):
                pass

        class CheckpointManagerOptions:
            def __init__(self, *args, **kwargs):
                pass

    monkeypatch.setattr(m_export, "jax", MockJax())
    monkeypatch.setattr(m_export, "jnp", MockJnp())
    monkeypatch.setattr(m_export, "ocp", MockOcp())

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "maxtext.models.gemma4" and "Gemma4Model" in fromlist:
            msg = "err"
            raise ValueError(msg)
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "exported_with_maxtext_orbax"


def test_export_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.maxtext.export as m_export

    monkeypatch.setattr(m_export, "jax", None)
    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "mock_exported"
