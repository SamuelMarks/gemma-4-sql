import pytest


def test_export_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.pytorch.export as m_export

    class MockTorch:
        zeros = lambda *args, **kwargs: 1

    monkeypatch.setattr(m_export, "torch", MockTorch())
    monkeypatch.setattr(m_export, "save_file", lambda *args, **kwargs: None)

    class MockGemma4ForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def state_dict(self):
            return {"w": 1}

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.models.gemma4" and "Gemma4ForCausalLM" in fromlist:
            return type("M", (), {"Gemma4ForCausalLM": MockGemma4ForCausalLM})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "exported_with_safetensors"


def test_export_model_error(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.pytorch.export as m_export

    class MockTorch:
        zeros = lambda *args, **kwargs: 1

    monkeypatch.setattr(m_export, "torch", MockTorch())
    monkeypatch.setattr(m_export, "save_file", lambda *args, **kwargs: None)

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.models.gemma4" and "Gemma4ForCausalLM" in fromlist:
            msg = "err"
            raise ValueError(msg)
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "exported_with_safetensors"


def test_export_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.pytorch.export as m_export

    monkeypatch.setattr(m_export, "torch", None)
    res = m_export.export_model("model", str(tmp_path))
    assert res["status"] == "mock_exported"
