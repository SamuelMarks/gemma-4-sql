import gemma_4_sql.backends.keras.export as kexp


class MockKeras:
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            pass

    def Input(self, *args, **kwargs):
        return "input"

    class layers:
        def Embedding(*args, **kwargs):
            return lambda x: "x"

        def Dense(*args, **kwargs):
            return lambda x: "x"


def test_export_keras_real(monkeypatch):
    monkeypatch.setattr(kexp, "keras", MockKeras())
    res = kexp.export_model("model", "out")
    assert res["backend"] == "keras"


def test_export_keras_error(monkeypatch):
    monkeypatch.setattr(kexp, "keras", MockKeras())

    def raise_err(*args, **kwargs):
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockKeras.Model, "save", raise_err)
    try:
        res = kexp.export_model("model", "out")
    except ValueError:
        pass


def test_export_keras_missing(monkeypatch):
    monkeypatch.setattr(kexp, "keras", None)
    res = kexp.export_model("model", "out")
    assert res["status"] == "mock_exported"


def test_export_keras_imports_fail(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(kexp)
    monkeypatch.undo()
    importlib.reload(kexp)
