"""Tests for Keras inference logic."""

import pytest

import gemma_4_sql.backends.keras.inference as inf


class MockTf:
    pass


def test_generate_sql_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "tf", MockTf())
    monkeypatch.setattr(inf, "keras", object())

    res = inf.generate_sql("mock-model", "test prompt", beam_width=2, max_length=3, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError
    if not isinstance(res["sql"], str):
        raise TypeError


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "keras", None)
    res = inf.generate_sql("mock-model", "test prompt")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


def test_generate_sql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "tf", MockTf())
    monkeypatch.setattr(inf, "keras", object())

    orig_import = __import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "keras_nlp.models":

            class MockGemma:
                @staticmethod
                def from_preset(*args: object, **kwargs: object) -> object:
                    class Model:
                        def generate(self, *args: object, **kwargs: object) -> object:
                            msg = "err"
                            raise RuntimeError(msg)

                    return Model()

            class MockModule:
                GemmaCausalLM = MockGemma

            return MockModule()
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = inf.generate_sql("mock-model", "test prompt", test_mode=False)
    if "failed" not in str(res["status"]):
        raise AssertionError
