"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.keras.train as mod


def test_keras_train_import_error():
    """Docstring for test_keras_train_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp" or name == "keras":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.keras is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_keras_train_success():
    """Docstring for test_keras_train_success."""

    class MockConfig:
        """Docstring for MockConfig."""

        model_name = "m"
        dataset = "d"
        epochs = 1
        batch_size = 1

    with patch.object(mod, "keras", MagicMock()):
        with patch.object(mod, "tf", MagicMock()):
            with patch("builtins.__import__") as mock_import:
                mock_model = MagicMock()
                mock_model.fit.return_value.history = {"loss": [0.1]}
                mock_import.return_value.GemmaCausalLM.from_preset.return_value = mock_model

                with patch.object(mod, "build_dataloader") as mock_build:
                    mock_build.return_value = {"loader": [1, 2, 3]}  # iterable
                    config = MockConfig()
                    res = mod.train_model(config)
                    print("FIRST BLOCK CALL COUNT:", mock_build.call_count)
                    assert res["status"] == "completed"

            # test invalid dataloader
            with patch("builtins.__import__") as mock_import:
                mock_import.return_value.GemmaCausalLM.from_preset.return_value = mock_model
                with patch.object(mod, "build_dataloader") as mock_build:
                    mock_build.return_value = {"loader": None}
                    res = mod.train_model(config)
                    print("CALL COUNT:", mock_build.call_count)
                    print("RES STATUS IS", res["status"])
                    print("RES STATUS IS", res["status"])
                    assert "failed: Invalid dataloader" in res["status"]
