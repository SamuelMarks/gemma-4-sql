"""Unit and behavioral tests for Keras model export."""

from __future__ import annotations

import pathlib

import pytest

from tests.conftest import requires_keras

pytestmark = requires_keras

import gemma_4_sql.backends.keras.export as mod
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_keras_export_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify `export_model` raises `DependencyMissingError` when `keras` is `None`."""
    monkeypatch.setattr(mod, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing for export"):
        mod.export_model("my_model", str(tmp_path))


def test_export_model_keras_success(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify `export_model` saves a real Keras model to `model.keras` and raises `ExportError` on load failure."""
    from tests.utils.tensor_helpers import TypedKerasModelInstanceProtocol, get_typed_keras, get_typed_keras_ops

    keras = get_typed_keras()
    ops = get_typed_keras_ops()

    real_model = keras.Sequential([keras.layers.Input(shape=(4,)), keras.layers.Dense(2)])
    x_sample = ops.ones((1, 4), dtype="float32")
    expected_out = ops.stop_gradient(real_model(x_sample))

    preset_calls: list[str] = []
    should_fail = False

    class _PresetLoader:
        """Typed `GemmaCausalLM` preset loader double."""

        @classmethod
        def from_preset(cls, model_name: str) -> TypedKerasModelInstanceProtocol:
            """Record preset name and return `real_model` or raise `ValueError`.

            Returns:
                The prebuilt `real_model` Sequential instance.

            Raises:
                ValueError: If `should_fail` is True.

            """
            preset_calls.append(model_name)
            if should_fail:
                raise ValueError("fake error")
            return real_model

    monkeypatch.setattr(mod, "GemmaCausalLM", _PresetLoader)

    export_dir = tmp_path / "keras_export"
    res = mod.export_model("my_model", str(export_dir))
    assert res["status"] == "exported_with_keras"
    assert res["format"] == "keras_v3/keras_tensor"
    saved_file = export_dir / "model.keras"
    assert saved_file.is_file()
    assert preset_calls == ["my_model"]

    reloaded_model = keras.saving.load_model(saved_file)
    assert callable(reloaded_model)
    reloaded_out = ops.stop_gradient(reloaded_model(x_sample))
    assert bool(ops.all(ops.isclose(reloaded_out, expected_out, atol=1e-5)))

    # ValueError during preset loading raises ExportError
    should_fail = True
    with pytest.raises(ExportError, match="Failed to load model my_model"):
        mod.export_model("my_model", str(export_dir))

    def _none_attr(*_a: str, **_k: str) -> None:
        """Return None for import_optional_symbol."""
        return

    # Missing GemmaCausalLM class raises ExportError
    monkeypatch.setattr(mod, "GemmaCausalLM", None)
    monkeypatch.setattr(mod, "import_optional_symbol", _none_attr)
    with pytest.raises(ExportError, match="Failed to load model my_model"):
        mod.export_model("my_model", str(export_dir))
