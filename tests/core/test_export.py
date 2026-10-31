"""Behavioral tests for `common_export` infrastructure and SDK export argument forwarding."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.backends.common_export import build_export_result, run_export_wrapper
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.type_hints import ExportResultDict, JSONDict, JSONValue
from tests.utils.test_utils import RecordingBackendDouble


def _return_double(double: RecordingBackendDouble) -> RecordingBackendDouble:
    """Return the provided RecordingBackendDouble."""
    return double


def test_export_model_jax_unified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify export_model routes to the JAX backend and forwards model_name, export_path, and kwargs."""
    import gemma_4_sql.sdk.export as sdk_exp

    jax_double = RecordingBackendDouble("jax")

    def _get_jax_backend(_name: str) -> RecordingBackendDouble:
        """Return jax_double."""
        return _return_double(jax_double)

    monkeypatch.setattr(sdk_exp, "get_backend", _get_jax_backend)

    export_dir = tmp_path / "jax_unified_export"
    res = export_model("test_model", str(export_dir), backend="jax", step=5)
    assert len(jax_double.export_calls) == 1
    assert jax_double.export_calls[0] == ("test_model", str(export_dir), {"step": 5})
    assert res["backend"] == "jax"
    assert res["format"] == "safetensors"


def test_export_model_custom_backend_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test unified export dispatch through real JaxBackend adapter to jax.export.export_model."""
    import gemma_4_sql.backends.jax.export as jax_exp

    captured_calls: list[tuple[str, str, JSONDict]] = []

    def _spy_jax_export(model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Record forwarded export arguments and return structured ExportResultDict."""
        captured_calls.append((model_name, export_path, dict(kwargs)))
        return {
            "backend": "jax",
            "model": model_name,
            "export_path": export_path,
            "file_path": str(Path(export_path) / "checkpoint"),
            "format": str(kwargs.get("format", "orbax/saved_model")),
            "status": "exported_with_orbax",
        }

    monkeypatch.setattr(jax_exp, "export_model", _spy_jax_export)

    res = export_model("mock", str(tmp_path), backend="jax", format="gguf")
    assert res["status"] == "exported_with_orbax"
    assert res["format"] == "gguf"
    assert captured_calls == [("mock", str(tmp_path), {"format": "gguf"})]


def test_common_export_wrapper_and_build_result(tmp_path: Path) -> None:
    """Verify `build_export_result` and `run_export_wrapper` handle error_message, raise_on_error=False, and exception passthrough."""
    res_err = build_export_result("jax", "m", str(tmp_path), tmp_path / "f", "failed: err", "orbax", error_message="err")
    assert res_err["status_code"] == "failed"
    assert res_err["error_message"] == "err"

    res_no_raise = run_export_wrapper(
        "jax",
        "m",
        str(tmp_path / "sub_no_raise"),
        "orbax",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        raise_on_error=False,
    )
    assert res_no_raise["status"] == "failed: boom"
    assert res_no_raise["status_code"] == "failed"
    assert res_no_raise["error_message"] == "boom"

    with pytest.raises(ExportError, match="Failed to export model 'm' on jax: boom"):
        run_export_wrapper("jax", "m", str(tmp_path / "sub"), "orbax", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(ExportError, match="direct export err"):
        run_export_wrapper("jax", "m", str(tmp_path / "sub2"), "orbax", lambda: (_ for _ in ()).throw(ExportError("direct export err")))

    with pytest.raises(DependencyMissingError, match="missing export dep"):
        run_export_wrapper(
            "jax",
            "m",
            str(tmp_path / "sub3"),
            "orbax",
            lambda: (_ for _ in ()).throw(DependencyMissingError("missing export dep")),
        )
