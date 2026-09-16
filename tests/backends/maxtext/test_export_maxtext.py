"""Tests for MaxText model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typing_extensions import Self

import gemma_4_sql.backends.maxtext.export as m_export
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reload behavior when dependencies are missing."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_export)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", None)
    importlib.reload(m_export)
    monkeypatch.undo()
    importlib.reload(m_export)


def test_export_missing_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that DependencyMissingError is raised when JAX/Orbax are missing."""
    monkeypatch.setattr(m_export, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText export dependencies"):
        m_export.export_model("model", str(tmp_path))

    monkeypatch.setattr(m_export, "jax", MagicMock())
    monkeypatch.setattr(m_export, "jnp", None)
    with pytest.raises(DependencyMissingError, match="MaxText export dependencies"):
        m_export.export_model("model", str(tmp_path))

    monkeypatch.setattr(m_export, "jnp", MagicMock())
    monkeypatch.setattr(m_export, "ocp", None)
    with pytest.raises(DependencyMissingError, match="MaxText export dependencies"):
        m_export.export_model("model", str(tmp_path))


def test_export_missing_maxtext_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that DependencyMissingError is raised when Gemma4Model is None and no params passed."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()
    monkeypatch.setattr(m_export, "jax", mock_jax)
    monkeypatch.setattr(m_export, "jnp", mock_jnp)
    monkeypatch.setattr(m_export, "ocp", mock_ocp)
    monkeypatch.setattr(m_export, "Gemma4Model", None)

    with pytest.raises(DependencyMissingError, match="MaxText dependency"):
        m_export.export_model("model", str(tmp_path))


def test_export_model_init_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that ExportError is raised when model initialization fails."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()

    class FaultyModel:
        """Mock model class that fails upon initialization."""

        def __init__(self, _name: str) -> None:
            """Initialize faulty model."""
            raise RuntimeError("Corrupt model architecture")

    monkeypatch.setattr(m_export, "jax", mock_jax)
    monkeypatch.setattr(m_export, "jnp", mock_jnp)
    monkeypatch.setattr(m_export, "ocp", mock_ocp)
    monkeypatch.setattr(m_export, "Gemma4Model", FaultyModel)

    with pytest.raises(ExportError, match="Failed to initialize MaxText model"):
        m_export.export_model("model", str(tmp_path))


def test_export_model_save_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that ExportError is raised when Orbax checkpoint save fails."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()

    class MockCheckpointManager:
        """Mock CheckpointManager that fails on save."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize mock checkpoint manager."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, *args: Any, **kwargs: Any) -> None:
            """Raise save error."""
            raise OSError("Disk full or permission denied")

    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = MockCheckpointManager
    mock_ocp.CheckpointManagerOptions = MagicMock()
    mock_ocp.PyTreeCheckpointer = MagicMock()

    monkeypatch.setattr(m_export, "jax", mock_jax)
    monkeypatch.setattr(m_export, "jnp", mock_jnp)
    monkeypatch.setattr(m_export, "ocp", mock_ocp)

    with pytest.raises(ExportError, match="Failed to save MaxText Orbax checkpoint"):
        m_export.export_model("model", str(tmp_path), params={"weights": [1, 2, 3]})


def test_export_model_success_with_model_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test successful export through Gemma4Model initialization."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_saved: dict[str, Any] = {}

    class MockCheckpointManager:
        """Mock CheckpointManager that records saved step and weights."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize mock checkpoint manager."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, weights: Any) -> None:
            """Record saved payload."""
            mock_saved["step"] = step
            mock_saved["weights"] = weights

    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = MockCheckpointManager
    mock_ocp.CheckpointManagerOptions = MagicMock()
    mock_ocp.PyTreeCheckpointer = MagicMock()

    class ValidModel:
        """Mock model class that successfully initializes."""

        def __init__(self, name: str) -> None:
            """Initialize valid model."""
            self.name = name

        def init(self, rng: Any, dummy_input: Any) -> dict[str, str]:
            """Initialize model parameters."""
            return {"params": f"initialized_for_{self.name}"}

    monkeypatch.setattr(m_export, "jax", mock_jax)
    monkeypatch.setattr(m_export, "jnp", mock_jnp)
    monkeypatch.setattr(m_export, "ocp", mock_ocp)
    monkeypatch.setattr(m_export, "Gemma4Model", ValidModel)

    res = m_export.export_model("gemma_model", str(tmp_path))
    assert res["status"] == "exported_with_maxtext_orbax"
    assert res["backend"] == "maxtext"
    assert res["model"] == "gemma_model"
    assert res["format"] == "maxtext/checkpoint"
    assert mock_saved["step"] == 0
    assert mock_saved["weights"] == {"params": "initialized_for_gemma_model"}


def test_export_model_success_with_params_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test successful export when params are directly provided via kwargs."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_saved: dict[str, Any] = {}

    class MockCheckpointManager:
        """Mock CheckpointManager that records saved weights."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize mock checkpoint manager."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, weights: Any) -> None:
            """Record saved payload."""
            mock_saved["step"] = step
            mock_saved["weights"] = weights

    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = MockCheckpointManager
    mock_ocp.CheckpointManagerOptions = MagicMock()
    mock_ocp.PyTreeCheckpointer = MagicMock()

    monkeypatch.setattr(m_export, "jax", mock_jax)
    monkeypatch.setattr(m_export, "jnp", mock_jnp)
    monkeypatch.setattr(m_export, "ocp", mock_ocp)

    res = m_export.export_model("custom_model", str(tmp_path), weights={"layer1": "w1"})
    assert res["status"] == "exported_with_maxtext_orbax"
    assert mock_saved["weights"] == {"layer1": "w1"}
