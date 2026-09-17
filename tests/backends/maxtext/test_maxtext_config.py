"""Unit tests for MaxText Gin configuration generator and bridge."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gemma_4_sql.backends.maxtext.config_generator import (
    MaxTextHyperparameters,
    _to_float,
    _to_int,
    build_maxtext_cli_args,
    build_maxtext_hyperparameters,
    generate_maxtext_gin_config,
    normalize_model_architecture,
    save_maxtext_gin_config,
)
from gemma_4_sql.type_hints import TrainingConfig


def test_to_float_and_to_int() -> None:
    """Test safe conversion helpers for float and int values."""
    assert _to_float(None, 2.5) == 2.5
    assert _to_float(math.pi, 1.0) == math.pi
    assert _to_float("4.5", 1.0) == 4.5
    assert _to_float(10, 1.0) == 10.0
    assert _to_float([], 9.9) == 9.9

    assert _to_int(None, 4) == 4
    assert _to_int(8, 1) == 8
    assert _to_int("16", 1) == 16
    assert _to_int(3.7, 1) == 3
    assert _to_int({}, 42) == 42


def test_normalize_model_architecture() -> None:
    """Test normalization of model architecture identifiers."""
    assert normalize_model_architecture("gemma-4") == "gemma4_2b"
    assert normalize_model_architecture("gemma-4-7b") == "gemma4_7b"
    assert normalize_model_architecture("GEMMA4_7B") == "gemma4_7b"
    assert normalize_model_architecture("gemma4_2b") == "gemma4_2b"

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture("")

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture(None)  # type: ignore[arg-type]


def test_build_maxtext_hyperparameters_default() -> None:
    """Test building default MaxText hyperparameters from basic TrainingConfig."""
    cfg = TrainingConfig(
        model_name="gemma-4",
        dataset="spider",
        epochs=2,
        learning_rate=2e-5,
        batch_size=4,
    )
    params = build_maxtext_hyperparameters(cfg)

    assert isinstance(params, MaxTextHyperparameters)
    assert params.model_architecture == "gemma4_2b"
    assert params.per_device_batch_size == 4.0
    assert params.global_batch_size == 16
    assert params.learning_rate == 2e-5
    assert params.steps == 2000
    assert params.dataset_name == "spider"
    assert params.mesh_axes == ["data", "fsdp", "tensor"]


def test_build_maxtext_hyperparameters_overrides() -> None:
    """Test building MaxText hyperparameters with extra_kwargs and explicit kwargs."""
    cfg = TrainingConfig(
        model_name="gemma-4-7b",
        dataset="boba",
        extra_kwargs={
            "per_device_batch_size": 8.0,
            "global_batch_size": 32,
            "opt_type": "adam",
        },
    )
    params = build_maxtext_hyperparameters(
        cfg,
        steps=500,
        warmup_steps_fraction=0.05,
        mesh_axes=["data", "model"],
        run_name="custom_run",
        base_output_directory="/tmp/output",
        checkpoint_period=100,
    )

    assert params.model_architecture == "gemma4_7b"
    assert params.per_device_batch_size == 8.0
    assert params.global_batch_size == 32
    assert params.opt_type == "adam"
    assert params.steps == 500
    assert params.warmup_steps_fraction == 0.05
    assert params.mesh_axes == ["data", "model"]
    assert params.run_name == "custom_run"
    assert params.base_output_directory == "/tmp/output"
    assert params.checkpoint_period == 100


def test_build_maxtext_hyperparameters_validation_failures() -> None:
    """Test validation constraints on all hyperparameter fields."""
    base_cfg = TrainingConfig()

    with pytest.raises(ValueError, match="per_device_batch_size must be positive"):
        build_maxtext_hyperparameters(base_cfg, per_device_batch_size=0)

    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        build_maxtext_hyperparameters(base_cfg, global_batch_size=-1)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_maxtext_hyperparameters(base_cfg, learning_rate=0)

    with pytest.raises(ValueError, match="steps must be positive"):
        build_maxtext_hyperparameters(base_cfg, steps=0)

    with pytest.raises(ValueError, match="learning_rate_schedule_steps must be positive"):
        build_maxtext_hyperparameters(base_cfg, learning_rate_schedule_steps=-5)

    with pytest.raises(ValueError, match=r"warmup_steps_fraction must be between 0\.0 and 1\.0"):
        build_maxtext_hyperparameters(base_cfg, warmup_steps_fraction=1.5)

    with pytest.raises(ValueError, match=r"warmup_steps_fraction must be between 0\.0 and 1\.0"):
        build_maxtext_hyperparameters(base_cfg, warmup_steps_fraction=-0.1)

    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(base_cfg, opt_type="")

    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(base_cfg, mesh_axes="invalid")

    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(base_cfg, mesh_axes=[123])

    with pytest.raises(ValueError, match="checkpoint_period must be positive"):
        build_maxtext_hyperparameters(base_cfg, checkpoint_period=0)


def test_generate_maxtext_gin_config() -> None:
    """Test Gin configuration string generation."""
    cfg = TrainingConfig(model_name="gemma-4", dataset="test_ds")
    gin_str = generate_maxtext_gin_config(cfg, base_output_directory="/tmp/maxtext")

    assert 'base_output_directory = "/tmp/maxtext"' in gin_str
    assert 'model_name = "gemma4_2b"' in gin_str
    assert "per_device_batch_size = 2.0" in gin_str
    assert "mesh_axes = ['data', 'fsdp', 'tensor']" in gin_str
    assert 'opt_type = "adamw"' in gin_str
    assert 'dataset_name = "test_ds"' in gin_str


def test_save_maxtext_gin_config(tmp_path: Path) -> None:
    """Test saving Gin configuration text to temporary and explicit files."""
    content = "model_name = 'gemma4_2b'\nsteps = 100\n"

    # Empty content error
    with pytest.raises(ValueError, match="gin_content cannot be empty"):
        save_maxtext_gin_config("")

    # Explicit output path
    dest = tmp_path / "subdir" / "test_config.gin"
    saved_path = save_maxtext_gin_config(content, output_path=dest)
    assert saved_path == dest.resolve()
    assert saved_path.is_file()
    assert saved_path.read_text(encoding="utf-8") == content

    # Temporary file path
    temp_saved = save_maxtext_gin_config(content)
    assert temp_saved.is_file()
    assert temp_saved.suffix == ".gin"
    assert temp_saved.read_text(encoding="utf-8") == content
    temp_saved.unlink(missing_ok=True)


def test_build_maxtext_cli_args(tmp_path: Path) -> None:
    """Test construction of CLI argument list for maxtext.train.main."""
    cfg = TrainingConfig(model_name="gemma-4", dataset="spider")

    # With explicit gin path
    dummy_gin = tmp_path / "run.gin"
    dummy_gin.write_text("model_name = 'gemma4_2b'\n", encoding="utf-8")
    args = build_maxtext_cli_args(cfg, gin_config_path=dummy_gin)
    assert args == ["train.py", str(dummy_gin.resolve())]

    # Without explicit gin path (auto-generated)
    auto_args = build_maxtext_cli_args(cfg)
    assert len(auto_args) == 2
    assert auto_args[0] == "train.py"
    auto_path = Path(auto_args[1])
    assert auto_path.is_file()
    auto_path.unlink(missing_ok=True)


def test_maxtext_hyperparameters_extra_kwargs() -> None:
    """Test build_maxtext_hyperparameters with extra_kwargs in TrainingConfig."""
    cfg = TrainingConfig(
        model_name="gemma-4-7b",
        dataset="custom_spider",
        epochs=3,
        extra_kwargs={
            "per_device_batch_size": 4,
            "global_batch_size": 16,
            "steps": 3000,
            "run_name": "custom_experiment",
            "checkpoint_period": 250,
            "base_output_directory": "/gs/bucket/output",
        },
    )
    hp = build_maxtext_hyperparameters(cfg)
    assert hp.model_architecture == "gemma4_7b"
    assert hp.per_device_batch_size == 4.0
    assert hp.global_batch_size == 16
    assert hp.steps == 3000
    assert hp.run_name == "custom_experiment"
    assert hp.checkpoint_period == 250
    assert hp.base_output_directory == "/gs/bucket/output"


def test_build_maxtext_cli_args_overrides(tmp_path: Path) -> None:
    """Test build_maxtext_cli_args with keyword parameter overrides passed into gin generation."""
    cfg = TrainingConfig(model_name="gemma-4", dataset="bench_ds")
    cli_args = build_maxtext_cli_args(cfg, opt_type="adamw", steps=500, run_name="test_run_override")
    assert len(cli_args) == 2
    gin_path = Path(cli_args[1])
    assert gin_path.is_file()
    content = gin_path.read_text(encoding="utf-8")
    assert 'opt_type = "adamw"' in content
    assert "steps = 500" in content
    assert 'run_name = "test_run_override"' in content
    gin_path.unlink(missing_ok=True)
