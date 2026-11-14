"""Unit and behavioral tests for MaxText Gin configuration generator without MagicMock."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.backends.maxtext.config_generator import (
    MaxTextHyperparameters,
    _to_float,
    _to_int,
    build_maxtext_cli_args,
    build_maxtext_hyperparameters,
    cleanup_temp_gin_configs,
    generate_maxtext_gin_config,
    normalize_model_architecture,
    save_maxtext_gin_config,
)
from gemma_4_sql.type_hints import TrainingConfig


def test_maxtext_hyperparameters_defaults() -> None:
    """Verify `MaxTextHyperparameters` defaults target Gemma 4 `gemma4_e2b`."""
    hp = MaxTextHyperparameters()
    assert hp.model_architecture == "gemma4_e2b"
    assert hp.per_device_batch_size == 2.0
    assert hp.global_batch_size == 8
    assert hp.learning_rate == 1e-4
    assert hp.learning_rate_schedule_steps == 1000
    assert hp.warmup_steps_fraction == 0.1
    assert hp.opt_type == "adamw"
    assert hp.mesh_axes == ["data", "fsdp", "tensor"]
    assert hp.base_output_directory == "./maxtext_output"
    assert hp.run_name == "gemma4_sql_run"
    assert hp.steps == 1000
    assert hp.dataset_name == "dummy"
    assert hp.checkpoint_period == 500


def test_to_float() -> None:
    """Verify `_to_float` handles primitives, booleans, None, and containers."""
    assert _to_float(None, 1.5) == 1.5
    assert _to_float(True, 1.5) == 1.5
    assert _to_float(2, 1.5) == 2.0
    assert _to_float(2.5, 1.5) == 2.5
    assert _to_float("3.5", 1.5) == 3.5
    assert _to_float([], 1.5) == 1.5
    assert _to_float({}, 1.5) == 1.5


def test_to_int() -> None:
    """Verify `_to_int` handles primitives, booleans, None, and containers."""
    assert _to_int(None, 5) == 5
    assert _to_int(False, 5) == 5
    assert _to_int(2, 5) == 2
    assert _to_int(2.5, 5) == 2
    assert _to_int("3", 5) == 3
    assert _to_int([], 5) == 5
    assert _to_int({}, 5) == 5


def test_normalize_model_architecture() -> None:
    """Verify `normalize_model_architecture` maps all Gemma 4 architectures accurately."""
    assert normalize_model_architecture("gemma-4") == "gemma4_e2b"
    assert normalize_model_architecture("gemma-4-e2b") == "gemma4_e2b"
    assert normalize_model_architecture("gemma-4-e4b") == "gemma4_e4b"
    assert normalize_model_architecture("  4b  ") == "gemma4_e4b"
    assert normalize_model_architecture("gemma-4-26b-a4b") == "gemma4_26b_a4b"
    assert normalize_model_architecture("gemma4-moe") == "gemma4_26b_a4b"
    assert normalize_model_architecture("gemma-4-31b") == "gemma4_31b"

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture("")

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture("   ")


def test_build_maxtext_hyperparameters_defaults() -> None:
    """Verify `build_maxtext_hyperparameters` derives defaults from real `TrainingConfig`."""
    config = TrainingConfig(
        model_name="gemma-4-e2b",
        dataset="dummy",
        epochs=1,
        learning_rate=1e-4,
        batch_size=2,
    )

    hp = build_maxtext_hyperparameters(config)
    assert hp.model_architecture == "gemma4_e2b"
    assert hp.per_device_batch_size == 2.0
    assert hp.global_batch_size == 8
    assert hp.learning_rate == 1e-4
    assert hp.steps == 1000
    assert hp.learning_rate_schedule_steps == 1000
    assert hp.warmup_steps_fraction == 0.1
    assert hp.opt_type == "adamw"
    assert hp.mesh_axes == ["data", "fsdp", "tensor"]
    assert hp.base_output_directory == "./maxtext_output"
    assert hp.run_name == "gemma4_e2b_dummy"
    assert hp.dataset_name == "dummy"
    assert hp.checkpoint_period == 500


def test_build_maxtext_hyperparameters_overrides() -> None:
    """Verify `build_maxtext_hyperparameters` applies `extra_kwargs` and explicit overrides."""
    config = TrainingConfig(
        model_name="gemma-4-e4b",
        dataset="my_dataset",
        epochs=2,
        learning_rate=2e-4,
        batch_size=4,
        extra_kwargs={"run_name": "custom_run", "warmup_steps_fraction": 0.2},
    )

    hp = build_maxtext_hyperparameters(
        config,
        opt_type="adam",
        mesh_axes=["fsdp", "tensor"],
        checkpoint_period=100,
        global_batch_size=16,
    )
    assert hp.model_architecture == "gemma4_e4b"
    assert hp.per_device_batch_size == 4.0
    assert hp.global_batch_size == 16
    assert hp.learning_rate == 2e-4
    assert hp.steps == 2000
    assert hp.warmup_steps_fraction == 0.2
    assert hp.opt_type == "adam"
    assert hp.mesh_axes == ["fsdp", "tensor"]
    assert hp.run_name == "custom_run"
    assert hp.dataset_name == "my_dataset"
    assert hp.checkpoint_period == 100


def test_build_maxtext_hyperparameters_validation_errors() -> None:
    """Verify `build_maxtext_hyperparameters` rejects invalid numerical and structural overrides."""
    config = TrainingConfig(model_name="gemma-4-e2b", extra_kwargs={})

    with pytest.raises(ValueError, match="per_device_batch_size must be positive"):
        build_maxtext_hyperparameters(config, per_device_batch_size=0)

    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        build_maxtext_hyperparameters(config, global_batch_size=-1)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_maxtext_hyperparameters(config, learning_rate=0.0)

    with pytest.raises(ValueError, match="steps must be positive"):
        build_maxtext_hyperparameters(config, steps=0)

    with pytest.raises(ValueError, match="learning_rate_schedule_steps must be positive"):
        build_maxtext_hyperparameters(config, learning_rate_schedule_steps=-5)

    with pytest.raises(ValueError, match=r"warmup_steps_fraction must be between 0\.0 and 1\.0"):
        build_maxtext_hyperparameters(config, warmup_steps_fraction=1.5)
    with pytest.raises(ValueError, match=r"warmup_steps_fraction must be between 0\.0 and 1\.0"):
        build_maxtext_hyperparameters(config, warmup_steps_fraction=-0.1)

    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(config, opt_type="   ")
    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(config, opt_type="")

    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(config, mesh_axes="data")
    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(config, mesh_axes=["data", 1])

    with pytest.raises(ValueError, match="checkpoint_period must be positive"):
        build_maxtext_hyperparameters(config, checkpoint_period=0)


def test_generate_maxtext_gin_config() -> None:
    """Verify `generate_maxtext_gin_config` formats valid Gin lines for Gemma 4."""
    config = TrainingConfig(
        model_name="gemma-4-31b",
        dataset="dummy",
        epochs=1,
        learning_rate=1e-4,
        batch_size=2,
        extra_kwargs={},
    )

    gin_str = generate_maxtext_gin_config(config)

    assert 'model_name = "gemma4_31b"' in gin_str
    assert "per_device_batch_size = 2.0" in gin_str
    assert "global_batch_size = 8" in gin_str
    assert "mesh_axes = ['data', 'fsdp', 'tensor']" in gin_str
    assert "learning_rate = 0.0001" in gin_str
    assert 'opt_type = "adamw"' in gin_str
    assert "steps = 1000" in gin_str
    assert "learning_rate_schedule_steps = 1000" in gin_str
    assert "warmup_steps_fraction = 0.1" in gin_str
    assert 'dataset_name = "dummy"' in gin_str
    assert "checkpoint_period = 500" in gin_str


def test_save_maxtext_gin_config_with_and_without_path(tmp_path: Path) -> None:
    """Verify `save_maxtext_gin_config` writes explicit paths and cleans up managed temporary dirs."""
    output_path = tmp_path / "custom" / "test.gin"
    gin_content = "some_config = 1"

    result_path = save_maxtext_gin_config(gin_content, output_path=output_path)
    assert result_path == output_path.resolve()
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == gin_content

    cleanup_temp_gin_configs()
    p1 = save_maxtext_gin_config(gin_content)
    p2 = save_maxtext_gin_config(gin_content)
    assert p1.exists() and p2.exists()
    assert p1.parent == p2.parent
    assert p1.read_text(encoding="utf-8") == gin_content
    managed_dir = p1.parent
    cleanup_temp_gin_configs()
    assert not p1.exists()
    assert not managed_dir.exists()


def test_save_maxtext_gin_config_empty_content() -> None:
    """Verify `save_maxtext_gin_config` rejects empty content."""
    with pytest.raises(ValueError, match="gin_content cannot be empty"):
        save_maxtext_gin_config("")

    with pytest.raises(ValueError, match="gin_content cannot be empty"):
        save_maxtext_gin_config("   ")


def test_build_maxtext_cli_args_with_and_without_gin_path(tmp_path: Path) -> None:
    """Verify `build_maxtext_cli_args` builds CLI arguments and cleans up temporary Gin files."""
    config = TrainingConfig(model_name="gemma-4-26b-a4b", dataset="dummy", epochs=1, learning_rate=1e-4, batch_size=2)

    args = build_maxtext_cli_args(config)
    assert len(args) == 2
    assert args[0] == "train.py"
    generated_path = Path(args[1])
    assert generated_path.exists()
    assert 'model_name = "gemma4_26b_a4b"' in generated_path.read_text(encoding="utf-8")
    cleanup_temp_gin_configs()
    assert not generated_path.exists()

    gin_path = tmp_path / "my_config.gin"
    gin_path.write_text("test = 1", encoding="utf-8")
    explicit_args = build_maxtext_cli_args(config, gin_config_path=gin_path)
    assert explicit_args == ["train.py", str(gin_path.resolve())]
