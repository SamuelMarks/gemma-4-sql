"""Custom type hints for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONDict = dict[str, JSONValue]

# TensorType is a generic alias for backend-specific tensors (JAX arrays, PyTorch tensors, etc.)
TensorType = TypeVar("TensorType", bound=Any)
ModelType = TypeVar("ModelType", bound=Any)


@dataclass
class DPOConfig:
    """Config for DPO execution."""

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
    batch_size: int = 2
    test_mode: bool = False


@dataclass
class ETLConfig:
    """Config for ETL execution."""

    dataset_name: str
    split: str
    batch_size: int = 32
    distributed: bool = False
    tokenizer_name: str | None = None
    duckdb_path: str | None = None
    duckdb_table: str | None = None


@dataclass
class TrainingConfig:
    """Config for training execution."""

    action: str = ""
    model_name: str = "gemma-4"
    dataset: str = "dummy"
    epochs: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 2
    backend: str = "jax"
    distributed_strategy: str = "none"
    extra_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class TrainerState:
    """State config for training loops."""

    dataloader: Any = None
    epochs: int = 1
    train_step: Any = None
    params: Any = None
    opt_state: Any = None
    policy_params: Any = None
    ref_params: Any = None
    policy_model: Any = None
    ref_model: Any = None
    optimizer: Any = None
    criterion: Any = None
    device: Any = None
    dummy_batch: Any = None
    beta: float = 0.1
    dataset: str = ""
    learning_rate: float = 0.0
    extra_kwargs: dict[str, object] | None = None
