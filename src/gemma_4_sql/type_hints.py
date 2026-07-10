"""Custom type hints for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONDict = dict[str, JSONValue]

# TensorType is a generic alias for backend-specific tensors (JAX arrays, PyTorch tensors, etc.)
TensorType = TypeVar("TensorType")
ModelType = TypeVar("ModelType")


@dataclass
class DPOConfig:
    """Config for DPO execution."""

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
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

    action: str
    model_name: str
    dataset: str
    epochs: int = 1
    learning_rate: float = 1e-05
    backend: str = "jax"
    distributed_strategy: str = "none"
    extra_kwargs: dict = None


@dataclass
class TrainerState:
    """State config for training loops."""

    dataloader: object = None
    epochs: int = 1
    train_step: object = None
    params: object = None
    opt_state: object = None
    policy_params: object = None
    ref_params: object = None
    policy_model: object = None
    ref_model: object = None
    optimizer: object = None
    criterion: object = None
    device: object = None
    dummy_batch: object = None
    beta: float = 0.1
    dataset: str = ""
    learning_rate: float = 0.0
    extra_kwargs: dict = None
