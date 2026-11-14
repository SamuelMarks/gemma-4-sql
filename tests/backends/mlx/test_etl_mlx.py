"""Unit and behavioral tests for the MLX ETL pipeline."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import overload
from unittest.mock import patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JSONDict
from tests.conftest import requires_mlx

pytestmark = requires_mlx


class _LengthTokenizer:
    """Typed `TokenizerProtocol` double encoding each string as `[len(text)]`."""

    bos_token_id: int = 2
    eos_token_id: int = 1

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode single string."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
        """Encode string sequence."""

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode string into 1-element length list.

        Returns:
            Encoded length list or nested length lists.
        """
        if not isinstance(text, str):
            return [[len(item)] for item in text]
        return [len(text)]

    def decode(self, token_ids: Sequence[int], /) -> str:
        """Decode token IDs as comma-separated integers.

        Returns:
            Comma-separated token ID string.
        """
        return ",".join(str(t) for t in token_ids)


class _FakeDatasetsModule:
    """Typed `datasets` module double."""

    def load_dataset(self, path: str, *, split: str = "train") -> list[JSONDict]:
        """Return deterministic 1-row dataset."""
        return [{"sql_prompt": f"{path}:{split}", "sql": "SELECT 1"}]


def test_mlx_etl_imports() -> None:
    """Test MLX ETL imports fallback when `datasets` is unavailable."""
    with patch.dict(sys.modules, {"datasets": None}):
        sys.modules.pop("gemma_4_sql.backends.mlx.etl", None)
        import gemma_4_sql.backends.mlx.etl as etl_module

        assert etl_module.datasets is None


def test_pad_batch() -> None:
    """Test `_pad_batch`, `get_trainer`, `_to_nested_scalar_list`, and `_iter_dataset_records` edge cases."""
    import gemma_4_sql.backends.mlx as mlx_pkg
    import gemma_4_sql.backends.mlx.etl as etl_module

    assert mlx_pkg.get_trainer() == "mlx_trainer"
    assert etl_module._to_nested_scalar_list("non_numeric") == 0.0
    assert etl_module._extract_token_lists([{"inputs": "not_seq"}], "inputs") == []

    class _MixedSeqDataset:
        """DatasetSequenceProtocol returning one mapping and one non-mapping row."""

        def __len__(self) -> int:
            """Return `2`."""
            return 2

        def __getitem__(self, idx: int) -> JSONDict | int:
            """Return mapping at `0` and int at `1`."""
            return {"sql_prompt": "q", "sql": "s"} if idx == 0 else 42

    assert list(etl_module._iter_dataset_records(_MixedSeqDataset())) == [{"sql_prompt": "q", "sql": "s"}]
    assert list(etl_module._iter_dataset_records(row for row in [{"sql": "SELECT 1"}])) == [{"sql": "SELECT 1"}]

    inputs = [[1, 2], [1, 2, 3]]
    targets = [[4], [4, 5]]
    res = etl_module._pad_batch(inputs, targets)
    assert res["inputs"] == [[1, 2, 0], [1, 2, 3]]
    assert res["targets"] == [[4, 0], [4, 5]]


def test_mlx_data_loader() -> None:
    """Test `MLXDataLoader` causal sequence formatting, multimodal inputs, and preference batching."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    tok = _LengthTokenizer()
    assert tok.encode(["ab"]) == [[2]]
    assert tok.decode([2]) == "2"

    ds: list[JSONDict] = [
        {"sql_prompt": "p1", "sql": "s1"},
        {"question": "p2", "query": "s2"},
        {"sql_prompt": "p3", "sql": "s3"},
    ]
    loader = etl_module.MLXDataLoader(ds, tok, bs=2)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["inputs"] == [[2, 2, 2], [2, 2, 2]]
    assert batches[0]["targets"] == [[2, 2, 1], [2, 2, 1]]
    assert batches[1]["inputs"] == [[2, 2, 2]]
    assert batches[1]["targets"] == [[2, 2, 1]]

    empty_loader = etl_module.MLXDataLoader([], tok, bs=2)
    assert list(empty_loader) == []

    ds_mm_pref: list[JSONDict] = [
        {"sql_prompt": "p1", "chosen": "SELECT 1", "rejected": "SELECT 2", "image": "img1", "audio": "aud1"},
        {"sql_prompt": "p2", "chosen": "SELECT 3", "rejected": "SELECT 4", "image": "img2", "audio": "aud2"},
    ]
    with (
        patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt", return_value={"prompt": "fmt"}),
        patch("gemma_4_sql.backends.common_multimodal.process_image", return_value={"pixel_values": [1.0]}),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", return_value={"audio_values": [0.5]}),
    ):
        mm_loader = etl_module.MLXDataLoader(ds_mm_pref, tok, bs=2)
        mm_batches = list(mm_loader)
        assert len(mm_batches) == 1
        assert "chosen_inputs" in mm_batches[0]
        assert "rejected_inputs" in mm_batches[0]
        assert mm_batches[0]["pixel_values"] == [[1.0], [1.0]]
        assert mm_batches[0]["audio_values"] == [[0.5], [0.5]]


def test_load_hf_or_duckdb() -> None:
    """Test `_load_hf_or_duckdb` routing and missing dependency handling."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    with patch.object(etl_module, "load_duckdb_dataset", return_value=[{"sql": "SELECT 1"}]):
        res = etl_module._load_hf_or_duckdb("ds", "split", "path", "table")
        assert res == [{"sql": "SELECT 1"}]

    etl_module.datasets = _FakeDatasetsModule()
    res_hf = etl_module._load_hf_or_duckdb("ds", "split", None, None)
    assert res_hf == [{"sql_prompt": "ds:split", "sql": "SELECT 1"}]

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError, match="Datasets dependency is missing"):
        etl_module._load_hf_or_duckdb("ds", "split", None, None)


def test_build_dataloader() -> None:
    """Test `build_dataloader` constructs `MLXDataLoader` and checks `datasets` availability."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    config = ETLConfig(dataset_name="ds", split="train", batch_size=2)
    etl_module.datasets = _FakeDatasetsModule()

    with (
        patch.object(etl_module, "_load_hf_or_duckdb", return_value=[{"sql_prompt": "q", "sql": "SELECT 1"}]),
        patch.object(etl_module, "SQLTokenizer", return_value=_LengthTokenizer()),
    ):
        res = etl_module.build_dataloader(config, duckdb_path="path", duckdb_table="table")
        assert res["dataset"] == "ds"
        assert res["split"] == "train"
        assert res["status"] == "loaded"
        assert res["batch_size"] == 2
        assert res["backend"] == "mlx"
        assert res.get("distributed") is False
        assert isinstance(res.get("loader"), etl_module.MLXDataLoader)

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError, match="Missing datasets"):
        etl_module.build_dataloader(config)
