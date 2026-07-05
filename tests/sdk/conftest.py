from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def mock_build_dataloader():
    with (
        mock.patch("gemma_4_sql.backends.jax.etl.build_dataloader", return_value={"status": "mocked", "dataset": "dummy/data", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.keras.etl.build_dataloader", return_value={"status": "mocked", "dataset": "dummy/data", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.maxtext.etl.build_dataloader", return_value={"status": "mocked", "dataset": "dummy/data", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.mlx.etl.build_dataloader", return_value={"status": "mocked", "dataset": "dummy/data", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.pytorch.etl.build_dataloader", return_value={"status": "mocked", "dataset": "dummy/data", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.jax.train.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.keras.train.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.maxtext.train.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.mlx.train.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.pytorch.train.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.jax.dpo.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.keras.dpo.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.maxtext.dpo.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.mlx.dpo.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
        mock.patch("gemma_4_sql.backends.pytorch.dpo.build_dataloader", return_value={"status": "mocked", "mock_samples": []}),
    ):
        yield
