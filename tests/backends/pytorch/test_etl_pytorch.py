"""Tests for PyTorch-specific ETL pipeline."""


def test_build_dataloader_pytorch_mocked() -> None:
    """Test PyTorch build_dataloader when libraries are missing via direct assignment."""
    etl_mod = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    orig_torch = etl_mod.torch
    try:
        etl_mod.torch = None  # type: ignore[attr-defined]
        res = etl_mod.build_dataloader("dummy/data", "train", 16, distributed=False)
        if not res["backend"] == "pytorch":
            raise AssertionError
        if not res["status"] == "mocked":
            raise AssertionError
        if "mock_samples" not in res:
            raise AssertionError
    finally:
        etl_mod.torch = orig_torch  # type: ignore[attr-defined]
