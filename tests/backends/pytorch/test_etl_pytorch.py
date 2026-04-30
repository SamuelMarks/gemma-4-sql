"""
Tests for PyTorch-specific ETL pipeline.
"""


def test_build_dataloader_pytorch_mocked() -> None:
    """Test PyTorch build_dataloader when libraries are missing via direct assignment."""
    import gemma_4_sql.backends.pytorch.etl as etl_mod

    # Store original
    orig_torch = etl_mod.torch

    try:
        # Force mock mode
        etl_mod.torch = None

        res = etl_mod.build_dataloader("dummy/data", "train", 16, False)
        assert res["backend"] == "pytorch"
        assert res["status"] == "mocked"
        assert "mock_samples" in res

    finally:
        # Restore
        etl_mod.torch = orig_torch
