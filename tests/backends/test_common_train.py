"""Module docstring."""

from gemma_4_sql.backends.common_train import generic_run_training_epochs


def test_generic_run_training_epochs_empty_dataloader():
    """Test with 0 epochs or empty dataloader."""
    # 0 epochs
    result = generic_run_training_epochs(0, [1, 2, 3], lambda x: float(x))
    assert result == 0.0

    # empty dataloader
    result = generic_run_training_epochs(1, [], lambda x: float(x))
    assert result == 0.0


def test_generic_run_training_epochs_standard():
    """Test standard case with multiple epochs."""
    dataloader = [1, 2, 3]
    # In each epoch, sum of loss is 1+2+3=6. Average = 6/3 = 2.0
    result = generic_run_training_epochs(2, dataloader, lambda x: float(x))
    assert result == 2.0


def test_generic_run_training_epochs_single_batch():
    """Test with single batch."""
    dataloader = [42]
    result = generic_run_training_epochs(1, dataloader, lambda x: float(x))
    assert result == 42.0
