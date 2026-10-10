"""Module docstring."""

from gemma_4_sql.sdk.evaluation import _process_batch_inputs


def test_extract_batch_ids_dict_missing_keys():
    """Docstring for test_extract_batch_ids_dict_missing_keys."""
    inputs, targets = _process_batch_inputs({"foo": "bar"})
    assert inputs == []
    assert targets == []


def test_extract_batch_ids_dict_no_getitem():
    """Docstring for test_extract_batch_ids_dict_no_getitem."""
    inputs, targets = _process_batch_inputs({"inputs": 123, "targets": 456})
    assert inputs == []
    assert targets == []
