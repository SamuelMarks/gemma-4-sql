"""Module docstring."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_failure_modes():
    """Docstring for test_failure_modes."""
    with pytest.raises(DependencyMissingError):
        raise DependencyMissingError("miss")
    with pytest.raises(InferenceError):
        raise InferenceError("err")
