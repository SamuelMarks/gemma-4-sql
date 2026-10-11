"""Provide module docstring."""

import gemma_4_sql.backends.maxtext as m_init


def test_init_get_trainer() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    assert m_init.get_trainer() == "maxtext_trainer"
