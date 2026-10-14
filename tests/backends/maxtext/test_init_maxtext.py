"""Unit and behavioral tests for init maxtext."""

import gemma_4_sql.backends.maxtext as m_init


def test_init_get_trainer() -> None:
    """Execute test helper or verification step.

    Raises:
        AssertionError: Description.

    """
    assert m_init.get_trainer() == "maxtext_trainer"
