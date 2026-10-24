"""Unit and behavioral tests for init maxtext."""

import gemma_4_sql.backends.maxtext as m_init


def test_init_get_trainer() -> None:
    """Verify init get trainer behavior."""
    assert m_init.get_trainer() == "maxtext_trainer"
