"""Provide module docstring."""

import gemma_4_sql.backends.maxtext as m_init

from tests.conftest import requires_maxtext

pytestmark = requires_maxtext


def test_init_get_trainer() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    if m_init.get_trainer() != "maxtext_trainer":
        raise AssertionError
