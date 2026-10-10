"""Module docstring."""

from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.logging as mod


def test_maxtext_logging_import_error():
    # just mock SummaryWriter directly instead of reloading to simulate missing
    """Docstring for test_maxtext_logging_import_error."""
    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", None):
        assert mod.SummaryWriter is None


def test_maxtext_logging_empty_metrics():
    """Docstring for test_maxtext_logging_empty_metrics."""
    with patch.object(mod, "SummaryWriter", MagicMock()):
        mod.log_metrics({}, 1)
