from unittest import mock

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.sdk.registry import get_backend


def test_registry_fallback():
    # Force _ENTRY_POINTS to None so we enter the block again
    mod._ENTRY_POINTS = None
    with mock.patch("sys.version_info", (3, 11)):
        try:
            get_backend("nonexistent")
        except Exception:
            pass

    mod._ENTRY_POINTS = None
    with mock.patch("sys.version_info", (3, 9)):
        try:
            get_backend("nonexistent")
        except Exception:
            pass
