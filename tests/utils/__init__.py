"""Reusable test utilities and assertion helpers for gemma-4-sql."""

from __future__ import annotations

from tests.utils.db_helpers import create_in_memory_sqlite, get_sample_ddl, insert_sample_users
from tests.utils.tensor_helpers import assert_sequences_close, compute_snr, create_dummy_weight_matrix
from tests.utils.test_utils import FakeDBConnection, FakeDBCursor, RecordingBackendDouble

__all__ = [
    "FakeDBConnection",
    "FakeDBCursor",
    "RecordingBackendDouble",
    "assert_sequences_close",
    "compute_snr",
    "create_dummy_weight_matrix",
    "create_in_memory_sqlite",
    "get_sample_ddl",
    "insert_sample_users",
]
