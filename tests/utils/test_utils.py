"""Unit tests for testing utilities in tests/utils."""

from __future__ import annotations

from typing import Any

import pytest

from tests.utils import (
    assert_sequences_close,
    compute_snr,
    create_dummy_weight_matrix,
    create_in_memory_sqlite,
    get_sample_ddl,
    insert_sample_users,
)


def test_db_helpers() -> None:
    """Test db helper functions."""
    ddl = get_sample_ddl()
    assert "CREATE TABLE users" in ddl
    conn = create_in_memory_sqlite(ddl)
    insert_sample_users(conn, [(1, "Alice", "alice@example.com")])
    cur = conn.cursor()
    res = cur.execute("SELECT name FROM users WHERE id=1;").fetchone()
    assert res[0] == "Alice"
    cur.close()
    conn.close()


def test_tensor_helpers() -> None:
    """Test tensor comparison helpers."""
    mat = create_dummy_weight_matrix(2, 3)
    assert len(mat) == 2
    assert len(mat[0]) == 3

    assert_sequences_close([1.0, 2.0], [1.0001, 1.9999], atol=1e-3)

    with pytest.raises(AssertionError, match="Length mismatch"):
        assert_sequences_close([1.0], [1.0, 2.0])

    with pytest.raises(AssertionError, match="Divergence at index"):
        assert_sequences_close([1.0], [2.0], atol=1e-5)

    snr_val = compute_snr([1.0, 2.0], [1.0, 2.0])
    assert snr_val == float("inf")

    snr_noise = compute_snr([10.0, 10.0], [9.9, 10.1])
    assert snr_noise > 0.0

    snr_zero = compute_snr([0.0, 0.0], [1.0, 1.0])
    assert snr_zero == 0.0

    with pytest.raises(ValueError, match="Sequence lengths must match"):
        compute_snr([1.0], [1.0, 2.0])


class FakeDBCursor:
    """Fake DB-API 2.0 cursor for testing."""

    def __init__(self, connection: FakeDBConnection) -> None:
        """Initialize fake cursor.

        Args:
            connection: The parent connection.
        """
        self.connection = connection
        self.arraysize = 1
        self.rowcount = -1
        self.description: Any = None
        self._results: list[Any] = []
        self._closed = False

    def execute(self, operation: str, parameters: Any = None) -> FakeDBCursor:
        """Fake execute method.

        Args:
            operation: SQL query.
            parameters: Query parameters.

        Returns:
            Self.

        Raises:
            Exception: If the query matches the failure trigger.
        """
        if self._closed:
            raise RuntimeError("Cursor closed")
        if operation == self.connection._fail_query:
            raise RuntimeError("Simulated execution failure")
        self._results = self.connection._results_queue.pop(0) if self.connection._results_queue else []
        self.rowcount = len(self._results)
        if self._results:
            self.description = True
        return self

    def fetchall(self) -> list[Any]:
        """Fetch all results.

        Returns:
            List of result rows.
        """
        if self._closed:
            raise RuntimeError("Cursor closed")
        return self._results

    def close(self) -> None:
        """Close the cursor."""
        self._closed = True


class FakeDBConnection:
    """Fake DB-API 2.0 connection for testing."""

    def __init__(self) -> None:
        """Initialize fake connection."""
        self._closed = False
        self._transactions_committed = 0
        self._transactions_rolled_back = 0
        self._results_queue: list[list[Any]] = []
        self._fail_query: str | None = None

    def cursor(self) -> FakeDBCursor:
        """Create a new fake cursor.

        Returns:
            A FakeDBCursor instance.
        """
        if self._closed:
            raise RuntimeError("Connection closed")
        return FakeDBCursor(self)

    def commit(self) -> None:
        """Commit the current transaction."""
        self._transactions_committed += 1

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._transactions_rolled_back += 1

    def close(self) -> None:
        """Close the connection."""
        self._closed = True

    def set_results(self, results: list[list[Any]]) -> None:
        """Set results for subsequent queries.

        Args:
            results: List of lists containing rows to return.
        """
        self._results_queue.extend(results)

    def set_fail_query(self, query: str) -> None:
        """Configure the connection to fail when executing a specific query.

        Args:
            query: The exact SQL query that should trigger an exception.
        """
        self._fail_query = query


def test_seed_table() -> None:
    """Test seed_table."""
    from tests.utils.db_helpers import seed_table

    conn = create_in_memory_sqlite("CREATE TABLE t (a INT, b TEXT);")
    seed_table(conn, "t", ["a", "b"], [(1, "A"), (2, "B")])
    cur = conn.cursor()
    assert len(cur.execute("SELECT * FROM t").fetchall()) == 2
    cur.close()
    conn.close()


def test_fake_db() -> None:
    """Test fake DB classes."""
    conn = FakeDBConnection()
    conn.set_results([[("row1",)], [("row2",)]])
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchall() == [("row1",)]
    assert cur.rowcount == 1
    conn.commit()
    conn.rollback()
    assert conn._transactions_committed == 1
    assert conn._transactions_rolled_back == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(Exception, match="Simulated execution failure"):
        cur.execute("FAIL")

    cur.close()
    with pytest.raises(Exception, match="Cursor closed"):
        cur.execute("SELECT 1")
    with pytest.raises(Exception, match="Cursor closed"):
        cur.fetchall()

    conn.close()
    with pytest.raises(Exception, match="Connection closed"):
        conn.cursor()


def test_multi_dialect_ddl() -> None:
    """Test multi-dialect DDL."""
    assert "SERIAL" in get_sample_ddl("postgres")
    assert "UINTEGER" in get_sample_ddl("duckdb")


def test_assert_tensors_close() -> None:
    """Test assert_tensors_close."""
    from tests.utils.tensor_helpers import assert_tensors_close

    class DummyTensor:
        """Dummy tensor for testing assert_tensors_close."""

        def __init__(self, data: Any) -> None:
            """Initialize DummyTensor."""
            self.data = data
            self.shape = (len(data),)

        def flatten(self) -> DummyTensor:
            """Return flattened tensor."""
            return self

        def tolist(self) -> Any:
            """Return tensor elements as list."""
            return self.data

    t1 = DummyTensor([1.0, 2.0])
    t2 = DummyTensor([1.0001, 1.9999])
    assert_tensors_close(t1, t2, backend="pytorch")

    with pytest.raises(ValueError):
        assert_tensors_close(t1, t2, backend="unknown")


def test_assert_tensors_close_jax() -> None:
    """Test assert_tensors_close for jax/mlx backends."""

    from tests.utils.tensor_helpers import assert_tensors_close

    class MockTensor:
        """Mock tensor with __array__ protocol."""

        def __init__(self, data: Any) -> None:
            """Initialize MockTensor."""
            self.data = data

        def __array__(self) -> Any:
            """Return numpy array representation."""
            import numpy as np

            return np.array(self.data)

    t1 = MockTensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = MockTensor([[1.0, 2.0], [3.0, 4.0]])

    assert_tensors_close(t1, t2, backend="jax")
    assert_tensors_close(t1, t2, backend="mlx")
