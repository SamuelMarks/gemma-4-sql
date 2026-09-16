"""Unit tests for testing utilities in tests/utils."""

from __future__ import annotations

import pytest

from tests.utils import (
    MockCausalModel,
    MockIterableDataLoader,
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


def test_mock_generators() -> None:
    """Test mock model and dataloader generators."""
    model = MockCausalModel("SELECT 42;")
    assert model.generate("prompt") == "SELECT 42;"

    batches = [{"x": 1}, {"x": 2}]
    loader = MockIterableDataLoader(batches)
    assert len(loader) == 2
    assert list(loader) == batches
