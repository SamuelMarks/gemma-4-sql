"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_duckdb_adapter(monkeypatch):
    """Docstring for test_duckdb_adapter."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as da

    mock_duckdb = MagicMock()
    mock_duckdb.Error = ValueError

    # Reload with mock duckdb
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        da.duckdb = mock_duckdb

        # error_classes
        assert da.DuckDBAdapter("path", {}).error_classes == (ValueError,)

        # connect existing_conn
        adapter1 = da.DuckDBAdapter("path", {"existing_conn": "conn1"})
        assert adapter1.connect() == "conn1"
        assert await adapter1.connect_async() == "conn1"

        # connect conn
        adapter2 = da.DuckDBAdapter("path", {"conn": "conn2"})
        assert adapter2.connect() == "conn2"

        # connect readonly
        adapter3 = da.DuckDBAdapter("path", {}, read_only=True)
        adapter3.connect()
        mock_duckdb.connect.assert_called_with("path", read_only=True)

        # setup_schema existing_conn
        adapter1.setup_schema("DDL")

        # setup_schema normal
        adapter3.conn = MagicMock()
        adapter3.setup_schema("DDL")
        adapter3.conn.execute.assert_called_with("DDL")

        # close existing_conn
        adapter1.close()

        # close normal
        adapter3.close()
        adapter3.conn.close.assert_called()

        # execute_with_feedback_async success
        adapter3.conn.cursor.return_value.execute.return_value.fetchall.return_value = [("row",)]
        res_fw = await adapter3.execute_with_feedback_async("SELECT")
        assert res_fw == (True, [("row",)], None)

        # execute_with_feedback_async failure
        adapter3.conn.cursor.return_value.execute.side_effect = ValueError("err")
        res_fw_err = await adapter3.execute_with_feedback_async("SELECT")
        assert res_fw_err == (False, [], "err")

        # execute_query_async success without cursor
        adapter3.conn.cursor = None
        del adapter3.conn.cursor
        adapter3.conn.execute.return_value.fetchall.return_value = [("row2",)]
        res_q = await adapter3.execute_query_async("SELECT")
        assert res_q == [("row2",)]

        # execute_query_async failure
        adapter3.conn.execute.side_effect = ValueError("err2")
        res_q_err = await adapter3.execute_query_async("SELECT")
        assert res_q_err == []

    # test missing duckdb
    with patch.dict("sys.modules", {"duckdb": None}):
        da.duckdb = None
        with pytest.raises(ImportError):
            da.DuckDBAdapter("path", {}).connect()
