"""Module docstring."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest


@pytest.mark.asyncio
async def test_postgres_adapter(monkeypatch):
    """Docstring for test_postgres_adapter."""
    import importlib

    import gemma_4_sql.sdk.adapters.postgres_adapter as pa

    # 27->31, 31->35, 36: error_classes with missing psycopg2 and asyncpg
    with patch.dict("sys.modules", {"psycopg2": None, "asyncpg": None}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        assert adapter.error_classes == (Exception,)
        with pytest.raises(ImportError):
            adapter.connect()
        with pytest.raises(ImportError):
            await adapter.connect_async()

    # 29->31: psycopg2 present but err is not exception subclass
    mock_psycopg2 = MagicMock()
    mock_psycopg2.Error = int
    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "asyncpg": None}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        assert adapter.error_classes == (Exception,)

    # both psycopg2 and asyncpg present with valid Error subclass
    class DummyPsy(Exception):
        """Docstring for DummyPsy."""

    mock_psycopg2_ok = MagicMock()
    mock_psycopg2_ok.Error = DummyPsy

    class DummyApg(Exception):
        """Docstring for DummyApg."""

    mock_apg_ok = MagicMock()
    mock_apg_ok.PostgresError = DummyApg

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2_ok, "asyncpg": mock_apg_ok}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        err_tuple = adapter.error_classes
        assert DummyPsy in err_tuple
        assert DummyApg in err_tuple

    # AttributeError inside error_classes
    mock_apg_bad = MagicMock()
    del mock_apg_bad.PostgresError
    with patch.dict("sys.modules", {"psycopg2": None, "asyncpg": mock_apg_bad}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        assert adapter.error_classes == (Exception,)

    # reload pa normally
    importlib.reload(pa)

    # exception in execute_query_async
    adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
    with patch.object(pa.PostgresAdapter, "error_classes", new_callable=PropertyMock, return_value=(ValueError,)):

        async def mock_connect(*a, **k):
            """Docstring for mock_connect."""
            m = MagicMock()
            m.fetch.side_effect = ValueError("err")

            async def close():
                """Docstring for close."""

            m.close = close
            return m

        adapter.connect_async = mock_connect
        res = await adapter.execute_query_async("SELECT")
        assert res == []

    # test success execute_query_async
    async def mock_connect_success(*a, **k):
        """Docstring for mock_connect_success."""

        class MockRecord:
            """Docstring for MockRecord."""

            def values(self):
                """Docstring for values."""
                return [1, 2]

        m = MagicMock()

        async def fetch(*a, **k):
            """Docstring for fetch."""
            return [MockRecord()]

        m.fetch = fetch

        async def execute(*a, **k):
            """Docstring for execute."""

        m.execute = execute

        async def close():
            """Docstring for close."""

        m.close = close
        return m

    adapter.connect_async = mock_connect_success
    res_succ = await adapter.execute_query_async("SELECT")
    assert res_succ == [(1, 2)]

    res_fw = await adapter.execute_with_feedback_async("SELECT")
    assert res_fw == (True, [(1, 2)], None)

    async def mock_connect_err(*a, **k):
        """Docstring for mock_connect_err."""
        m = MagicMock()

        async def fetch(*a, **k):
            """Docstring for fetch."""
            raise ValueError("err")

        m.fetch = fetch

        async def execute(*a, **k):
            """Docstring for execute."""
            raise ValueError("err")

        m.execute = execute

        async def close():
            """Docstring for close."""

        m.close = close
        return m

    adapter.connect_async = mock_connect_err
    res_fw_err = await adapter.execute_with_feedback_async("SELECT")
    assert res_fw_err == (False, [], "err")

    # test sync connect
    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2_ok}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        adapter.db_path = "path"
        adapter.db_kwargs = {}
        pa.psycopg2.connect = MagicMock(return_value="conn")
        assert adapter.connect() == "conn"

        adapter2 = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        adapter2.db_path = "path"
        adapter2.db_kwargs = {"user": "u", "password": "p"}
        adapter2.connect()
        pa.psycopg2.connect.assert_called_with("path", user="u", password="p")

        adapter3 = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        adapter3.db_path = ":memory:"
        adapter3.db_kwargs = {}
        adapter3.connect()
        pa.psycopg2.connect.assert_called_with()

        pa.psycopg2.connect.side_effect = Exception("err")
        with pytest.raises(Exception):
            adapter.connect()

    # test async connect
    with patch.dict("sys.modules", {"asyncpg": mock_apg_ok}):
        importlib.reload(pa)
        adapter = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        adapter.db_path = "path"
        adapter.db_kwargs = {}

        async def _mock_ac(*a, **k):
            """Docstring for _mock_ac."""
            return "aconn"

        pa.asyncpg.connect = getattr(pytest, "AsyncMock", MagicMock)(side_effect=_mock_ac)

        res = await adapter.connect_async()
        assert res == "aconn"

        adapter2 = pa.PostgresAdapter.__new__(pa.PostgresAdapter)
        adapter2.db_path = ":memory:"
        adapter2.db_kwargs = {}
        res2 = await adapter2.connect_async()
        assert res2 == "aconn"

    # test setup_schema
    adapter.conn = MagicMock()
    adapter.setup_schema("DDL")
    adapter.conn.cursor.return_value.execute.assert_called_with("DDL")

    adapter.conn.cursor.return_value.execute.side_effect = ValueError("err")
    with pytest.raises(ValueError):
        adapter.setup_schema("DDL")
