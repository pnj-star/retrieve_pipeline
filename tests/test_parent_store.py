from __future__ import annotations

import asyncio

from retrieve_skill.parent_store import MySQLParentStore, ParentStoreConfig


class _FakeCursor:
    def __init__(self):
        self.sql = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        self.sql = " ".join(sql.split())
        self.params = params

    def fetchall(self):
        return [
            {
                "tenant_id": self.params[0],
                "kb_id": self.params[1],
                "parent_id": self.params[3],
                "status": self.params[2],
                "content": "parent content",
                "doc_version": 1,
            }
        ]


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        return None


class _FakePool:
    def __init__(self, cursor):
        self.cursor = cursor

    def connection(self):
        return _FakeConnection(self.cursor)


def test_parent_store_selects_fields_needed_for_scope_validation() -> None:
    config = ParentStoreConfig(
        host="mysql",
        port=3306,
        database="rag_test",
        user="root",
        table="rag_parent_block",
        status="active",
    )
    cursor = _FakeCursor()
    pool = _FakePool(cursor)
    store = MySQLParentStore(config, pool=pool)
    context = type("Context", (), {"tenant_id": "t1", "kb_id": "kb1"})()

    rows = asyncio.run(store.aget_parent_blocks(["p1", "", "p1"], context=context))

    cursor_sql = cursor.sql
    assert rows == {
        "p1": {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "parent_id": "p1",
            "status": "active",
            "content": "parent content",
            "doc_version": 1,
        }
    }
    assert "tenant_id" in cursor_sql
    assert "kb_id" in cursor_sql
    assert "status" in cursor_sql
