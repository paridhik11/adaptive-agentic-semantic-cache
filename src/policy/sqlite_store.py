"""Transactional local-disk storage. No transaction spans provider execution."""
import sqlite3
import time
from contextlib import contextmanager


class SQLiteResponseStore:
    """One database per deployment version; wall-clock TTL survives restarts.

    Connections are short-lived. Coordination works across local processes;
    provider single-flight remains process-local. Do not use network filesystems.
    """

    def __init__(self, path, *, namespace, clock=time.time):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("persistent cache requires a deployment namespace")
        if str(path) == ':memory:':
            raise ValueError("use a file for persistent storage")
        self.path, self.namespace, self.clock = str(path), namespace, clock
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), namespace TEXT NOT NULL, epoch INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (1, ?, 0)', (namespace,))
            self._check(db)
            db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL, expires REAL NOT NULL, bytes INTEGER NOT NULL, touched INTEGER NOT NULL)')

    def _check(self, db):
        row = db.execute('SELECT namespace, epoch FROM metadata WHERE id=1').fetchone()
        if row is None or row[0] != self.namespace:
            raise sqlite3.DatabaseError('cache namespace mismatch; use a new database')
        return row[1]

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=1)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def lookup(self, key):
        with self._transaction() as db:
            epoch = self._check(db)
            db.execute('DELETE FROM entries WHERE expires <= ?', (self.clock(),))
            row = db.execute('SELECT payload FROM entries WHERE key=?', (key,)).fetchone()
            if row:
                db.execute('UPDATE entries SET touched=(SELECT COALESCE(MAX(touched),0)+1 FROM entries) WHERE key=?', (key,))
            return (row[0] if row else None), epoch

    def admit(self, key, tenant, payload, epoch, *, ttl, max_entries, max_bytes):
        size = len(payload.encode('utf-8'))
        if size > max_bytes:
            return False
        with self._transaction() as db:
            if self._check(db) != epoch:
                return False
            now = self.clock()
            db.execute('DELETE FROM entries WHERE expires <= ?', (now,))
            db.execute('INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, (SELECT COALESCE(MAX(touched),0)+1 FROM entries))', (key, tenant, payload, now + ttl, size))
            while True:
                count, total = db.execute('SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM entries').fetchone()
                if count <= max_entries and total <= max_bytes:
                    break
                db.execute('DELETE FROM entries WHERE key=(SELECT key FROM entries ORDER BY touched, key LIMIT 1)')
            return True

    def invalidate(self, tenant_id=None):
        with self._transaction() as db:
            self._check(db)
            db.execute('UPDATE metadata SET epoch=epoch+1 WHERE id=1')
            if tenant_id is None:
                return db.execute('DELETE FROM entries').rowcount
            return db.execute('DELETE FROM entries WHERE tenant=?', (tenant_id,)).rowcount

    @property
    def invalidation_epoch(self):
        with self._transaction() as db:
            return self._check(db)

    @property
    def size(self):
        with self._transaction() as db:
            self._check(db)
            db.execute('DELETE FROM entries WHERE expires <= ?', (self.clock(),))
            return db.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
