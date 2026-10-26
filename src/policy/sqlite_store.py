"""Transactional local-disk storage. No transaction spans provider execution."""
import sqlite3
import time
import uuid
import json
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
            if 'source_token' not in {row[1] for row in db.execute('PRAGMA table_info(entries)')}:
                db.execute('ALTER TABLE entries ADD COLUMN source_token TEXT')
            db.execute('CREATE TABLE IF NOT EXISTS semantic_candidates (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, source_key TEXT NOT NULL, source_token TEXT NOT NULL, epoch INTEGER NOT NULL, request TEXT NOT NULL, vector TEXT NOT NULL, bytes INTEGER NOT NULL, touched INTEGER NOT NULL)')

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
            db.execute('INSERT OR REPLACE INTO entries (key,tenant,payload,expires,bytes,touched,source_token) VALUES (?, ?, ?, ?, ?, (SELECT COALESCE(MAX(touched),0)+1 FROM entries), ?)', (key, tenant, payload, now + ttl, size, uuid.uuid4().hex))
            while True:
                count, total = db.execute('SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM entries').fetchone()
                if count <= max_entries and total <= max_bytes:
                    break
                db.execute('DELETE FROM entries WHERE key=(SELECT key FROM entries ORDER BY touched, key LIMIT 1)')
            return True

    def _prune_candidates(self, db):
        db.execute('DELETE FROM semantic_candidates WHERE epoch != ? OR NOT EXISTS (SELECT 1 FROM entries e WHERE e.key=semantic_candidates.source_key AND e.source_token=semantic_candidates.source_token AND e.expires>?)', (self._check(db), self.clock()))

    def admit_candidate(self, *, namespace, source_key, tenant, payload, epoch,
                        request, vector, max_entries, max_bytes):
        """Bind a retrieval record to an exact admission, never to a hit alias."""
        vector = json.dumps(vector, allow_nan=False)
        size = sum(len(s.encode()) for s in (namespace, source_key, request, vector, payload))
        if size > max_bytes:
            return False
        with self._transaction() as db:
            if self._check(db) != epoch:
                return False
            self._prune_candidates(db)
            source = db.execute('SELECT source_token FROM entries WHERE key=? AND tenant=? AND payload=? AND expires>?', (source_key, tenant, payload, self.clock())).fetchone()
            if source is None or source[0] is None:
                return False
            identity = uuid.uuid4().hex
            # One record per source/namespace, bounded globally across namespaces.
            db.execute('DELETE FROM semantic_candidates WHERE namespace=? AND source_key=?', (namespace, source_key))
            db.execute('INSERT INTO semantic_candidates VALUES (?,?,?,?,?,?,?,?,(SELECT COALESCE(MAX(touched),0)+1 FROM semantic_candidates))', (identity, namespace, source_key, source[0], epoch, request, vector, size))
            while True:
                count, total = db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM semantic_candidates').fetchone()
                if count <= max_entries and total <= max_bytes:
                    break
                db.execute('DELETE FROM semantic_candidates WHERE id=(SELECT id FROM semantic_candidates ORDER BY touched,id LIMIT 1)')
            return True

    def recover_candidates(self, namespace, tenant, epoch):
        with self._transaction() as db:
            if self._check(db) != epoch:
                return []
            self._prune_candidates(db)
            return db.execute('SELECT c.id,c.source_key,c.source_token,c.request,c.vector,e.payload,e.expires FROM semantic_candidates c JOIN entries e ON e.key=c.source_key AND e.source_token=c.source_token WHERE c.namespace=? AND e.tenant=? AND c.epoch=? AND e.expires>? ORDER BY c.touched', (namespace, tenant, epoch, self.clock())).fetchall()

    def candidate_current(self, identity, source_key, source_token, namespace, tenant, epoch):
        """Final approval-time snapshot: source, record, expiry and epoch together."""
        with self._transaction() as db:
            if self._check(db) != epoch:
                return False
            return db.execute('SELECT 1 FROM semantic_candidates c JOIN entries e ON e.key=c.source_key AND e.source_token=c.source_token WHERE c.id=? AND c.source_key=? AND c.source_token=? AND c.namespace=? AND e.tenant=? AND c.epoch=? AND e.expires>?', (identity,source_key,source_token,namespace,tenant,epoch,self.clock())).fetchone() is not None

    def invalidate(self, tenant_id=None):
        with self._transaction() as db:
            self._check(db)
            db.execute('UPDATE metadata SET epoch=epoch+1 WHERE id=1')
            db.execute('DELETE FROM semantic_candidates')
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
