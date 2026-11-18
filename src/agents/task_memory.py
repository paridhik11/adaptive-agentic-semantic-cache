"""Bounded local vector retrieval over compatible completed tasks, never answers automatically."""
from contextlib import nullcontext
import hashlib
import json
import threading

from src.policy.encoders import normalize


class LocalTaskMemory:
    VERSION='scoped-task-memory-v1'

    def __init__(self, encoder, *, threshold=.15, top_k=3):
        if not callable(getattr(encoder,'encode',None)) or not isinstance(getattr(encoder,'identity',None),str) or not encoder.identity:
            raise ValueError('identified local memory encoder required')
        if type(threshold) not in (int,float) or not 0<=threshold<=1 or type(top_k) is not int or not 1<=top_k<=3:
            raise ValueError('invalid bounded memory retrieval settings')
        self.encoder,self.threshold,self.top_k=encoder,threshold,top_k
        self.identity=f'{self.VERSION}:{encoder.identity}:{threshold}:{top_k}'
        self._lock=threading.Lock()

    @staticmethod
    def setup(db):
        db.execute('CREATE TABLE IF NOT EXISTS document_memory_vectors (task_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, fingerprint TEXT NOT NULL, vector TEXT NOT NULL, bytes INTEGER NOT NULL, touched INTEGER NOT NULL)')

    def _encode(self, text, ledger):
        with self._lock, ledger.measure('memory_embedding') if ledger is not None else nullcontext():
            raw=self.encoder.encode(text)
            if not 0<len(raw)<=4096:raise ValueError('memory embedding outside bounds')
            return normalize(raw)

    def retrieve(self, db, states, *, query, namespace, ledger):
        # Only caller-selected, scoped, completed states may enter this index.
        db.execute('DELETE FROM document_memory_vectors WHERE task_id NOT IN (SELECT id FROM document_tasks)')
        if not states:return []
        vectors=[]
        for state in states:
            fingerprint=hashlib.sha256(json.dumps([state['question'],state['answer']],ensure_ascii=False).encode()).hexdigest()
            row=db.execute('SELECT vector FROM document_memory_vectors WHERE task_id=? AND namespace=? AND fingerprint=?',
                           (state['id'],namespace,fingerprint)).fetchone()
            if row is None:
                vector=self._encode(state['question'],ledger)
                encoded=json.dumps(vector,allow_nan=False)
                size=len(encoded.encode())
                db.execute('INSERT OR REPLACE INTO document_memory_vectors VALUES (?,?,?,?,?,(SELECT COALESCE(MAX(touched),0)+1 FROM document_memory_vectors))',
                           (state['id'],namespace,fingerprint,encoded,size))
            else:
                vector=normalize(json.loads(row[0]))
            vectors.append((state,vector))
        # Global logical bounds; original completed task records remain intact.
        while True:
            count,size=db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM document_memory_vectors').fetchone()
            if count<=256 and size<=2*1024*1024:break
            db.execute('DELETE FROM document_memory_vectors WHERE task_id=(SELECT task_id FROM document_memory_vectors ORDER BY touched,task_id LIMIT 1)')
        query_vector=self._encode(query,ledger)
        with ledger.measure('memory_search') if ledger is not None else nullcontext():
            ranked=[]
            for state,vector in vectors:
                if len(vector)!=len(query_vector):raise ValueError('memory encoder dimension changed')
                score=sum(a*b for a,b in zip(query_vector,vector))
                if score>=self.threshold:ranked.append((score,state['id'],state))
            ranked.sort(key=lambda row:(-row[0],row[1]))
            return [{'question':s['question'],'answer':s['answer'],
                     'evidence':[{'document_id':t['step']['arguments']['document_id']} for t in s['tools'] if t['step']['tool']=='read'],
                     'source_task_id':sid,'similarity':score,'provenance':'prior_workflow_task_not_independent_truth'}
                    for score,sid,s in ranked[:self.top_k]]
