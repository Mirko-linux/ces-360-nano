# rag/cache.py
import json
import os
import time

CACHE_FILE = "rag/cache.json"
TTL = 7 * 24 * 3600  # 7 giorni

class RAGCache:
    def __init__(self):
        self.cache = {}
        self.load()

    def _hash(self, query):
        from hashlib import sha256
        return sha256(query.encode()).hexdigest()

    def get(self, query):
        key = self._hash(query)
        entry = self.cache.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > TTL:
            del self.cache[key]
            self.save()
            return None
        return entry["results"]

    def set(self, query, results):
        key = self._hash(query)
        self.cache[key] = {
            "results": results,
            "timestamp": time.time()
        }
        self.save()

    def load(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
        else:
            self.cache = {}

    def save(self):
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)