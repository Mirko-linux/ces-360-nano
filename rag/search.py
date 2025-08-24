# rag/search.py
from duckduckgo_search import DDGS
import time
import random

def search_reliable(query, max_results=5, retries=3):
    """
    Cerca con duckduckgo_search, con retry e fallback
    """
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, 
                    region="it-it", 
                    safesearch="moderate", 
                    timelimit=None,
                    max_results=max_results
                ))
            # Converti in formato semplice
            return [
                {
                    "title": r["title"],
                    "href": r["href"],
                    "body": r.get("body", "")[:300]
                }
                for r in results
            ]
        except Exception as e:
            print(f"[Ricerca] Tentativo {attempt + 1} fallito: {e}")
            time.sleep(random.uniform(1, 3))  # Pausa casuale
            continue

    print("[Ricerca] Tutti i tentativi falliti")
    return None