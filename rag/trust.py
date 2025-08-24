# rag/trust.py
TRUSTED_DOMAINS = {
    "wikipedia.org": 10,
    "repubblica.it": 8,
    "corriere.it": 8,
    "ansa.it": 9,
    "rai.it": 9,
    "giustizia.it": 10,
    "leonia.micronazioni.it": 100,
    "micronazioni.it": 8,
    "studenti.it": 6,
    "treccani.it": 10,
}

def get_trust_score(url):
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url:
            return score
    return 3  # bassa affidabilità