# rag/rag_agent.py
from .search import search_reliable
from .scraper import scrape_result
from .cache import RAGCache
from core.attachment_handler import process_attachment

LEONIA_KEYWORDS = ["leonia", "ghelda", "agepoli", "zafiria", "zanardi", "orlando",
                   "costituzione", "pdl", "pmc", "senato", "legge", "decreto",
                   "micronazione", "lumenaria", "ostracismo", "magistratura"]

class RAGAgent:
    def __init__(self):
        self.cache = RAGCache()

    def should_use_rag(self, query):
        return not any(kw in query.lower() for kw in LEONIA_KEYWORDS)

    def generate_context(self, query, attachment_path=None):
        context = ""

        # 1. Allegato
        if attachment_path:
            context += f"[ALLEGATO]\n{process_attachment(attachment_path)}\n\n"

        # 2. RAG se non è argomento leonese
        if self.should_use_rag(query):
            cached = self.cache.get(f"rag_{query}")
            if cached:
                context += f"[RICERCA - CACHE]\n{cached}\n"
            else:
                results = search_reliable(query)
                if not results:
                    context += "\n[NESSUNA INFORMAZIONE TROVATA ONLINE]\n"
                else:
                    combined = ""
                    for r in results:
                        text = scrape_result(r["href"])
                        combined += f"[{r['title']}] {text}\n\n"
                    context += f"[RICERCA]\n{combined}"
                    self.cache.set(f"rag_{query}", combined)

        return context