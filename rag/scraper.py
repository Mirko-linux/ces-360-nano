# rag/scraper.py
import requests
from bs4 import BeautifulSoup
import time

def scrape_result(url, timeout=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')

        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split()[:500])  # Primi 500 parole
    except Exception as e:
        return f"[Errore nella pagina: {str(e)}]"