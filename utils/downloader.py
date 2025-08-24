# build_corpus.py
import os
import PyPDF2
import docx
from PIL import Image
import pytesseract
import gdown
from glob import glob
from tqdm import tqdm

# Config
RAW_DIR = "dataset/raw"
PROCESSED_FILE = "dataset/processed/corpus_leonese.txt"

# Assicurati che le cartelle esistano
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)

# Lista di ID Google Drive dei tuoi file (presi dal tuo link)
# Estrai gli ID dai link: https://drive.google.com/file/d/1abc123/view → 1abc123
FILE_ID_MAP = {
    "Costituzione_di_Leonia.pdf": "1zMLG6YqYA4gnK69AWDev66NBMjh7Xb7T",  # cartella, non file singolo
    # Ma visto che non puoi elencarli tutti, usiamo un workaround
}

# 🔴 WORKAROUND: Non possiamo scaricare l'intera cartella con gdown direttamente
# Soluzione: tu esporti la lista degli ID (una volta sola) oppure usiamo un metodo furbo

print("⚠️ Attenzione: gdown non scarica cartelle intere.")
print("Per ora, aggiungiamo manualmente alcuni ID noti (esempio).")
print("Per automatizzare tutto, dovrai esportare gli ID dei file.")

# Esempio di ID reali (sostituisci con quelli veri quando li hai)
EXAMPLE_FILES = {
    "Costituzione di Leonia_(0).pdf": "1abc123...",  # ← SOSTITUISCI CON ID VERI
    "Fondamenti di Micronazionalismo Leonense.pdf": "1def456...",
    "Storia Leonense - Volume 1.pdf": "1ghi789..."
}

def extract_text_from_pdf(path):
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"[ERRORE PDF: {e}]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"[ERRORE DOCX: {e}]"

def extract_text_from_image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        return f"[ERRORE OCR: {e}]"

# Scarica ed elabora
all_text = ""

for filename, file_id in EXAMPLE_FILES.items():
    print(f"📥 Scaricando: {filename}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(RAW_DIR, filename)
    
    try:
        gdown.download(url, output_path, quiet=False)
        
        print(f"🔍 Estrazione testo da {filename}...")
        text = ""
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(output_path)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(output_path)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(output_path)
        
        # Aggiungi al corpus
        all_text += f"\n\n[DOCUMENTO: {filename}]\n{text}\n[/DOCUMENTO]\n"
        
    except Exception as e:
        print(f"❌ Errore con {filename}: {e}")

# Salva il corpus
with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
    f.write(all_text.strip())

print(f"✅ Corpus creato! Salvato in: {PROCESSED_FILE}")
print(f"📏 Dimensione: {len(all_text)} caratteri")