# build_corpus.py
import os
import PyPDF2
import docx
from tqdm import tqdm

# Crea le cartelle
os.makedirs("dataset/raw", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)

def extract_text_from_pdf(path):
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text
    except:
        return "[ERRORE: PDF non leggibile]"

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return "[ERRORE: DOCX non leggibile]"

# Elabora tutti i file nella cartella raw
all_text = ""

files = os.listdir("dataset/raw")
for filename in tqdm(files, desc="Elaborando file"):
    filepath = os.path.join("dataset/raw", filename)
    print(f"📄 Elaborando: {filename}")
    
    text = ""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(filepath)
    elif filename.lower().endswith(".docx"):
        text = extract_text_from_docx(filepath)
    else:
        text = "[TIPO DI FILE NON SUPPORTATO]"

    all_text += f"\n\n[DOCUMENTO: {filename}]\n{text}\n[/DOCUMENTO]\n"

# Salva il corpus finale
output_path = "dataset/processed/corpus_leonese.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text.strip())

print(f"✅ Corpus creato! Salvato in: {output_path}")
print(f"📈 Totale caratteri: {len(all_text)}")