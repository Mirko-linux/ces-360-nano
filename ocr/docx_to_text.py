# ocr/docx_to_text.py
import docx

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"[ERRORE DOCX: {e}]"