# core/attachment_handler.py
import os

from ocr.pdf_to_text import extract_text_from_pdf
from ocr.docx_to_text import extract_text_from_docx
from ocr.ocr_utils import ocr_pdf_scanned, ocr_image

def process_attachment(file_path):
    """Estrae testo da qualsiasi allegato"""
    if not os.path.exists(file_path):
        return "[ERRORE: File non trovato]"

    _, ext = os.path.splitext(file_path.lower())

    try:
        if ext == ".pdf":
            # Prova prima come PDF testuale
            text = extract_text_from_pdf(file_path)
            if len(text.strip()) < 100:  # Probabilmente scansionato
                text = ocr_pdf_scanned(file_path)
            return text

        elif ext == ".docx":
            return extract_text_from_docx(file_path)

        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            return ocr_image(file_path)

        else:
            return f"[Formato non supportato: {ext}]"

    except Exception as e:
        return f"[ERRORE ELABORAZIONE: {e}]"