# ocr/ocr_utils.py
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

def ocr_image(img_path):
    try:
        return pytesseract.image_to_string(Image.open(img_path))
    except Exception as e:
        return f"[ERRORE OCR IMMAGINE: {e}]"

def ocr_pdf_scanned(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        full_text = ""
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            full_text += f" [PAGINA {i+1}]\n{text}\n"
        return full_text
    except Exception as e:
        return f"[ERRORE OCR PDF SCANSIONATO: {e}]"