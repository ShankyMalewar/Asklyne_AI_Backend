from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

class OCRHandler:
    def __init__(self):
        # Set tesseract path manually if needed (for Windows)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text_from_image(self, image_path: str) -> str:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        pages = convert_from_path(pdf_path)
        full_text = ""
        for page in pages:
            text = pytesseract.image_to_string(page)
            full_text += text + "\n"
        return full_text.strip()
