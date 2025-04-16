# utils.py
import fitz  # PyMuPDF
from docx import Document
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from PDF file content (bytes)."""
    text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text += page.get_text("text") + "\n"
        logger.info("Successfully extracted text from PDF.")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Could not process PDF file: {e}")
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    """Extracts text from DOCX file content (bytes)."""
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        logger.info("Successfully extracted text from DOCX.")
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise ValueError(f"Could not process DOCX file: {e}")
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    """Extracts text from TXT file content (bytes)."""
    try:
        # Try common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
             try:
                 text = file_content.decode(encoding)
                 logger.info(f"Successfully extracted text from TXT using {encoding}.")
                 return text
             except UnicodeDecodeError:
                 continue
        # If all fail, raise an error
        raise ValueError("Could not decode TXT file with common encodings (utf-8, latin-1, cp1252).")
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}")
        raise ValueError(f"Could not process TXT file: {e}")

def extract_text(file_content: bytes, file_name: str) -> str:
    """Detects file type and extracts text."""
    lower_filename = file_name.lower()
    if lower_filename.endswith(".pdf"):
        return extract_text_from_pdf(file_content)
    elif lower_filename.endswith(".docx"):
        return extract_text_from_docx(file_content)
    elif lower_filename.endswith(".txt"):
        return extract_text_from_txt(file_content)
    else:
        raise ValueError("Unsupported file type. Please use PDF, DOCX, or TXT.")