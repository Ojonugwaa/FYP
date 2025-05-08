

# import fitz  # PyMuPDF
# import docx2txt
# import tempfile

# def extract_text_from_cv(file_bytes: bytes, filename: str) -> str:
#     if filename.lower().endswith(".pdf"):
#         with fitz.open(stream=file_bytes, filetype="pdf") as doc:
#             return " ".join([page.get_text() for page in doc])
#     elif filename.lower().endswith(".docx"):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
#             tmp.write(file_bytes)
#             tmp.flush()
#             return docx2txt.process(tmp.name)
#     else:
#         raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")


import fitz
import docx2txt
import tempfile
import pytesseract
from PIL import Image
import io

def extract_text_from_cv(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                page_text = page.get_text()
                if not page_text.strip():
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    page_text = pytesseract.image_to_string(img)
                text += page_text + " "
            return text.strip()
    elif filename.lower().endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return docx2txt.process(tmp.name)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")