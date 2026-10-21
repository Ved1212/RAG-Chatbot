"""Step 1: read PDFs into (page_number, text) pieces."""
import io
from pypdf import PdfReader


def load_pdf(filename: str, data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"source": filename, "page": i, "text": text})
    return pages
