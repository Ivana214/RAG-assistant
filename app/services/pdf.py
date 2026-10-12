import io

from pypdf import PdfReader


class PDFProcessingError(ValueError):
    pass


def extract_pdf_pages(content: bytes) -> dict:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise PDFProcessingError("无法解析 PDF 文件") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PDFProcessingError("PDF 已加密，无法读取") from exc

    pages = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append({"page_number": index, "text": text})

    return {"total_pages": len(reader.pages), "pages": pages}
