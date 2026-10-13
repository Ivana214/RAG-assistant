import pymupdf


class PDFProcessingError(ValueError):
    pass


def extract_pdf_pages(content: bytes) -> dict:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError("无法解析 PDF 文件") from exc

    try:
        if document.needs_pass and not document.authenticate(""):
            raise PDFProcessingError("PDF 已加密，无法读取")

        pages = []
        for index, page in enumerate(document, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            text = text.strip()
            if text:
                pages.append({"page_number": index, "text": text})

        return {"total_pages": document.page_count, "pages": pages}
    finally:
        document.close()
