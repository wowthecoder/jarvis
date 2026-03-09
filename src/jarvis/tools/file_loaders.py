"""LangChain tools for reading file attachments (PDF, Excel, CSV, text)."""
import os
from langchain_core.tools import tool


@tool
def load_pdf(file_path: str) -> str:
    """Read and extract text from a PDF file. Returns the full text content."""
    from pypdf import PdfReader

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i + 1} ---\n{text}")
        full_text = "\n\n".join(pages)
        # Truncate very large PDFs to avoid context overflow
        max_chars = 50_000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[... truncated ...]"
        return full_text or "No text could be extracted from this PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"


@tool
def load_excel(file_path: str) -> str:
    """Read an Excel (.xlsx/.xls) file and return its contents as text tables."""
    import pandas as pd

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        xl = pd.ExcelFile(file_path)
        output = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            output.append(f"=== Sheet: {sheet_name} ===\n{df.to_string(index=False)}")
        result = "\n\n".join(output)
        max_chars = 30_000
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[... truncated ...]"
        return result
    except Exception as e:
        return f"Error reading Excel file: {e}"


@tool
def load_csv(file_path: str) -> str:
    """Read a CSV file and return its contents as a text table."""
    import pandas as pd

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        df = pd.read_csv(file_path)
        result = df.to_string(index=False)
        max_chars = 20_000
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[... truncated ...]"
        return result
    except Exception as e:
        return f"Error reading CSV file: {e}"


@tool
def read_text_file(file_path: str) -> str:
    """Read a plain text file (txt, py, json, xml, html, md, etc.)."""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        max_chars = 20_000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... truncated ...]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


# Convenience list: tools available to agents that handle file attachments
FILE_TOOLS = [load_pdf, load_excel, load_csv, read_text_file]
