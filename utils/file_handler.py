"""
utils/file_handler.py
All PDF imports are LAZY (inside functions) — never crashes at startup.
"""
import pandas as pd


def load_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, None

    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df, None

    elif name.endswith(".pdf"):
        text = _extract_pdf(uploaded_file)
        return None, text

    return None, "⚠️ Unsupported file type."


def _extract_pdf(uploaded_file) -> str:
    import io
    data = uploaded_file.read()

    # Try pdfminer.six first
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        text = extract_text(io.BytesIO(data), laparams=LAParams())
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # Fallback: pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        text = "\n".join(pages)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return "⚠️ Could not extract text. This PDF may be image/scanned."
