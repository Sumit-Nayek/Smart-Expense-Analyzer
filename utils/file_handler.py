"""
utils/file_handler.py  —  Production-ready, zero top-level PDF imports.
PDF libraries are imported INSIDE the function so a missing package
never crashes the app at startup.
"""

import pandas as pd


def load_file(uploaded_file):
    """
    Returns (df, raw_text):
      - CSV / Excel  →  (DataFrame, None)
      - PDF          →  (None, extracted_text_str)
    """
    name = uploaded_file.name.lower()

    # ── CSV ──────────────────────────────────────────────────────────────────
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, None

    # ── Excel ────────────────────────────────────────────────────────────────
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df, None

    # ── PDF ──────────────────────────────────────────────────────────────────
    elif name.endswith(".pdf"):
        raw_text = _extract_pdf_text(uploaded_file)
        return None, raw_text

    else:
        return None, "⚠️ Unsupported file type. Please upload CSV, XLSX, or PDF."


# ─────────────────────────────────────────────────────────────────────────────
# PDF helper — tries multiple libraries in order, gracefully falls back
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf_text(uploaded_file) -> str:
    """
    Tries pdfminer.six first (most reliable on Streamlit Cloud),
    then pypdf as fallback.  Both are imported lazily so startup
    never fails even if one is missing.
    """
    import io

    # Read bytes once so both libraries can use it
    file_bytes = uploaded_file.read()

    # ── Attempt 1: pdfminer.six ───────────────────────────────────────────
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        from pdfminer.layout import LAParams
        text = pdfminer_extract(
            io.BytesIO(file_bytes),
            laparams=LAParams()
        )
        if text and text.strip():
            return text.strip()
    except Exception:
        pass  # fall through to next attempt

    # ── Attempt 2: pypdf ─────────────────────────────────────────────────
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [
            page.extract_text()
            for page in reader.pages
            if page.extract_text()
        ]
        text = "\n".join(pages)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # ── Fallback ─────────────────────────────────────────────────────────
    return (
        "⚠️ Could not extract text from this PDF. "
        "It may be a scanned/image-based PDF with no selectable text."
    )
