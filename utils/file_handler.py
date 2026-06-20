"""
file_handler.py
PDF uses pdfminer.six — confirmed pre-installed on Streamlit Cloud.
CSV/Excel uses pandas + openpyxl — also pre-installed.
"""

import pandas as pd
from io import StringIO

# PDF extraction via pdfminer.six (pre-installed on Streamlit Cloud)
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def load_file(uploaded_file):
    """
    Returns (df, raw_text):
      - CSV        → (DataFrame, None)
      - Excel      → (DataFrame, None)
      - PDF        → (None, str)
    """
    filename = uploaded_file.name.lower()

    # ── CSV ──────────────────────────────────────────────────────────────
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, None

    # ── Excel ─────────────────────────────────────────────────────────────
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df, None

    # ── PDF ───────────────────────────────────────────────────────────────
    elif filename.endswith(".pdf"):
        import io
        # pdfminer needs a file-like object; uploaded_file works directly
        raw_text = extract_text(uploaded_file, laparams=LAParams())
        if not raw_text or not raw_text.strip():
            raw_text = "Could not extract text from this PDF. It may be a scanned image."
        return None, raw_text.strip()

    else:
        return None, "Unsupported file type. Please upload CSV, XLSX, or PDF."
