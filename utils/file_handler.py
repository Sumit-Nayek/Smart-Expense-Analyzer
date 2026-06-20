# utils/file_handler.py

# ✅ USE PyPDF2 instead — already in your requirements.txt
import pypdf   
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        raw_text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        return None, raw_text



