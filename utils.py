import pdfplumber
import openai
import pandas as pd
import io

# Set your OpenAI key here or from Streamlit secrets
openai.api_key = "your-api-key"

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(pages)

def ask_llm_to_structure(text):
    prompt = f"""
You're a financial assistant. Extract a table of transactions from the following text.

Text:
{text}

Please return only a CSV with columns: Date, Description, Category, Amount, Type (Debit/Credit).
Use proper formatting and no explanation.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

def csv_text_to_df(csv_text):
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None
