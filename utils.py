# import pdfplumber
import pandas as pd
# import requests
import io
import pymupdf  # PyMuPDF

def extract_text_from_pdf(file):
    pdf_doc = pymupdf.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in pdf_doc:
        full_text += page.get_text()
    return full_text


# Function to call Hugging Face Inference API
def ask_hf_inference_api(text, hf_token, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    headers = {"Authorization": f"Bearer {hf_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    prompt = f"""### Task:
Extract a table of financial transactions from this text.

### Output format (CSV):
Date, Description, Category, Amount, Type

### Text:
{text}

### CSV:"""

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    try:
        return response.json()[0]["generated_text"].split("### CSV:")[-1].strip()
    except:
        return "ERROR: " + str(response.json())

# Function to parse CSV output to DataFrame
def csv_text_to_df(csv_text):
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None
def csv_text_to_df(csv_text):
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None
