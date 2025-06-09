import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import io

# Load model (do this once)
@st.cache_resource
def load_llm_pipeline():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return pipe

llm_pipe = load_llm_pipeline()

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(pages)

def ask_llm_to_structure(text, pipe):
    prompt = f"""### Task:
Extract a table of financial transactions from this bank or wallet statement text. 

### Output format (CSV):
Date, Description, Category, Amount, Type

### Text:
{text}

### CSV:"""
    response = pipe(prompt)[0]['generated_text']
    return response.split("### CSV:")[-1].strip()

def csv_text_to_df(csv_text):
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None
