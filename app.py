# import streamlit as st
# import pandas as pd
# import pdfplumber  # Replaced PyPDF2 with pdfplumber
# import io
# import re
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans

# # Set page config
# st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

# # Function to extract text from PDF using pdfplumber
# def extract_text_from_pdf(pdf_file):
#     try:
#         with pdfplumber.open(pdf_file) as pdf:
#             text = ""
#             for page in pdf.pages:
#                 # Try extracting plain text
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + "\n"
#                 # If text is insufficient, try extracting tables
#                 if not extracted_text or len(extracted_text.strip()) < 50:
#                     tables = page.extract_tables()
#                     for table in tables:
#                         for row in table:
#                             # Join row cells into a single string
#                             text += " ".join(str(cell) for cell in row if cell) + "\n"
#             return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#         return ""

# # Function to parse transactions from PDF text
# def parse_pdf_transactions(text):
#     transactions = []
#     # Simple regex to extract transaction-like patterns (customize based on your PDF format)
#     pattern = r'(\d{2}/\d{2}/\d{4})\s+([^\d\n]+)\s+([\d,.]+)'
#     matches = re.findall(pattern, text, re.MULTILINE)
    
#     for match in matches:
#         date, description, amount = match
#         try:
#             amount = float(amount.replace(',', ''))
#             transactions.append({
#                 'Date': pd.to_datetime(date, format='%d/%m/%Y'),
#                 'Description': description.strip(),
#                 'Amount': amount
#             })
#         except ValueError:
#             continue
#     return transactions

# # Function to read CSV transactions
# def read_csv_transactions(csv_file):
#     try:
#         df = pd.read_csv(csv_file)
#         if 'Date' not in df.columns or 'Description' not in df.columns or 'Amount' not in df.columns:
#             st.error("CSV must contain 'Date', 'Description', and 'Amount' columns")
#             return []
#         df['Date'] = pd.to_datetime(df['Date'])
#         return df.to_dict('records')
#     except Exception as e:
#         st.error(f"Error reading CSV: {e}")
#         return []

# # Function to categorize transactions using KMeans clustering
# def categorize_transactions(transactions):
#     if not transactions:
#         return []
    
#     df = pd.DataFrame(transactions)
#     descriptions = df['Description'].values
#     vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
#     X = vectorizer.fit_transform(descriptions)
    
#     # Apply KMeans clustering
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     df['Category'] = kmeans.fit_predict(X)
    
#     # Map cluster numbers to meaningful categories
#     category_map = {
#         0: 'Groceries',
#         1: 'Utilities',
#         2: 'Entertainment',
#         3: 'Transportation',
#         4: 'Miscellaneous'
#     }
#     df['Category'] = df['Category'].map(category_map)
#     return df.to_dict('records')

# # Function to plot expense trends
# def plot_expense_trends(df):
#     if df.empty:
#         st.warning("No data to plot")
#         return
    
#     # Monthly expenses
#     df['Month'] = df['Date'].dt.to_period('M')
#     monthly_expenses = df.groupby('Month')['Amount'].sum()
    
#     plt.figure(figsize=(10, 5))
#     monthly_expenses.plot(kind='line', marker='o')
#     plt.title('Monthly Expense Trends')
#     plt.xlabel('Month')
#     plt.ylabel('Total Expenses ($)')
#     plt.xticks(rotation=45)
#     st.pyplot(plt)
    
#     # Category-wise expenses
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x='Category', y='Amount', data=df, estimator=sum)
#     plt.title('Expenses by Category')
#     plt.xticks(rotation=45)
#     st.pyplot(plt)

# # Streamlit app
# st.title("Smart Expense Analyzer")
# st.write("Upload your bank statement (PDF or CSV) to analyze your expenses.")

# uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'csv'])

# if uploaded_file is not None:
#     # Process file based on type
#     transactions = []
#     if uploaded_file.name.endswith('.pdf'):
#         text = extract_text_from_pdf(uploaded_file)
#         if text:
#             transactions = parse_pdf_transactions(text)
#     elif uploaded_file.name.endswith('.csv'):
#         transactions = read_csv_transactions(uploaded_file)
    
#     if transactions:
#         # Categorize transactions
#         categorized_transactions = categorize_transactions(transactions)
#         df = pd.DataFrame(categorized_transactions)
        
#         # Display transactions
#         st.subheader("Categorized Transactions")
#         st.dataframe(df)
        
#         # Plot expense trends
#         st.subheader("Expense Trends")
#         plot_expense_trends(df)
        
#         # Download categorized transactions
#         csv = df.to_csv(index=False)
#         st.download_button(
#             label="Download Categorized Transactions",
#             data=csv,
#             file_name="categorized_transactions.csv",
#             mime="text/csv"
#         )
#     else:
#         st.error("No valid transactions found in the uploaded file.")
# else:
#     st.info("Please upload a PDF or CSV file to begin.")
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from utils.file_handler import load_file
from utils.eda import generate_eda_report
from utils.llm_agent import summarize_report, ask_question

# Load environment variables (.env) to get Hugging Face token
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Smart Report Analyzer", layout="wide")
st.title("📊 Smart Report Analyzer")
st.write("Upload a CSV, Excel, or PDF file to get insights and summaries using Hugging Face LLMs.")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    with st.spinner("📂 Reading the file..."):
        df, raw_text = load_file(uploaded_file)

    # If it's a structured file (CSV/XLSX)
    if df is not None:
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Exploratory Data Analysis")
        eda_figs = generate_eda_report(df)
        for fig in eda_figs:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 AI-Generated Summary")
        with st.spinner("Generating summary from Hugging Face model..."):
            summary = summarize_report(df)
        st.success(summary)

        st.subheader("💬 Ask a Question About the Data")
        user_question = st.text_input("Type your question:")
        if user_question:
            with st.spinner("Analyzing..."):
                answer = ask_question(df, user_question)
            st.success(answer)

    # If it's a PDF file
    elif raw_text:
        st.subheader("📄 Extracted Text from PDF")
        st.text(raw_text[:1000])  # Preview part of the text

        st.subheader("🧠 AI-Generated Summary")
        with st.spinner("Generating summary from Hugging Face model..."):
            summary = summarize_report(raw_text)
        st.success(summary)

        st.subheader("💬 Ask a Question About the Document")
        user_question = st.text_input("Type your question:")
        if user_question:
            with st.spinner("Analyzing..."):
                answer = ask_question(raw_text, user_question)
            st.success(answer)
