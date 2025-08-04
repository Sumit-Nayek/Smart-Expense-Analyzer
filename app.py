
# import streamlit as st
# import pandas as pd
# import PyPDF2
# import io
# import re
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans

# # Set page config
# st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

# # Function to extract text from PDF using PyPDF2
# def extract_text_from_pdf(pdf_file):
#     try:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#         return text
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
import streamlit as st
import pandas as pd
import pypdf  # Replaced PyPDF2 with pypdf
import io
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

# Function to extract text from PDF using pypdf
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to parse transactions from PDF text
def parse_pdf_transactions(text):
    transactions = []
    # Simple regex to extract transaction-like patterns (customize based on your PDF format)
    pattern = r'(\d{2}/\d{2}/\d{4})\s+([^\d\n]+)\s+([\d,.]+)'
    matches = re.findall(pattern, text, re.MULTILINE)
    
    for match in matches:
        date, description, amount = match
        try:
            amount = float(amount.replace(',', ''))
            transactions.append({
                'Date': pd.to_datetime(date, format='%d/%m/%Y'),
                'Description': description.strip(),
                'Amount': amount
            })
        except ValueError:
            continue
    return transactions

# Function to read CSV transactions
def read_csv_transactions(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if 'Date' not in df.columns or 'Description' not in df.columns or 'Amount' not in df.columns:
            st.error("CSV must contain 'Date', 'Description', and 'Amount' columns")
            return []
        df['Date'] = pd.to_datetime(df['Date'])
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return []

# Function to categorize transactions using KMeans clustering
def categorize_transactions(transactions):
    if not transactions:
        return []
    
    df = pd.DataFrame(transactions)
    descriptions = df['Description'].values
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Category'] = kmeans.fit_predict(X)
    
    # Map cluster numbers to meaningful categories
    category_map = {
        0: 'Groceries',
        1: 'Utilities',
        2: 'Entertainment',
        3: 'Transportation',
        4: 'Miscellaneous'
    }
    df['Category'] = df['Category'].map(category_map)
    return df.to_dict('records')

# Function to plot expense trends
def plot_expense_trends(df):
    if df.empty:
        st.warning("No data to plot")
        return
    
    # Monthly expenses
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_expenses = df.groupby('Month')['Amount'].sum()
    
    plt.figure(figsize=(10, 5))
    monthly_expenses.plot(kind='line', marker='o')
    plt.title('Monthly Expense Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Expenses ($)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    # Category-wise expenses
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Category', y='Amount', data=df, estimator=sum)
    plt.title('Expenses by Category')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Streamlit app
st.title("Smart Expense Analyzer")
st.write("Upload your bank statement (PDF or CSV) to analyze your expenses.")

uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'csv'])

if uploaded_file is not None:
    # Process file based on type
    transactions = []
    if uploaded_file.name.endswith('.pdf'):
        text = extract_text_from_pdf(uploaded_file)
        if text:
            transactions = parse_pdf_transactions(text)
    elif uploaded_file.name.endswith('.csv'):
        transactions = read_csv_transactions(uploaded_file)
    
    if transactions:
        # Categorize transactions
        categorized_transactions = categorize_transactions(transactions)
        df = pd.DataFrame(categorized_transactions)
        
        # Display transactions
        st.subheader("Categorized Transactions")
        st.dataframe(df)
        
        # Plot expense trends
        st.subheader("Expense Trends")
        plot_expense_trends(df)
        
        # Download categorized transactions
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Categorized Transactions",
            data=csv,
            file_name="categorized_transactions.csv",
            mime="text/csv"
        )
    else:
        st.error("No valid transactions found in the uploaded file.")
else:
    st.info("Please upload a PDF or CSV file to begin.")
