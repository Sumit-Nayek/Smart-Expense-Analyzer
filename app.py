# import streamlit as st

# import pandas as pd
# import plotly.express as px
# import streamlit as st
# from utils import extract_text_from_pdf, ask_hf_inference_api, csv_text_to_df

# hf_token = st.secrets["HF_API_TOKEN"]

# st.set_page_config(page_title="Smart Expense Analyzer 💸", layout="wide")
# st.title("📊 Smart Expense Analyzer")
# st.markdown("Upload your PhonePe, Paytm, or Bank **PDF**, and view your **expense insights**.")

# # # Hugging Face token input (for development)
# # hf_token = st.text_input("🔐 Enter your Hugging Face API Token", type="password")

# uploaded_file = st.file_uploader("📁 Upload Transaction PDF", type=["pdf"])

# if uploaded_file and hf_token:
#     with st.spinner("📄 Extracting text..."):
#         text = extract_text_from_pdf(uploaded_file)
#     st.success("✅ Text extracted.")
#     st.text_area("Preview (first 2000 chars)", text[:2000], height=200)

#     if st.button("🚀 Analyze with LLM"):
#         with st.spinner("🤖 Calling Hugging Face Inference API..."):
#             csv_output = ask_hf_inference_api(text, hf_token)

#         if csv_output.startswith("ERROR"):
#             st.error(csv_output)
#         else:
#             st.code(csv_output[:1000], language="csv")
#             df = csv_text_to_df(csv_output)

#             if df is not None:
#                 st.success("✅ Data structured successfully!")

#                 # Clean and convert
#                 df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#                 df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
#                 df.dropna(subset=['Date', 'Amount'], inplace=True)

#                 expenses = df[df['Type'].str.lower() == 'debit']

#                 # Visualization 1: Pie
#                 if 'Category' in df.columns:
#                     st.subheader("📌 Spending by Category")
#                     fig1 = px.pie(expenses, names='Category', values='Amount', hole=0.4)
#                     st.plotly_chart(fig1, use_container_width=True)

#                 # Visualization 2: Trend
#                 st.subheader("📆 Daily Expense Trend")
#                 trend = expenses.groupby('Date')['Amount'].sum().reset_index()
#                 fig2 = px.line(trend, x='Date', y='Amount', markers=True)
#                 st.plotly_chart(fig2, use_container_width=True)

#                 # Visualization 3: Vendors
#                 st.subheader("🏪 Top Descriptions/Vendors")
#                 top = expenses['Description'].value_counts().head(10).reset_index()
#                 top.columns = ['Vendor', 'Count']
#                 fig3 = px.bar(top, x='Vendor', y='Count')
#                 st.plotly_chart(fig3, use_container_width=True)

#                 # Download
#                 st.subheader("📥 Export")
#                 st.download_button("Download CSV", df.to_csv(index=False), "expenses.csv", "text/csv")
#             else:
#                 st.error("⚠️ Failed to parse CSV from LLM output.")
import streamlit as st
import pandas as pd
import PyPDF2
import io
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
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
