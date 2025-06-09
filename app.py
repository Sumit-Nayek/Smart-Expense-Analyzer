import streamlit as st
import pandas as pd
import plotly.express as px
from utils import extract_text_from_pdf, ask_llm_to_structure, csv_text_to_df, load_llm_pipeline

llm_pipe = load_llm_pipeline()
st.set_page_config(page_title="Smart Expense Analyzer 💸", layout="wide")
st.title("📊 Smart Expense Analyzer")
st.markdown("Upload a PhonePe, Paytm, or Bank **PDF transaction history**, and we'll show your **expense patterns**!")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted!")

    st.text_area("📄 Extracted Text Preview", text[:2000], height=200)

    if st.button("🚀 Analyze Expenses with LLM"):
        with st.spinner("💬 Processing with Hugging Face LLM..."):
            csv_output = ask_llm_to_structure(text, llm_pipe)

        st.code(csv_output[:1000], language="csv")  # Preview raw LLM CSV

        df = csv_text_to_df(csv_output)
        if df is not None:
            st.success("✅ Transactions structured successfully!")

            # Data cleaning
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df.dropna(subset=['Date', 'Amount'], inplace=True)

            # Filter for Debit
            expenses = df[df['Type'].str.lower() == 'debit']

            # Visualization 1: Spending by Category
            if 'Category' in df.columns:
                st.subheader("📌 Spending by Category")
                fig1 = px.pie(expenses, names='Category', values='Amount', hole=0.3)
                st.plotly_chart(fig1, use_container_width=True)

            # Visualization 2: Daily Expense Trend
            st.subheader("📆 Daily Expense Trend")
            trend = expenses.groupby('Date')['Amount'].sum().reset_index()
            fig2 = px.line(trend, x='Date', y='Amount', markers=True)
            st.plotly_chart(fig2, use_container_width=True)

            # Visualization 3: Top Vendors
            st.subheader("🏪 Top Vendors/Descriptions")
            top_vendors = expenses['Description'].value_counts().head(10).reset_index()
            top_vendors.columns = ['Vendor', 'Count']
            fig3 = px.bar(top_vendors, x='Vendor', y='Count')
            st.plotly_chart(fig3, use_container_width=True)

            # Download option
            st.subheader("📥 Download Structured Data")
            st.download_button("Download as CSV", df.to_csv(index=False), file_name="expenses.csv", mime="text/csv")
        else:
            st.error("❌ Failed to parse CSV. Check LLM output format.")
