import streamlit as st
from utils import extract_text_from_pdf, ask_hf_inference_api, csv_text_to_df
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import extract_text_from_pdf, ask_hf_inference_api, csv_text_to_df

hf_token = st.secrets["HF_API_TOKEN"]

st.set_page_config(page_title="Smart Expense Analyzer 💸", layout="wide")
st.title("📊 Smart Expense Analyzer")
st.markdown("Upload your PhonePe, Paytm, or Bank **PDF**, and view your **expense insights**.")

# # Hugging Face token input (for development)
# hf_token = st.text_input("🔐 Enter your Hugging Face API Token", type="password")

uploaded_file = st.file_uploader("📁 Upload Transaction PDF", type=["pdf"])

if uploaded_file and hf_token:
    with st.spinner("📄 Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Text extracted.")
    st.text_area("Preview (first 2000 chars)", text[:2000], height=200)

    if st.button("🚀 Analyze with LLM"):
        with st.spinner("🤖 Calling Hugging Face Inference API..."):
            csv_output = ask_hf_inference_api(text, hf_token)

        if csv_output.startswith("ERROR"):
            st.error(csv_output)
        else:
            st.code(csv_output[:1000], language="csv")
            df = csv_text_to_df(csv_output)

            if df is not None:
                st.success("✅ Data structured successfully!")

                # Clean and convert
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                df.dropna(subset=['Date', 'Amount'], inplace=True)

                expenses = df[df['Type'].str.lower() == 'debit']

                # Visualization 1: Pie
                if 'Category' in df.columns:
                    st.subheader("📌 Spending by Category")
                    fig1 = px.pie(expenses, names='Category', values='Amount', hole=0.4)
                    st.plotly_chart(fig1, use_container_width=True)

                # Visualization 2: Trend
                st.subheader("📆 Daily Expense Trend")
                trend = expenses.groupby('Date')['Amount'].sum().reset_index()
                fig2 = px.line(trend, x='Date', y='Amount', markers=True)
                st.plotly_chart(fig2, use_container_width=True)

                # Visualization 3: Vendors
                st.subheader("🏪 Top Descriptions/Vendors")
                top = expenses['Description'].value_counts().head(10).reset_index()
                top.columns = ['Vendor', 'Count']
                fig3 = px.bar(top, x='Vendor', y='Count')
                st.plotly_chart(fig3, use_container_width=True)

                # Download
                st.subheader("📥 Export")
                st.download_button("Download CSV", df.to_csv(index=False), "expenses.csv", "text/csv")
            else:
                st.error("⚠️ Failed to parse CSV from LLM output.")
