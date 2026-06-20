"""
llm_agent.py — Powered by NVIDIA NIM (meta/llama-3.1-8b-instruct)
No Hugging Face dependencies, no version conflicts.
Requires: pip install openai
Add to Streamlit secrets: NVIDIA_API_KEY = "nvapi-xxxx"
"""

import streamlit as st
from openai import OpenAI
import pandas as pd


# ── NIM client ────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = st.secrets.get("NVIDIA_API_KEY", None)
    if not api_key:
        st.error("NVIDIA_API_KEY not found. Add it to your Streamlit secrets.")
        st.stop()
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def _call_nim(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """Single helper — calls NIM and returns clean text."""
    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",   # free, clean output, low credit cost
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ API error: {str(e)}"


# ── Public functions (called from app.py) ─────────────────────────────────────

def summarize_report(data) -> str:
    """
    Accepts either a DataFrame (CSV/XLSX) or a plain string (PDF text).
    Returns an AI-generated summary.
    """
    system = (
        "You are a professional data analyst. "
        "Provide a clear, structured summary with key insights, patterns, and recommendations. "
        "Use bullet points where helpful. Be concise but thorough."
    )

    if isinstance(data, pd.DataFrame):
        # Build a compact text snapshot of the dataframe
        shape_info   = f"Rows: {data.shape[0]}, Columns: {data.shape[1]}"
        col_info     = f"Columns: {', '.join(data.columns.tolist())}"
        dtype_info   = data.dtypes.to_string()
        null_info    = data.isnull().sum().to_string()
        stats        = data.describe(include="all").to_string()
        sample       = data.head(5).to_string(index=False)

        user = f"""Analyze this dataset and provide a comprehensive summary.

Dataset Overview:
{shape_info}
{col_info}

Column Data Types:
{dtype_info}

Missing Values:
{null_info}

Statistical Summary:
{stats}

Sample Data (first 5 rows):
{sample}

Please provide:
1. Dataset overview
2. Key patterns and trends
3. Notable statistics
4. Data quality observations
5. Actionable recommendations"""

    else:
        # PDF / raw text
        preview = str(data)[:3000]   # keep within token budget
        user = f"""Summarize the following document content:

{preview}

Please provide:
1. Main topics covered
2. Key points and findings
3. Important conclusions
4. Recommendations (if any)"""

    return _call_nim(system, user, max_tokens=1024)


def ask_question(data, question: str) -> str:
    """
    Answers a natural-language question about the data.
    Accepts either a DataFrame or a plain string.
    """
    system = (
        "You are a helpful data analyst assistant. "
        "Answer questions about the provided data accurately and concisely. "
        "If you cannot determine the answer from the data, say so clearly."
    )

    if isinstance(data, pd.DataFrame):
        shape_info = f"Rows: {data.shape[0]}, Columns: {data.shape[1]}"
        col_info   = f"Columns: {', '.join(data.columns.tolist())}"
        stats      = data.describe(include="all").to_string()
        sample     = data.head(10).to_string(index=False)

        user = f"""Dataset Information:
{shape_info}
{col_info}

Statistical Summary:
{stats}

Sample Data (first 10 rows):
{sample}

Question: {question}

Please answer based on the data above. Be specific and precise."""

    else:
        preview = str(data)[:3000]
        user = f"""Document Content:
{preview}

Question: {question}

Please answer based on the document content above."""

    return _call_nim(system, user, max_tokens=512)
