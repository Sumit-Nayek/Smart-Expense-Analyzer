"""
utils/llm_agent.py — Powered by NVIDIA NIM (meta/llama-3.1-8b-instruct)
Requires: NVIDIA_API_KEY in Streamlit secrets.
"""
import streamlit as st
import pandas as pd


def _get_client():
    from openai import OpenAI
    api_key = st.secrets.get("NVIDIA_API_KEY", None)
    if not api_key:
        st.error("❌ NVIDIA_API_KEY not found. Add it to your Streamlit secrets.")
        st.stop()
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    client = _get_client()
    try:
        res = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ API error: {e}"


def summarize_report(data) -> str:
    system = (
        "You are a professional data analyst. "
        "Provide a structured summary with key insights, patterns, and recommendations. "
        "Use bullet points. Be concise but thorough."
    )
    if isinstance(data, pd.DataFrame):
        user = f"""Analyze this dataset:

Shape: {data.shape[0]} rows × {data.shape[1]} columns
Columns: {', '.join(data.columns.tolist())}

Data Types:
{data.dtypes.to_string()}

Missing Values:
{data.isnull().sum().to_string()}

Statistics:
{data.describe(include='all').to_string()}

Sample (5 rows):
{data.head(5).to_string(index=False)}

Provide: overview, key patterns, data quality notes, recommendations."""
    else:
        user = f"""Summarize this document:

{str(data)[:3000]}

Provide: main topics, key findings, conclusions, recommendations."""

    return _call(system, user, max_tokens=1024)


def ask_question(data, question: str) -> str:
    system = (
        "You are a helpful data analyst. "
        "Answer questions about the data accurately and concisely. "
        "If the answer cannot be determined from the data, say so clearly."
    )
    if isinstance(data, pd.DataFrame):
        user = f"""Dataset:
Shape: {data.shape[0]} rows × {data.shape[1]} columns
Columns: {', '.join(data.columns.tolist())}
Statistics:
{data.describe(include='all').to_string()}
Sample (10 rows):
{data.head(10).to_string(index=False)}

Question: {question}"""
    else:
        user = f"""Document:
{str(data)[:3000]}

Question: {question}"""

    return _call(system, user, max_tokens=512)
