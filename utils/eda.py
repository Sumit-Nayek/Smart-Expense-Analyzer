"""
utils/eda.py — Generates Plotly charts for a DataFrame.
"""
import pandas as pd
import plotly.express as px


def generate_eda_report(df: pd.DataFrame) -> list:
    figs = []

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Detect date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except Exception:
                pass

    # 1. Summary stats table
    if num_cols:
        stats = df[num_cols].describe().reset_index()
        fig = px.imshow(
            df[num_cols].describe().round(2),
            text_auto=True,
            title="📊 Statistical Summary",
            color_continuous_scale="Blues"
        )
        figs.append(fig)

    # 2. Histogram for each numeric column
    for col in num_cols[:4]:  # limit to 4
        fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                           color_discrete_sequence=["#636EFA"])
        figs.append(fig)

    # 3. Bar chart for top categorical columns
    for col in cat_cols[:2]:
        top = df[col].value_counts().head(10).reset_index()
        top.columns = [col, "count"]
        fig = px.bar(top, x=col, y="count",
                     title=f"Top values in {col}",
                     color_discrete_sequence=["#EF553B"])
        figs.append(fig)

    # 4. Line chart if date + numeric exists
    if date_col and num_cols:
        fig = px.line(df.sort_values(date_col), x=date_col, y=num_cols[0],
                      title=f"{num_cols[0]} over Time")
        figs.append(fig)

    # 5. Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True,
                        title="🔗 Correlation Heatmap",
                        color_continuous_scale="RdBu_r")
        figs.append(fig)

    if not figs:
        fig = px.bar(title="No numeric data found to visualize.")
        figs.append(fig)

    return figs
