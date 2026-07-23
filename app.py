import streamlit as st
import pandas as pd

from utils import (
    load_dataset,
    detect_columns,
    clean_data
)

from charts import (
    revenue_trend,
    monthly_trend,
    category_chart,
    product_chart,
    location_chart,
    payment_chart,
    revenue_profit_chart,
    correlation_chart
)

from forecasting import (
    prepare_forecast_data,
    train_model,
    forecast_future,
    model_summary
)

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Based Sales Forecasting and Business Analytics Dashboard")

st.markdown("""
This dashboard analyzes historical sales data, provides business insights,
and predicts future sales using a Machine Learning model.

Upload a CSV or Excel sales dataset to begin the analysis.
""")

# ----------------------------
# Upload File
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ------------------------
    # Load Dataset
    # ------------------------

    df = load_dataset(uploaded_file)

    detected = detect_columns(df)

    df = clean_data(df, detected)

    revenue_col = detected["revenue"]
    date_col = detected["date"]
    profit_col = detected["profit"]
    category_col = detected["category"]
    product_col = detected["product"]
    location_col = detected["location"]
    payment_col = detected["payment"]

    if revenue_col is None or date_col is None:
        st.error(
            "Revenue or Date column not detected."
        )
        st.stop()

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📋 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.write("Duplicate Rows :", df.duplicated().sum())

    if date_col:
        st.write(
            "Date Range :",
            df[date_col].min(),
            "to",
            df[date_col].max()
        )

    c1, c2 = st.columns(2)

    # ------------------------
    # Sidebar Filters
    # ------------------------

    st.sidebar.title("Filters")
    filtered_df = df.copy()

    if category_col is not None:
        categories = sorted(filtered_df[category_col].dropna().unique())

        selected_categories = st.sidebar.multiselect(
            "Category",
            categories
        )

        if selected_categories:
            filtered_df = filtered_df[
                filtered_df[category_col].isin(selected_categories)
            ]

    if location_col is not None:
        locations = sorted(filtered_df[location_col].dropna().unique())

        selected_locations = st.sidebar.multiselect(
            "Location",
            locations
        )

        if selected_locations:
            filtered_df = filtered_df[
                filtered_df[location_col].isin(selected_locations)
            ]

    # ------------------------
    # KPI Cards
    # ------------------------

    st.divider()

    st.subheader("Business Overview")

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Total Revenue",
            f"{filtered_df[revenue_col].sum():,.2f}"
        )

    with k2:

        if profit_col:
            st.metric(
                "Total Profit",
                f"{filtered_df[profit_col].sum():,.2f}"
            )
        else:
            st.metric(
                "Total Profit",
                "N/A"
            )

    with k3:

        st.metric(
            "Total Records",
            len(filtered_df)
        )

    with k4:

        st.metric(
            "Average Revenue",
            f"{filtered_df[revenue_col].mean():,.2f}"
        )

    # ------------------------
    # Business Insights
    # ------------------------

    st.divider()
    st.subheader("📊 Business Insights")

    col1, col2 = st.columns(2)

    with col1:

        if category_col:

            best_category = (
                filtered_df.groupby(category_col)[revenue_col]
                .sum()
                .idxmax()
            )

            st.success(f"🏆 Best Category : {best_category}")

        if location_col:

            best_location = (
                filtered_df.groupby(location_col)[revenue_col]
                .sum()
                .idxmax()
            )

            st.success(f"📍 Best State : {best_location}")

    with col2:

        if payment_col:

            best_payment = (
                filtered_df.groupby(payment_col)[revenue_col]
                .sum()
                .idxmax()
            )

            st.success(f"💳 Best Payment Mode : {best_payment}")

        st.success(
            f"💰 Average Revenue : ₹ {filtered_df[revenue_col].mean():,.2f}"
        )

    # ------------------------
    # Revenue Analysis
    # ------------------------

    st.divider()
    st.header("📈 Revenue Analysis")

    revenue_trend(
        filtered_df,
        date_col,
        revenue_col
    )

    monthly_trend(
        filtered_df,
        date_col,
        revenue_col
    )

    # ------------------------
    # Category Analysis
    # ------------------------

    if category_col:

        st.subheader("🏆 Revenue by Category")

        category_chart(
            filtered_df,
            category_col,
            revenue_col
        )

    # ------------------------
    # Product Analysis
    # ------------------------

    if product_col:

        st.subheader("📦 Top 10 Products")

        product_chart(
            filtered_df,
            product_col,
            revenue_col
        )

    # ------------------------
    # Location Analysis
    # ------------------------

    if location_col:

        st.subheader("📍 Revenue by Location")

        location_chart(
            filtered_df,
            location_col,
            revenue_col
        )

    # ------------------------
    # Payment Analysis
    # ------------------------

    if payment_col:

        st.subheader("💳 Payment Mode Analysis")

        payment_chart(
            filtered_df,
            payment_col,
            revenue_col
        )

    # ------------------------
    # Revenue vs Profit
    # ------------------------

    if profit_col and category_col:

        st.subheader("💰 Revenue vs Profit")

        revenue_profit_chart(
            filtered_df,
            revenue_col,
            profit_col,
            category_col
        )

    # ------------------------
    # Correlation Heatmap
    # ------------------------

    st.subheader("📊 Correlation Heatmap")

    correlation_chart(filtered_df)

    # ------------------------
    # Sales Forecasting
    # ------------------------

    st.divider()
    st.header("🤖 90-Day Sales Forecast")
    st.info("""
### 🤖 Machine Learning Model Information

**Algorithm Used:** Linear Regression

**Evaluation Metrics:**
- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

**Forecast Period:** Next 90 Days

This model analyzes historical sales data and predicts future sales trends based on past revenue patterns.
""")

    try:

        forecast_df = prepare_forecast_data(
            filtered_df,
            date_col,
            revenue_col
        )

        if len(forecast_df) < 5:

            st.warning(
                "Not enough historical data available for forecasting."
            )

        else:

            model, metrics, processed_df = train_model(
                forecast_df
            )

            summary = model_summary(metrics)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "R² Score",
                    summary["R² Score"]
                )

            with c2:
                st.metric(
                    "MAE",
                    summary["MAE"]
                )

            with c3:
                st.metric(
                    "RMSE",
                    summary["RMSE"]
                )

            future_df = forecast_future(
                model,
                processed_df,
                days=90
            )

            st.subheader("Forecast Table")

            st.dataframe(
                future_df,
                use_container_width=True
            )

            st.subheader("Forecast Chart")

            st.line_chart(
                future_df,
                x="Date",
                y="Forecasted Revenue"
            )

            csv = future_df.to_csv(
                index=False
            )

            st.download_button(
                "⬇ Download Forecast CSV",
                csv,
                "sales_forecast.csv",
                "text/csv"
            )

    except Exception as e:

        st.error(
            f"Forecasting Error : {e}"
        )

    # ------------------------
    # Download Filtered Dataset
    # ------------------------

    st.divider()

    csv_data = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Filtered Dataset",
        csv_data,
        "filtered_dataset.csv",
        "text/csv"
    )

    # ------------------------
    # Dataset Summary
    # ------------------------

    with st.expander("📋 Dataset Summary"):

        st.write("Shape :", filtered_df.shape)

        st.write("Columns")

        st.write(filtered_df.columns.tolist())

        st.write("Statistics")

        st.dataframe(
            filtered_df.describe(
                include="all"
            )
        )

    # ------------------------
    # Footer
    # ------------------------

    st.divider()

    st.markdown("""
<div style="text-align:center;">

# 📊 AI-Based Sales Forecasting and Business Analytics Dashboard

### Developed by

## Aryan Sharma

**Technology Stack**

Python • Streamlit • Pandas • Plotly • Scikit-learn

© 2026 Aryan Sharma. All Rights Reserved.

</div>
""", unsafe_allow_html=True)
