import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------
# Revenue Trend
# ------------------------------------------------

def revenue_trend(df, date_col, revenue_col):

    temp = df.copy()

    temp[date_col] = pd.to_datetime(temp[date_col])

    temp = (
        temp.groupby(date_col)[revenue_col]
        .sum()
        .reset_index()
    )

    fig = px.line(
        temp,
        x=date_col,
        y=revenue_col,
        title="Revenue Trend",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Monthly Trend
# ------------------------------------------------

def monthly_trend(df, date_col, revenue_col):

    temp = df.copy()

    temp[date_col] = pd.to_datetime(temp[date_col])

    temp["Month"] = temp[date_col].dt.to_period("M").astype(str)

    temp = (
        temp.groupby("Month")[revenue_col]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        temp,
        x="Month",
        y=revenue_col,
        title="Monthly Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Category Analysis
# ------------------------------------------------

def category_chart(df, category_col, revenue_col):

    temp = (
        df.groupby(category_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        temp,
        x=category_col,
        y=revenue_col,
        title="Revenue by Category"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Product Analysis
# ------------------------------------------------

def product_chart(df, product_col, revenue_col):

    temp = (
        df.groupby(product_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        temp,
        x=product_col,
        y=revenue_col,
        title="Top 10 Products"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Location Analysis
# ------------------------------------------------

def location_chart(df, location_col, revenue_col):

    temp = (
        df.groupby(location_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        temp,
        x=location_col,
        y=revenue_col,
        title="Revenue by Location"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Payment Analysis
# ------------------------------------------------

def payment_chart(df, payment_col, revenue_col):

    temp = (
        df.groupby(payment_col)[revenue_col]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        temp,
        names=payment_col,
        values=revenue_col,
        title="Payment Mode Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Revenue vs Profit
# ------------------------------------------------

def revenue_profit_chart(df, revenue_col, profit_col, category_col):

    temp = (
        df.groupby(category_col)[[revenue_col, profit_col]]
        .sum()
        .reset_index()
    )

    fig = px.scatter(
        temp,
        x=revenue_col,
        y=profit_col,
        text=category_col,
        size=revenue_col,
        title="Revenue vs Profit"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Top Customers
# ------------------------------------------------

def customer_chart(df, customer_col, revenue_col):

    temp = (
        df.groupby(customer_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        temp,
        x=customer_col,
        y=revenue_col,
        title="Top 10 Customers"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------

def correlation_chart(df):

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)