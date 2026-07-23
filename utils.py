import pandas as pd

# -----------------------------
# Possible column names
# -----------------------------

REVENUE_COLUMNS = [
    "sales",
    "revenue",
    "amount",
    "total",
    "total amount",
    "order amount",
    "transaction amount",
    "price",
    "income",
    "net sales"
]

DATE_COLUMNS = [
    "date",
    "order date",
    "invoice date",
    "transaction date",
    "purchase date",
    "bill date"
]

PROFIT_COLUMNS = [
    "profit",
    "net profit",
    "margin",
    "earning"
]

CATEGORY_COLUMNS = [
    "category",
    "product category",
    "segment",
    "department"
]

PRODUCT_COLUMNS = [
    "product",
    "product name",
    "item",
    "item name"
]

LOCATION_COLUMNS = [
    "state",
    "city",
    "location",
    "region",
    "country"
]

PAYMENT_COLUMNS = [
    "payment",
    "payment mode",
    "payment method",
    "mode"
]


# -----------------------------
# Load Dataset
# -----------------------------

def load_dataset(file):

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    else:
        raise ValueError("Unsupported File Format")


# -----------------------------
# Find Matching Column
# -----------------------------

def find_column(columns, keywords):

    lower_cols = {col.lower().strip(): col for col in columns}

    # Exact Match
    for key in keywords:
        if key.lower() in lower_cols:
            return lower_cols[key.lower()]

    # Partial Match
    for col in columns:
        name = col.lower().strip()

        for key in keywords:
            if key.lower() in name:
                return col

    return None


# -----------------------------
# Detect Important Columns
# -----------------------------

def detect_columns(df):

    cols = df.columns.tolist()

    return {

        "date": find_column(cols, DATE_COLUMNS),

        "revenue": find_column(cols, REVENUE_COLUMNS),

        "profit": find_column(cols, PROFIT_COLUMNS),

        "category": find_column(cols, CATEGORY_COLUMNS),

        "product": find_column(cols, PRODUCT_COLUMNS),

        "location": find_column(cols, LOCATION_COLUMNS),

        "payment": find_column(cols, PAYMENT_COLUMNS)

    }


# -----------------------------
# Clean Dataset
# -----------------------------

def clean_data(df, detected):

    df = df.copy()

    date_col = detected["date"]
    revenue_col = detected["revenue"]
    profit_col = detected["profit"]

    if date_col:

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    if revenue_col:

        df[revenue_col] = pd.to_numeric(
            df[revenue_col],
            errors="coerce"
        )

    if profit_col:

        df[profit_col] = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        )

    df = df.dropna()

    return df