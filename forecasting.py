import math
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ---------------------------------------
# Prepare Data
# ---------------------------------------

def prepare_forecast_data(df, date_col, revenue_col):

    temp = df.copy()

    temp[date_col] = pd.to_datetime(
        temp[date_col],
        errors="coerce"
    )

    temp = temp.dropna(subset=[date_col, revenue_col])

    temp[revenue_col] = pd.to_numeric(
        temp[revenue_col],
        errors="coerce"
    )

    temp = temp.dropna(subset=[revenue_col])

    temp = (
        temp.groupby(date_col)[revenue_col]
        .sum()
        .reset_index()
        .sort_values(date_col)
    )

    temp["Day"] = (
        temp[date_col] - temp[date_col].min()
    ).dt.days

    return temp


# ---------------------------------------
# Train Model
# ---------------------------------------

def train_model(df):

    X = df[["Day"]]
    y = df.iloc[:, 1]

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict(X)

    metrics = {

    "R2": r2_score(y, prediction),

    "MAE": mean_absolute_error(y, prediction),

    "RMSE": math.sqrt(
        mean_squared_error(
            y,
            prediction
        )
    )

}  

    return model, metrics, df


# ---------------------------------------
# Forecast Future
# ---------------------------------------

def forecast_future(
    model,
    processed_df,
    days=90
):

    last_day = processed_df["Day"].max()

    future_days = pd.DataFrame({

        "Day": range(
            last_day + 1,
            last_day + days + 1
        )

    })

    future_days["Forecasted Revenue"] = model.predict(
        future_days
    )

    first_date = processed_df.iloc[0, 0]

    future_days["Date"] = (
        first_date
        + pd.to_timedelta(
            future_days["Day"],
            unit="D"
        )
    )

    future_days = future_days[
        [
            "Date",
            "Forecasted Revenue"
        ]
    ]

    return future_days


# ---------------------------------------
# Model Summary
# ---------------------------------------

def model_summary(metrics):

    return {
        "R² Score": round(metrics["R2"], 3),
        "MAE": round(metrics["MAE"], 2),
        "RMSE": round(metrics["RMSE"], 2)
    }