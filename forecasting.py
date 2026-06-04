import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sales.csv")

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create TotalPrice if not available
if "TotalPrice" not in df.columns:
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Daily sales
daily_sales = (
    df.groupby("InvoiceDate")["TotalPrice"]
    .sum()
    .reset_index()
)

# Prophet format
daily_sales.columns = ["ds", "y"]

print(daily_sales.head())

# Create model
model = Prophet()

# Train model
model.fit(daily_sales)

print("Model Trained Successfully")

# Future dates
future = model.make_future_dataframe(
    periods=30
)

# Forecast
forecast = model.predict(future)

print(
    forecast[
        ["ds", "yhat"]
    ].tail(30)
)

forecast[
    ["ds", "yhat"]
].to_csv(
    "data/forecast.csv",
    index=False
)

print(
    "Forecast CSV Saved"
)

joblib.dump(
    model,
    "models/forecast_model.pkl"
)

print(
    "Forecast Model Saved"
)

fig = model.plot(forecast)

plt.title(
    "Demand Forecast"
)

plt.show()

fig2 = model.plot_components(
    forecast
)

plt.show()

future_sales = forecast.tail(30)

top_days = future_sales.sort_values(
    by="yhat",
    ascending=False
).head(10)

print("\nTop Future Sales Days")

print(
    top_days[
        ["ds", "yhat"]
    ]
)