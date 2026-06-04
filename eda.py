import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_sales.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Sales Trend
sales = df.groupby("InvoiceDate")["TotalPrice"].sum()

plt.figure(figsize=(12,5))
sales.plot()
plt.title("Sales Trend")
plt.show()

# Heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")
plt.show()

# Top Products
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")
plt.title("Top 10 Products")
plt.show()