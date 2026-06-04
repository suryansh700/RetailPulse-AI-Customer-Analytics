import pandas as pd
import numpy as np

# Load cleaned sales data
df = pd.read_csv(
    "data/cleaned_sales.csv"
)

print("Data Loaded")

product_sales = (
    df.groupby("Description")
    .agg({
        "Quantity":"sum"
    })
    .reset_index()
)

print(
    product_sales.head()
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

days = (
    df["InvoiceDate"].max()
    -
    df["InvoiceDate"].min()
).days

product_sales["Average_Daily_Sales"] = (
    product_sales["Quantity"]
    /
    days
)

np.random.seed(42)

product_sales["Current_Stock"] = np.random.randint(
    10,
    500,
    size=len(product_sales)
)

product_sales["Reorder_Level"] = (
    product_sales["Average_Daily_Sales"]
    * 7
).round()

product_sales["Need_Reorder"] = (
    product_sales["Current_Stock"]
    <
    product_sales["Reorder_Level"]
)

product_sales["Recommended_Order"] = (
    product_sales["Reorder_Level"]
    -
    product_sales["Current_Stock"]
)

product_sales["Recommended_Order"] = (
    product_sales["Recommended_Order"]
    .clip(lower=0)
)

reorder_products = product_sales[
    product_sales["Need_Reorder"] == True
]

print(
    "\nProducts Requiring Reorder:\n"
)

print(
    reorder_products.head(20)
)

product_sales.to_csv(
    "data/inventory_report.csv",
    index=False
)

print(
    "\nInventory Report Saved"
)

import matplotlib.pyplot as plt

top_inventory = (
    product_sales
    .sort_values(
        by="Quantity",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12,5))

plt.bar(
    top_inventory["Description"],
    top_inventory["Quantity"]
)

plt.xticks(
    rotation=90
)

plt.title(
    "Top Selling Products"
)

plt.show()

reorder_count = (
    product_sales["Need_Reorder"]
    .value_counts()
)

plt.figure(figsize=(6,4))

plt.pie(
    reorder_count,
    labels=["No Reorder","Need Reorder"],
    autopct="%1.1f%%"
)

plt.title(
    "Inventory Status"
)

plt.show()

print("\nInventory Summary")

print(
    "Total Products:",
    len(product_sales)
)

print(
    "Products Needing Reorder:",
    len(reorder_products)
)