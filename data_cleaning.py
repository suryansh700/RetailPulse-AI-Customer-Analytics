import pandas as pd

# Load dataset
df = pd.read_csv("data/sales.csv", encoding="latin1")

print("Original Shape:")
print(df.shape)

# Remove missing values
df.dropna(inplace=True)

# Remove negative quantity
df = df[df["Quantity"] > 0]

# Remove negative price
df = df[df["UnitPrice"] > 0]

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\nCleaned Shape:")
print(df.shape)

# Save cleaned data
df.to_csv("data/cleaned_sales.csv", index=False)

print("\nCleaned file saved.")