import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_sales.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create TotalPrice if missing
if "TotalPrice" not in df.columns:
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Snapshot date
snapshot_date = df["InvoiceDate"].max()

# RFM Calculation
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nRFM Table:")
print(rfm.head())

# Scaling
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm)

# KMeans
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nCluster Counts:")
print(rfm["Cluster"].value_counts())

# Save model
joblib.dump(
    kmeans,
    "models/segmentation_model.pkl"
)

# Save RFM data
rfm.to_csv(
    "data/customer_segments.csv"
)

print("\nModel Saved Successfully")
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="Set2"
)

plt.title(
    "Customer Segments"
)

plt.show()
cluster_summary = rfm.groupby(
    "Cluster"
).mean()

print("\nCluster Summary:")
print(cluster_summary)
segment_names = {
    0: "Regular",
    1: "At Risk",
    2: "VIP",
    3: "Lost"
}

rfm["Segment"] = rfm["Cluster"].map(
    segment_names
)

rfm.to_csv(
    "data/customer_segments.csv"
)

print(
    rfm[["Segment"]].head()
)