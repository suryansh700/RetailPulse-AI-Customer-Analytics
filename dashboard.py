import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 📊 RetailPulse

### AI Powered Customer Analytics & Demand Forecasting

Predict Demand • Detect Churn • Optimize Inventory
""")

# Sidebar
st.sidebar.title("📊 RetailPulse")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👥 Customer Segmentation",
        "📈 Demand Forecasting",
        "⚠️ Churn Prediction",
        "📦 Inventory Optimization"
    ]
)

# HOME PAGE
if menu == "🏠 Home":

    st.header("Project Overview")

    col1,col2,col3,col4 = st.columns(4)

    try:
        customers = pd.read_csv(
            "data/customer_segments.csv"
        )

        forecast = pd.read_csv(
            "data/forecast.csv"
        )

        churn = pd.read_csv(
            "data/churn_predictions.csv"
        )

        inventory = pd.read_csv(
            "data/inventory_report.csv"
        )

        st.subheader("📈 Business Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="👥 Customers",
                value=f"{len(customers):,}"
            )

        with col2:
            st.metric(
                label="📈 Forecast Days",
                value=f"{len(forecast):,}"
            )

        with col3:
            st.metric(
                label="⚠️ High Risk Customers",
                value=f"{len(churn[churn['Risk_Probability'] > 0.80]):,}"
            )

        with col4:
            st.metric(
                label="📦 Products",
                value=f"{len(inventory):,}"
            )

    except:
        st.warning(
            "Run all modules first"
        )

# CUSTOMER SEGMENTATION
elif menu == "👥 Customer Segmentation":

    st.header("Customer Segmentation")

    df = pd.read_csv(
        "data/customer_segments.csv"
    )

    st.dataframe(df.head())

    fig = px.scatter(
        df,
        x="Frequency",
        y="Monetary",
        color="Segment",
        hover_data=["Recency"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Segments CSV",
        df.to_csv(index=False),
        "customer_segments.csv"
    )

# DEMAND FORECASTING
elif menu == "📈 Demand Forecasting":

    st.header("Demand Forecast")

    forecast = pd.read_csv(
        "data/forecast.csv"
    )

    st.dataframe(
        forecast.tail(30)
    )

    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="30-Day Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Forecast",
        forecast.to_csv(index=False),
        "forecast.csv"
    )

# CHURN
elif menu == "⚠️ Churn Prediction":

    st.header("Customer Churn Prediction")

    churn = pd.read_csv(
        "data/churn_predictions.csv"
    )

    st.dataframe(
        churn.head()
    )

    high_risk = churn[
        churn["Risk_Probability"] > 0.80
    ]

    st.subheader(
        "High Risk Customers"
    )

    st.dataframe(high_risk)

    fig = px.histogram(
        churn,
        x="Risk_Probability",
        nbins=20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# INVENTORY
elif menu == "📦 Inventory Optimization":
    st.header(
        "Inventory Optimization"
    )

    inventory = pd.read_csv(
        "data/inventory_report.csv"
    )

    st.dataframe(
        inventory.head()
    )

    reorder = inventory[
        inventory["Need_Reorder"] == True
    ]

    st.subheader(
        "Products Needing Reorder"
    )

    st.dataframe(reorder)

    fig = px.bar(
        inventory.head(20),
        x="Description",
        y="Quantity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Inventory Report",
        inventory.to_csv(index=False),
        "inventory_report.csv"
    )