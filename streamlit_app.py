import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn App", page_icon="📊", layout="wide")

st.title("Telco Customer Churn Analysis")
st.write("This Streamlit app analyzes Telco customer data and shows churn insights.")

uploaded_file = st.file_uploader("Upload Telco CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head())

    st.subheader("2. Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("3. Missing Values")
    st.dataframe(df.isnull().sum())

    if "Churn" in df.columns:
        st.subheader("4. Churn Distribution")
        churn_counts = df["Churn"].value_counts()
        st.bar_chart(churn_counts)

    if "Contract" in df.columns and "Churn" in df.columns:
        st.subheader("5. Churn by Contract Type")
        contract_churn = pd.crosstab(df["Contract"], df["Churn"])
        st.bar_chart(contract_churn)

    if "InternetService" in df.columns and "Churn" in df.columns:
        st.subheader("6. Churn by Internet Service")
        internet_churn = pd.crosstab(df["InternetService"], df["Churn"])
        st.bar_chart(internet_churn)

    if "MonthlyCharges" in df.columns:
        st.subheader("7. Monthly Charges")
        fig, ax = plt.subplots()
        ax.hist(df["MonthlyCharges"], bins=30)
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    if "tenure" in df.columns:
        st.subheader("8. Tenure Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["tenure"], bins=30)
        ax.set_xlabel("Tenure")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    if "TotalCharges" in df.columns:
        st.subheader("9. Total Charges")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        fig, ax = plt.subplots()
        ax.hist(df["TotalCharges"].dropna(), bins=30)
        ax.set_xlabel("Total Charges")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    st.subheader("10. Summary Statistics")
    st.dataframe(df.describe())

else:
    st.info("Please upload your Telco CSV file.")
