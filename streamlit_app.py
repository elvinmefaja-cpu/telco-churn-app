import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Telco Churn App", page_icon="📊")

st.title("Telco Customer Churn App")
st.write("Upload a CSV file to preview customer data.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Column Names")
    st.write(list(df.columns))

    if "Churn" in df.columns:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    if "MonthlyCharges" in df.columns:
        st.subheader("Monthly Charges")
        st.line_chart(df["MonthlyCharges"])

else:
    st.info("Please upload a CSV file.")
