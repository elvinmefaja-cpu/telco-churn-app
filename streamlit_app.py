import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Telco Customer Churn Predictor")

pipeline = joblib.load("pipeline.pkl")

uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    if "customerID" in raw_df.columns:
        customer_ids = raw_df["customerID"]
        input_df = raw_df.drop(columns=["customerID"])
    else:
        customer_ids = range(len(raw_df))
        input_df = raw_df.copy()

    if "Churn" in input_df.columns:
        input_df = input_df.drop(columns=["Churn"])

    input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

    input_df["tenure_bin"] = pd.cut(
        input_df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["New", "Mid", "Loyal"],
        include_lowest=True
    )

    input_df["is_new_customer"] = (input_df["tenure"] < 6).astype(int)

    input_df["avg_monthly_spend"] = (
        input_df["TotalCharges"] /
        input_df["tenure"].replace(0, np.nan)
    )

    input_df["is_streaming_user"] = (
        (input_df["StreamingTV"] == "Yes") |
        (input_df["StreamingMovies"] == "Yes")
    ).astype(int)

    input_df["security_bundle"] = (
        (input_df["OnlineSecurity"] == "Yes") &
        (input_df["TechSupport"] == "Yes")
    ).astype(int)

    input_df["long_contract"] = (
        input_df["Contract"] != "Month-to-month"
    ).astype(int)

    predictions = pipeline.predict(input_df)
    probabilities = pipeline.predict_proba(input_df)[:, 1]

    results = pd.DataFrame({
        "customerID": customer_ids,
        "Churn_pred": predictions,
        "Churn_prob": probabilities.round(4)
    })

    st.subheader("Predictions")
    st.dataframe(results)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download predictions.csv",
        csv,
        "predictions.csv",
        "text/csv"
    )
