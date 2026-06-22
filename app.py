import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load("telecom_churn_model.pkl")

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Telecom Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

# Input Form
col1, col2 = st.columns(2)

with col1:
    state = st.number_input("State (Encoded)", min_value=0, max_value=50, value=10)
    account_length = st.number_input("Account Length", value=100)
    area_code = st.selectbox("Area Code", [408, 415, 510])

    international_plan = st.selectbox(
        "International Plan",
        [0, 1]
    )

    voice_mail_plan = st.selectbox(
        "Voice Mail Plan",
        [0, 1]
    )

    number_vmail_messages = st.number_input(
        "Number of Voicemail Messages",
        value=0
    )

    total_day_minutes = st.number_input(
        "Total Day Minutes",
        value=180.0
    )

    total_day_calls = st.number_input(
        "Total Day Calls",
        value=100
    )

with col2:

    total_day_charge = st.number_input(
        "Total Day Charge",
        value=30.0
    )

    total_eve_minutes = st.number_input(
        "Total Evening Minutes",
        value=200.0
    )

    total_eve_calls = st.number_input(
        "Total Evening Calls",
        value=100
    )

    total_eve_charge = st.number_input(
        "Total Evening Charge",
        value=17.0
    )

    total_night_minutes = st.number_input(
        "Total Night Minutes",
        value=200.0
    )

    total_night_calls = st.number_input(
        "Total Night Calls",
        value=100
    )

    total_night_charge = st.number_input(
        "Total Night Charge",
        value=9.0
    )

    total_intl_minutes = st.number_input(
        "Total International Minutes",
        value=10.0
    )

    total_intl_calls = st.number_input(
        "Total International Calls",
        value=4
    )

    total_intl_charge = st.number_input(
        "Total International Charge",
        value=2.7
    )

    customer_service_calls = st.number_input(
        "Customer Service Calls",
        value=1
    )

if st.button("Predict Churn"):

    input_data = pd.DataFrame([[
        state,
        account_length,
        area_code,
        international_plan,
        voice_mail_plan,
        number_vmail_messages,
        total_day_minutes,
        total_day_calls,
        total_day_charge,
        total_eve_minutes,
        total_eve_calls,
        total_eve_charge,
        total_night_minutes,
        total_night_calls,
        total_night_charge,
        total_intl_minutes,
        total_intl_calls,
        total_intl_charge,
        customer_service_calls
    ]], columns=[
        'State',
        'Account length',
        'Area code',
        'International plan',
        'Voice mail plan',
        'Number vmail messages',
        'Total day minutes',
        'Total day calls',
        'Total day charge',
        'Total eve minutes',
        'Total eve calls',
        'Total eve charge',
        'Total night minutes',
        'Total night calls',
        'Total night charge',
        'Total intl minutes',
        'Total intl calls',
        'Total intl charge',
        'Customer service calls'
    ])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Likely To Churn")
        st.write(f"Churn Probability: {probability[1]*100:.2f}%")
    else:
        st.success("✅ Customer Likely To Stay")
        st.write(f"Stay Probability: {probability[0]*100:.2f}%")