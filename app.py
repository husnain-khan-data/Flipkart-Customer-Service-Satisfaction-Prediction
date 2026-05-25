import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("small_rf_model.pkl")

# App Title
st.title("Flipkart Customer Satisfaction Prediction")

st.write("Enter customer details to predict CSAT Score")

# -------------------------------
# Input Fields
# -------------------------------

channel_name = st.selectbox(
    "Channel Name",
    [0, 1, 2],
    format_func=lambda x: {
        0: "Chat",
        1: "Call",
        2: "Email"
    }[x]
)

category = st.selectbox(
    "Category",
    [0, 1, 2],
    format_func=lambda x: {
        0: "Delivery",
        1: "Payment",
        2: "Product Quality"
    }[x]
)

item_price = st.number_input(
    "Item Price",
    min_value=0.0,
    value=1000.0
)

connected_handling_time = st.number_input(
    "Connected Handling Time",
    min_value=0.0,
    value=5.0
)

agent_shift = st.selectbox(
    "Agent Shift",
    [0, 1, 2],
    format_func=lambda x: {
        0: "Morning",
        1: "Evening",
        2: "Night"
    }[x]
)

response_time = st.number_input(
    "Response Time",
    min_value=0.0,
    value=10.0
)

# Log feature
response_time_log = np.log1p(response_time)

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict CSAT Score"):

    input_data = pd.DataFrame([[
        channel_name,
        category,
        item_price,
        connected_handling_time,
        agent_shift,
        response_time,
        response_time_log
    ]], columns=[
        'channel_name',
        'category',
        'Item_price',
        'connected_handling_time',
        'Agent Shift',
        'Response_Time',
        'Response_Time_Log'
    ])

    prediction = model.predict(input_data)

    # -------------------------------
    # Professional Output
    # -------------------------------

    if prediction[0] == 1:
        st.error(f"Predicted CSAT Score: {prediction[0]} - Very Dissatisfied Customer 😡")

    elif prediction[0] == 2:
        st.warning(f"Predicted CSAT Score: {prediction[0]} - Dissatisfied Customer 😕")

    elif prediction[0] == 3:
        st.info(f"Predicted CSAT Score: {prediction[0]} - Neutral Customer 😐")

    elif prediction[0] == 4:
        st.success(f"Predicted CSAT Score: {prediction[0]} - Satisfied Customer 🙂")

    else:
        st.success(f"Predicted CSAT Score: {prediction[0]} - Highly Satisfied Customer 😍")
