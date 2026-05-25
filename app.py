import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math

# Load model
model = joblib.load('small_rf_model.pkl')

# Title
st.title("Flipkart Customer Satisfaction Prediction")

st.write("Enter customer details to predict CSAT Score")

# Inputs

channel_name = st.selectbox(
    "Channel Name",
    [0, 1, 2]
)

category = st.selectbox(
    "Category",
    [0, 1, 2, 3]
)

item_price = st.number_input(
    "Item Price",
    min_value=0.0
)

connected_handling_time = st.number_input(
    "Connected Handling Time",
    min_value=0.0
)

agent_shift = st.selectbox(
    "Agent Shift",
    [0, 1, 2]
)

response_time = st.number_input(
    "Response Time",
    min_value=1.0
)

response_time_log = math.log(response_time)

# Prediction

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

    st.success(f"Predicted CSAT Score: {prediction[0]}")
