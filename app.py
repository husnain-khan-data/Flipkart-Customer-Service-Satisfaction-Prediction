import streamlit as st
import joblib
import pandas as pd
import math

# Load Model
model = joblib.load('small_rf_model.pkl')

# Title
st.title("Flipkart Customer Satisfaction Prediction")

st.write("Enter customer details to predict CSAT Score")

# -----------------------------
# Channel Name Mapping
# -----------------------------

channel_map = {
    "Chat": 0,
    "Email": 1,
    "Call": 2
}

channel_selected = st.selectbox(
    "Channel Name",
    list(channel_map.keys())
)

channel_name = channel_map[channel_selected]

# -----------------------------
# Category Mapping
# -----------------------------

category_map = {
    "Electronics": 0,
    "Fashion": 1,
    "Grocery": 2,
    "Mobile": 3
}

category_selected = st.selectbox(
    "Category",
    list(category_map.keys())
)

category = category_map[category_selected]

# -----------------------------
# Numeric Inputs
# -----------------------------

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

# -----------------------------
# Agent Shift Mapping
# -----------------------------

shift_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Night": 2
}

shift_selected = st.selectbox(
    "Agent Shift",
    list(shift_map.keys())
)

agent_shift = shift_map[shift_selected]

# -----------------------------
# Response Time
# -----------------------------

response_time = st.number_input(
    "Response Time",
    min_value=1.0,
    value=10.0
)

response_time_log = math.log(response_time)

# -----------------------------
# Prediction
# -----------------------------

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
