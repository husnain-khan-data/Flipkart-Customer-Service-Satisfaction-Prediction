import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("small_rf_model.pkl")

# Title
st.title("Flipkart Customer Satisfaction Prediction")

st.write("Enter customer details to predict CSAT Score")

# Inputs
agent_experience = st.number_input("Agent Experience Level", min_value=0)

response_time = st.number_input("Response Time", min_value=0)

issue_resolved = st.selectbox("Issue Resolved", [0, 1])

customer_rating = st.slider("Customer Rating", 1, 5)

# Dummy extra features
feature5 = 0
feature6 = 0
feature7 = 0

# Prediction
if st.button("Predict CSAT Score"):

    input_data = np.array([[agent_experience,
                            response_time,
                            issue_resolved,
                            customer_rating,
                            feature5,
                            feature6,
                            feature7]])

    prediction = model.predict(input_data)

    st.success(f"Predicted CSAT Score: {prediction[0]}")
