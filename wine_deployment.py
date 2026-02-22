# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("dtc_model.pkl")

st.title("Wine Type Prediction")

# Show model classes (for debugging)
st.write("Model classes:", model.classes_)

# Realistic default values
fixed_acidity = st.number_input("fixed acidity", value=7.0)
volatile_acidity = st.number_input("volatile acidity", value=0.5)
citric_acid = st.number_input("citric acid", value=0.3)
residual_sugar = st.number_input("residual sugar", value=2.0)
chlorides = st.number_input("chlorides", value=0.07)
free_sulfur_dioxide = st.number_input("free sulfur dioxide", value=15.0)
total_sulfur_dioxide = st.number_input("total sulfur dioxide", value=40.0)
density = st.number_input("density", value=0.995)
pH = st.number_input("pH", value=3.3)
sulphates = st.number_input("sulphates", value=0.6)
alcohol = st.number_input("alcohol", value=10.0)
quality = st.number_input("quality", value=6)

# Create DataFrame
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
    "quality": [quality]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    st.write("Raw prediction:", prediction)

    # Automatically handle numeric or string labels
    if isinstance(prediction, str):
        if prediction.lower() == "white":
            st.success("White Wine 🍷")
        else:
            st.error("Red Wine 🍷")
    else:
        # If numeric, use model.classes_ to decode properly
        predicted_label = model.classes_[prediction]
        st.write("Decoded label:", predicted_label)

        if str(predicted_label).lower() == "white":
            st.success("White Wine 🍷")
        else:
            st.error("Red Wine 🍷")
