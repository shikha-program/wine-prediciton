# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("dtc_model.pkl")

st.title("Wine Type Prediction")

fixed_acidity = st.number_input("fixed acidity")
volatile_acidity = st.number_input("volatile acidity")
citric_acid = st.number_input("citric acid")
residual_sugar = st.number_input("residual sugar")
chlorides = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free sulfur dioxide")
total_sulfur_dioxide = st.number_input("total sulfur dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates = st.number_input("sulphates")
alcohol = st.number_input("alcohol")
quality = st.number_input("quality")   # include only if used during training

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

    if prediction == 1:
        st.success("White Wine 🍷")
    else:
        st.error("Red Wine 🍷")
