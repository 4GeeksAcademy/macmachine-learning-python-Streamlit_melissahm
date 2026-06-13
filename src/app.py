import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("data/processed/random_forest_regressor.sav", "rb"))
scaler = pickle.load(open("data/processed/scaler.sav", "rb"))
onehot_encoder = pickle.load(open("data/processed/onehot_encoder.sav", "rb"))

num_variables = list(scaler.feature_names_in_)
cat_variables = list(onehot_encoder.feature_names_in_)

st.title("Predicción de precio de coches")
st.write("Aplicación web con Streamlit para estimar el precio de un coche usado.")

company = st.selectbox(
    "Marca del coche",
    ["Hyundai", "Renault", "Chevrolet", "Mercedes", "Maruti", "Ford", "Toyota", "Honda"]
)

year = st.number_input(
    "Año del coche",
    min_value=1990,
    max_value=2026,
    value=2015
)

kms_driven = st.number_input(
    "Kilómetros recorridos",
    min_value=0,
    max_value=500000,
    value=50000
)

fuel_type = st.selectbox(
    "Tipo de combustible",
    ["Petrol", "Diesel", "LPG"]
)

if st.button("Predecir precio"):
    input_data = pd.DataFrame([{
        "company": company,
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type
    }])

    input_num = scaler.transform(input_data[num_variables])
    input_num = pd.DataFrame(input_num, columns=num_variables)

    input_cat = onehot_encoder.transform(input_data[cat_variables])
    input_cat = pd.DataFrame(
        input_cat,
        columns=onehot_encoder.get_feature_names_out(cat_variables)
    )

    input_final = pd.concat([input_num, input_cat], axis=1)

    input_final = input_final.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    pred_log = model.predict(input_final)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"Precio estimado del coche: {pred_price:,.0f}")