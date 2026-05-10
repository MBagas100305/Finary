import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

def load_data(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    # Jika file ada di dalam folder 'finary'
    elif os.path.exists(os.path.join('finary', file_name)):
        return joblib.load(os.path.join('finary', file_name))
    else:
        raise FileNotFoundError(f"File {file_name} tidak ditemukan di root maupun folder finary")

try:
    model = load_data('model_finary.pkl')
    encoder = load_data('label_encoder.pkl')
    model_features = load_data('model_features.pkl')
except Exception as e:
    st.error(f"Gagal memuat file model: {e}")
    st.stop()
# ---------------------------------------

st.title("📊 FINARY")
st.markdown("Dashboard ini menggunakan **AI (Random Forest)** untuk memprediksi kondisi finansial Anda.")

# 2. Sidebar untuk Input User
st.sidebar.header("Input Data Keuangan")
income = st.sidebar.number_input("Pendapatan Bulanan", min_value=0)
expense = st.sidebar.number_input("Total Pengeluaran", min_value=0)
debt = st.sidebar.number_input("Total Cicilan/Hutang", min_value=0)

# 3. Hitung Feature Engineering (Harus sama dengan di Notebook)
expense_ratio = expense / income if income > 0 else 0
net_cash_flow = income - expense
debt_pressure = debt / income if income > 0 else 0

if st.sidebar.button("Prediksi Kondisi"):
    input_data = np.zeros((1, 98)) 
    input_data[0, 0] = expense_ratio
    input_data[0, 1] = net_cash_flow
    
    prediction = model.predict(input_data)
    
    # Tampilkan Hasil
    st.subheader("Hasil Analisis AI")
    if prediction[0] == 0: # Misal 0 = Growth
        st.success("Kondisi Anda: **GROWTH** (Sangat Sehat)")
    elif prediction[0] == 1: # Misal 1 = Stable
        st.info("Kondisi Anda: **STABLE** (Cukup Aman)")
    else:
        st.warning("Kondisi Anda: **SURVIVAL** (Waspada!)")

# 4. Tampilkan Insight Statis (Bisa dari hasil A/B Testing tadi)
st.divider()
st.subheader("Insight Strategis")
col1, col2 = st.columns(2)
col1.metric("Efektivitas AI", "92%", "+7% Saving")
col2.metric("Rata-rata Penghematan", "IDR 150.000", "per user")
