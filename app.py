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
    # 1. Buat DataFrame dengan satu baris berisi nol, kolom sesuai model_features
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    
    # 2. Masukkan input user ke kolom yang tepat
    if 'expense_ratio' in input_df.columns:
        input_df['expense_ratio'] = expense_ratio
    if 'net_cash_flow' in input_df.columns:
        input_df['net_cash_flow'] = net_cash_flow
    if 'debt_pressure' in input_df.columns:
        input_df['debt_pressure'] = debt_pressure
    
    # 3. Prediksi (Urutan kolom otomatis mengikuti model_features)
    try:
        input_df = input_df[model_features] 
        
        prediction = model.predict(input_df)
        
        # 4. Transformasi hasil angka ke label teks
        res_label = encoder.inverse_transform(prediction)
        
        st.subheader("Hasil Analisis AI")
        if res_label[0] == 'Growth':
            st.success(f"Kondisi Anda: **{res_label[0]}** (Sangat Sehat) 🚀")
        elif res_label[0] == 'Stable':
            st.info(f"Kondisi Anda: **{res_label[0]}** (Cukup Aman) ✅")
        else:
            st.warning(f"Kondisi Anda: **{res_label[0]}** (Waspada!) ⚠️")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# 5. Tampilkan Insight Statis
st.divider()
st.subheader("Insight Strategis")
col1, col2 = st.columns(2)
col1.metric("Efektivitas AI", "92%", "+7% Saving")
col2.metric("Rata-rata Penghematan", "IDR 150.000", "per user")
