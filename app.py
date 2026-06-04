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
income = st.sidebar.number_input("Pendapatan Bulanan (IDR)", min_value=0, value=5000000, step=50000)
expense = st.sidebar.number_input("Total Pengeluaran (IDR)", min_value=0, value=3000000, step=50000)
debt = st.sidebar.number_input("Total Cicilan/Hutang (IDR)", min_value=0, value=500000, step=50000)

# 3. Hitung Feature Engineering (Harus sama dengan di Notebook)
expense_ratio = expense / income if income > 0 else 0
net_cash_flow = income - expense
debt_pressure = debt / income if income > 0 else 0

# Kita bungkus proses prediksi dan insight ke dalam tombol trigger agar berjalan bersamaan
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
        kondisi = res_label[0]
        
        st.subheader("Hasil Analisis AI")
        if kondisi == 'Growth':
            st.success(f"Kondisi Anda: **{kondisi}** (Sangat Sehat) 🚀")
        elif kondisi == 'Stable':
            st.info(f"Kondisi Anda: **{kondisi}** (Cukup Aman) ✅")
        else:
            st.warning(f"Kondisi Anda: **{kondisi}** (Waspada!) ⚠️")
            
        # 5. Tampilkan Insight DINAMIS (Dimasukkan ke dalam blok IF prediksi)
        st.divider()
        st.subheader("Insight Strategis")
        col1, col2, col3 = st.columns(3)
        
        # Metrik 1: Rasio Pengeluaran Dinamis
        col1.metric(
            label="Rasio Pengeluaran", 
            value=f"{expense_ratio * 100:.1f}%", 
            delta="Aman (≤ 70%)" if expense_ratio <= 0.7 else "Terlalu Tinggi (> 70%)",
            delta_color="normal" if expense_ratio <= 0.7 else "inverse"
        )
        
        # Metrik 2: Sisa Uang / Net Cash Flow Dinamis
        col2.metric(
            label="Sisa Uang (Cash Flow)", 
            value=f"IDR {net_cash_flow:,.0f}", 
            delta="Positif" if net_cash_flow > 0 else "Defisit!",
            delta_color="normal" if net_cash_flow > 0 else "inverse"
        )
        
        # Metrik 3: Rekomendasi Alokasi Tabungan berdasarkan prediksi AI
        if kondisi == 'Growth':
            rekomendasi_saving = income * 0.30  # Amankan 30% untuk investasi
            tips = "Disarankan investasi agresif"
        elif kondisi == 'Stable':
            rekomendasi_saving = income * 0.20  # Amankan 20% standard
            tips = "Amankan dana darurat"
        else:
            rekomendasi_saving = income * 0.10  # Pangkas pengeluaran, amankan minimal 10%
            tips = "Kurangi pengeluaran tersier!"
            
        col3.metric(
            label="Target Tabungan Minimal", 
            value=f"IDR {rekomendasi_saving:,.0f}", 
            delta=tips
        )
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
else:
    # Kondisi stand-by saat user baru pertama kali buka web dan belum klik tombol prediksi
    st.info("💡 Silakan isi data keuangan Anda di sidebar kiri, lalu klik **'Prediksi Kondisi'** untuk melihat hasil analisis dan insight.")
