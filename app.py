import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Konfigurasi Halaman & Tema Minimalis
st.set_page_config(
    page_title="FINARY | Predictive Financial Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk merapikan font dan UI agar terlihat konsisten
st.markdown("""
    <style>
    .reportview-container { background: #fdfdfd; }
    .metric-label { font-size: 14px !important; color: #555555 !important; font-weight: 500; }
    .status-badge { padding: 6px 14px; border-radius: 4px; font-size: 14px; font-weight: 600; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

def load_data(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    elif os.path.exists(os.path.join('finary', file_name)):
        return joblib.load(os.path.join('finary', file_name))
    else:
        raise FileNotFoundError(f"File {file_name} tidak ditemukan pada direktori sistem.")

try:
    model = load_data('model_finary.pkl')
    encoder = load_data('label_encoder.pkl')
    model_features = load_data('model_features.pkl')
except Exception as e:
    st.error(f"Sistem gagal menginisialisasi modul analitik: {e}")
    st.stop()

# ------------------ SIDEBAR CONTROL PANEL ------------------
st.sidebar.markdown("### **Panel Kendali Data**")
st.sidebar.markdown("Silakan masukkan parameter keuangan bulanan entitas secara akurat.")

with st.sidebar.form(key="financial_form"):
    income = st.number_input("Pendapatan Bulanan (IDR)", min_value=0, value=0, step=100000)
    expense = st.number_input("Total Pengeluaran (IDR)", min_value=0, value=0, step=100000)
    debt = st.number_input("Total Komitmen Cicilan / Hutang (IDR)", min_value=0, value=0, step=50000)
    
    submit_button = st.form_submit_button(label="Jalankan Prediksi Sistem", use_container_width=True)

# ------------------ MAIN DASHBOARD HEADER ------------------
st.title("FINARY Analytics")
st.caption("Platform Analitik Prediktif Kesehatan Finansial Berbasis Komputasi Random Forest")
st.markdown("<hr style='margin-top:0; margin-bottom:25px;'>", unsafe_allow_html=True)

# Feature Engineering
expense_ratio = expense / income if income > 0 else 0
net_cash_flow = income - expense
debt_pressure = debt / income if income > 0 else 0

if submit_button:
    if income == 0:
        st.warning("Eror Validasi: Nilai pendapatan bulanan tidak boleh kosong untuk melakukan kalkulasi rasio.")
        st.stop()

    # Konstruksi Dataframe untuk Input Model
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    if 'expense_ratio' in input_df.columns:
        input_df['expense_ratio'] = expense_ratio
    if 'net_cash_flow' in input_df.columns:
        input_df['net_cash_flow'] = net_cash_flow
    if 'debt_pressure' in input_df.columns:
        input_df['debt_pressure'] = debt_pressure

    try:
        # Eksekusi Model
        input_df = input_df[model_features] 
        prediction = model.predict(input_df)
        res_label = encoder.inverse_transform(prediction)
        kondisi = res_label[0]
        
        # Segmentasi Layout Utama
        tab1, tab2 = st.tabs(["Ringkasan Eksekutif", "Metrik & Struktur Data"])
        
        with tab1:
            st.markdown("### Kesimpulan Kondisi Finansial")
            
            with st.container(border=True):
                col_status, col_desc = st.columns([1, 2])
                
                with col_status:
                    # Desain Badge Status Premium & Clean (Tanpa emoji, fokus pada warna solid perusahaan)
                    if kondisi == 'Growth':
                        st.html("<h1 style='color:#1b5e20; margin:0; font-weight:800; font-size:32px;'>GROWTH</h1>")
                        st.html("<span class='status-badge' style='background-color:#e8f5e9; color:#1b5e20;'>Klasifikasi: Sangat Sehat</span>")
                    elif kondisi == 'Stable':
                        st.html("<h1 style='color:#0d47a1; margin:0; font-weight:800; font-size:32px;'>STABLE</h1>")
                        st.html("<span class='status-badge' style='background-color:#e3f2fd; color:#0d47a1;'>Klasifikasi: Optimal</span>")
                    else:
                        st.html("<h1 style='color:#b71c1c; margin:0; font-weight:800; font-size:32px;'>WARNING</h1>")
                        st.html("<span class='status-badge' style='background-color:#ffebee; color:#b71c1c;'>Klasifikasi: Risiko Tinggi</span>")
                
                with col_desc:
                    st.markdown("**Hasil Penilaian Komputasi AI:**")
                    if kondisi == 'Growth':
                        st.markdown("Struktur keuangan menunjukkan ekspansi positif dengan kapasitas retensi modal yang tinggi. Direkomendasikan untuk meningkatkan alokasi pada instrumen pertumbuhan investasi jangka panjang.")
                    elif kondisi == 'Stable':
                        st.markdown("Kondisi neraca keuangan berada pada posisi ekuilibrium yang aman. Fokus manajemen saat ini diarahkan pada pemenuhan likuiditas dana darurat sebelum melakukan ekspansi aset.")
                    else:
                        st.markdown("Indikator mendeteksi adanya tekanan pada arus kas akibat ketidakseimbangan rasio pengeluaran atau beban leverage. Diperlukan tindakan korektif berupa restrukturisasi anggaran sesegera mungkin.")

            st.markdown("### Indikator Utama")
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                
                m1.metric(
                    label="Rasio Pengeluaran Bulanan", 
                    value=f"{expense_ratio * 100:.2f}%", 
                    delta="Di bawah ambang batas (≤ 70%)" if expense_ratio <= 0.7 else "Melebihi ambang batas (> 70%)",
                    delta_color="normal" if expense_ratio <= 0.7 else "inverse"
                )
                
                m2.metric(
                    label="Arus Kas Bersih (Net Cash Flow)", 
                    value=f"IDR {net_cash_flow:,.0f}", 
                    delta="Surplus" if net_cash_flow > 0 else "Defisit",
                    delta_color="normal" if net_cash_flow > 0 else "inverse"
                )
                
                # Formula kalkulasi target dana simpanan minimum
                if kondisi == 'Growth':
                    rekomendasi_saving = income * 0.30
                    tips = "Alokasikan ke portofolio produktif"
                elif kondisi == 'Stable':
                    rekomendasi_saving = income * 0.20
                    tips = "Prioritaskan instrumen likuiditas tinggi"
                else:
                    rekomendasi_saving = income * 0.10
                    tips = "Retensi modal minimal untuk mitigasi risiko"
                    
                m3.metric(
                    label="Proyeksi Minimum Tabungan", 
                    value=f"IDR {rekomendasi_saving:,.0f}", 
                    delta=tips
                )

        with tab2:
            st.markdown("### Parameter Komposisi Keuangan")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.text_input("Total Pendapatan Terdaftar", value=f"IDR {income:,.0f}", disabled=True)
                    st.text_input("Total Pengeluaran Operasional", value=f"IDR {expense:,.0f}", disabled=True)
                with c2:
                    st.text_input("Total Kewajiban / Hutang", value=f"IDR {debt:,.0f}", disabled=True)
                    st.text_input("Rasio Tekanan Leverage (Debt Ratio)", value=f"{debt_pressure * 100:.2f}%", disabled=True)

    except Exception as e:
        st.error(f"Kegagalan sistem pada pemrosesan komputasi internal: {e}")

else:
    # State Awal Aplikasi (Clean minimalis placeholder)
    with st.container(border=True):
        st.markdown("##### **Sistem Analitik Keuangan FINARY**")
        st.markdown("Status: *Menunggu Input Data*")
        st.caption("Silakan masukkan variabel data keuangan Anda melalui panel kendali di sebelah kiri, kemudian tekan tombol 'Jalankan Prediksi Sistem' untuk memulai analisis komparatif AI.")
