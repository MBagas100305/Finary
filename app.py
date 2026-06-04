import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Konfigurasi Halaman Modern (Harus di bagian paling atas)
st.set_page_config(
    page_title="FINARY - AI Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    elif os.path.exists(os.path.join('finary', file_name)):
        return joblib.load(os.path.join('finary', file_name))
    else:
        raise FileNotFoundError(f"File {file_name} tidak ditemukan di root maupun folder finary")

try:
    model = load_data('model_finary.pkl')
    encoder = load_data('label_encoder.pkl')
    model_features = load_data('model_features.pkl')
except Exception as e:
    st.error(f"⚠️ Gagal memuat komponen AI: {e}")
    st.stop()

# ------------------ SIDEBAR & INPUT ------------------
st.sidebar.header("🎯 Navigasi & Input")
st.sidebar.markdown("Masukkan detail kondisi finansial bulanan Anda di bawah ini:")

with st.sidebar.form(key="financial_form"):
    income = st.number_input("💵 Pendapatan Bulanan (IDR)", min_value=0, value=0, step=100000)
    expense = st.number_input("📉 Total Pengeluaran (IDR)", min_value=0, value=0, step=100000)
    debt = st.number_input("💳 Total Cicilan/Hutang (IDR)", min_value=0, value=0, step=50000)
    
    submit_button = st.form_submit_button(label="⚡ Analisis Finansial", use_container_width=True)

# ------------------ MAIN HERO SECTION ------------------
st.title("📊 FINARY")
st.caption("Advanced AI Financial Health Management Analytics powered by Random Forest")
st.markdown("---")

# Feature Engineering
expense_ratio = expense / income if income > 0 else 0
net_cash_flow = income - expense
debt_pressure = debt / income if income > 0 else 0

if submit_button:
    # Validasi input kosong agar UI tidak rusak atau menampilkan angka nol semua
    if income == 0:
        st.warning("⚠️ Mohon isi 'Pendapatan Bulanan' Anda terlebih dahulu pada sidebar.")
        st.stop()

    # 1. Menyiapkan Dataframe untuk Model
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    if 'expense_ratio' in input_df.columns:
        input_df['expense_ratio'] = expense_ratio
    if 'net_cash_flow' in input_df.columns:
        input_df['net_cash_flow'] = net_cash_flow
    if 'debt_pressure' in input_df.columns:
        input_df['debt_pressure'] = debt_pressure

    try:
        # 2. Eksekusi Prediksi AI
        input_df = input_df[model_features] 
        prediction = model.predict(input_df)
        res_label = encoder.inverse_transform(prediction)
        kondisi = res_label[0]
        
        # 3. Layout Pengelompokan Hasil & Insight
        tab1, tab2 = st.tabs(["🎯 Hasil Analisis & Rekomendasi", "📈 Detail Komposisi Keuangan"])
        
        with tab1:
            st.markdown("### Status Kesehatan Finansial")
            
            # Tampilan Status Modern menggunakan Containers berbatasan
            with st.container(border=True):
                col_status, col_desc = st.columns([1, 2])
                with col_status:
                    if kondisi == 'Growth':
                        st.html("<h2 style='color:#2ecc71; margin-top:0;'>🚀 GROWTH</h2>")
                        st.html("<span style='background-color:rgba(46,204,113,0.1); padding:5px 10px; border-radius:5px; color:#2ecc71;'>Sangat Sehat</span>")
                    elif kondisi == 'Stable':
                        st.html("<h2 style='color:#3498db; margin-top:0;'>✅ STABLE</h2>")
                        st.html("<span style='background-color:rgba(52,152,219,0.1); padding:5px 10px; border-radius:5px; color:#3498db;'>Cukup Aman</span>")
                    else:
                        st.html("<h2 style='color:#e74c3c; margin-top:0;'>⚠️ WARNING</h2>")
                        st.html("<span style='background-color:rgba(231,76,60,0.1); padding:5px 10px; border-radius:5px; color:#e74c3c;'>Waspada!</span>")
                
                with col_desc:
                    if kondisi == 'Growth':
                        st.markdown("**Analisis AI:** Arus kas Anda bekerja dengan sangat optimal. Anda memiliki ruang yang luas untuk mengalokasikan dana ke pos investasi jangka panjang.")
                    elif kondisi == 'Stable':
                        st.markdown("**Analisis AI:** Finansial Anda berada dalam posisi seimbang dan aman. Fokus utama saat ini adalah memperkuat fondasi dana darurat sebelum beralih ke instrumen berisiko tinggi.")
                    else:
                        st.markdown("**Analisis AI:** Deteksi pola pengeluaran atau beban cicilan Anda berada dalam zona kritis. Diperlukan evaluasi segera dan pemangkasan pengeluaran non-prioritas.")

            st.markdown("### Insight Strategis")
            # Tampilan Metrik Dinamis dengan Pembungkus Card Modern
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                
                m1.metric(
                    label="Rasio Pengeluaran", 
                    value=f"{expense_ratio * 100:.1f}%", 
                    delta="Di Bawah Batas Aman (≤ 70%)" if expense_ratio <= 0.7 else "Melebihi Batas Aman (> 70%)",
                    delta_color="normal" if expense_ratio <= 0.7 else "inverse"
                )
                
                m2.metric(
                    label="Sisa Kas (Net Cash Flow)", 
                    value=f"IDR {net_cash_flow:,.0f}", 
                    delta="Surplus Akumulatif" if net_cash_flow > 0 else "Mengalami Defisit",
                    delta_color="normal" if net_cash_flow > 0 else "inverse"
                )
                
                # Menentukan parameter alokasi berdasarkan kondisi
                if kondisi == 'Growth':
                    rekomendasi_saving = income * 0.30
                    tips = "Alokasikan ke instrumen investasi produktif"
                elif kondisi == 'Stable':
                    rekomendasi_saving = income * 0.20
                    tips = "Saran simpan di reksadana pasar uang"
                else:
                    rekomendasi_saving = income * 0.10
                    tips = "Tahan pengeluaran tersier sekunder!"
                    
                m3.metric(
                    label="Target Tabungan Minimal", 
                    value=f"IDR {rekomendasi_saving:,.0f}", 
                    delta=tips
                )

        with tab2:
            st.markdown("### Breakdown Indikator Keuangan")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**Pendapatan Bersih:** IDR {income:,.0f}")
                    st.info(f"**Alokasi Belanja:** IDR {expense:,.0f}")
                with c2:
                    st.info(f"**Beban Cicilan/Hutang:** IDR {debt:,.0f}")
                    st.info(f"**Rasio Tekanan Hutang:** {debt_pressure * 100:.1f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat engine memproses komputasi: {e}")

else:
    # State awal ketika aplikasi dibuka, disajikan lewat struktur card info modern
    with st.container(border=True):
        st.markdown("#### 👋 Selamat Datang di FINARY Analytics Platform")
        st.info("Sistem siap menerima parameter data. Silakan lengkapi formulir keuangan Anda di sidebar sebelah kiri, kemudian klik tombol **Analisis Finansial** untuk mengaktifkan model prediksi.")
