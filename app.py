import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Konfigurasi Halaman Komersial
st.set_page_config(
    page_title="FINARY Intelligence | Financial Advisory Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate Style Sheet (Adaptif Mode Terang & Gelap)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Menggunakan font Inter untuk seluruh elemen */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* KARTU METRIK: Menggunakan variabel bawaan Streamlit agar otomatis berubah warna */
    .consulting-card {
        background-color: var(--background-secondary-color); /* Otomatis abu-abu terang di light mode, abu-abu gelap di dark mode */
        border-left: 4px solid var(--primary-color);        /* Garis aksen mengikuti tema utama Streamlit */
        padding: 24px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* TEXT STYLING: Memanfaatkan variabel text-color agar tidak tenggelam saat background berubah */
    .kpi-title { 
        font-size: 12px; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        color: var(--text-color);
        opacity: 0.7; /* Membuat teks judul metrik sedikit lebih redup (elegan) */
        font-weight: 600; 
    }
    
    .kpi-value { 
        font-size: 32px; 
        font-weight: 700; 
        color: var(--text-color); /* Otomatis hitam di light mode, putih di dark mode */
        margin: 8px 0; 
    }
    
    .kpi-benchmark { 
        font-size: 11.5px; 
        color: var(--text-color);
        opacity: 0.5;
    }
    
    /* STATUS BADGES: Dibuat sedikit transparan agar warna teks tetap terbaca tajam */
    .badge-growth { background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-stable { background: rgba(59, 130, 246, 0.15); color: #3b82f6; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-warning { background: rgba(239, 68, 68, 0.15); color: #ef4444; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

def load_data(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    elif os.path.exists(os.path.join('finary', file_name)):
        return joblib.load(os.path.join('finary', file_name))
    else:
        raise FileNotFoundError(f"File {file_name} tidak ditemukan pada repositori data.")

try:
    model = load_data('model_finary.pkl')
    encoder = load_data('label_encoder.pkl')
    model_features = load_data('model_features.pkl')
except Exception as e:
    st.error(f"Sistem gagal memuat pustaka analitik internal: {e}")
    st.stop()

# ------------------ SIDEBAR ADVISORY CONTROLLER ------------------
st.sidebar.markdown("<br><p class='kpi-title'>Parameter Input</p>", unsafe_allow_html=True)
st.sidebar.markdown("Konfigurasikan variabel neraca likuiditas di bawah ini untuk memulai audit prediktif.")

with st.sidebar.form(key="financial_form"):
    income = st.number_input("Pendapatan Bersih Bulanan (IDR)", min_value=0, value=0, step=500000)
    expense = st.number_input("Beban Operasional / Pengeluaran (IDR)", min_value=0, value=0, step=250000)
    debt = st.number_input("Liabilitas / Komitmen Cicilan (IDR)", min_value=0, value=0, step=100000)
    
    submit_button = st.form_submit_button(label="Jalankan Diagnostik Finansial", use_container_width=True)

# ------------------ TOP NAVIGATION / BRANDING ------------------
col_brand, col_nav = st.columns([1, 1])
with col_brand:
    st.markdown("<h2 style='font-weight:700; color:#0f172a; margin-bottom:0;'>FINARY <span style='font-weight:300; color:#64748b;'>Intelligence</span></h2>", unsafe_allow_html=True)
    st.caption("Firma Penasihat Manajemen Finansial Berbasis Komputasi Prediktif & Machine Learning")

st.markdown("<br>", unsafe_allow_html=True)

# Feature Engineering Internals
expense_ratio = expense / income if income > 0 else 0
net_cash_flow = income - expense
debt_pressure = debt / income if income > 0 else 0

# ------------------ EXECUTIVE DIAGNOSTIC LOGIC ------------------
if submit_button:
    if income == 0:
        st.warning("Gagal Menjalankan Diagnostik: Variabel Pendapatan Bersih Bulanan wajib diisi untuk menghindari eror kalkulasi rasio.")
        st.stop()

    # Formulasi Model Dataframe
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    if 'expense_ratio' in input_df.columns:
        input_df['expense_ratio'] = expense_ratio
    if 'net_cash_flow' in input_df.columns:
        input_df['net_cash_flow'] = net_cash_flow
    if 'debt_pressure' in input_df.columns:
        input_df['debt_pressure'] = debt_pressure

    try:
        # Eksekusi Komputasi Random Forest
        input_df = input_df[model_features] 
        prediction = model.predict(input_df)
        res_label = encoder.inverse_transform(prediction)
        kondisi = res_label[0]
        
        # Algoritma Scoring Dinamis ala Konsultan (Financial Health Index: 0 - 100)
        # Menghitung skor secara matematis berdasarkan kombinasi performa rasio
        base_score = 100
        base_score -= (expense_ratio * 40)  # Bobot penalti rasio pengeluaran (Max 40 poin)
        base_score -= (debt_pressure * 40)   # Bobot penalti rasio utang (Max 40 poin)
        if net_cash_flow < 0:
            base_score -= 20                 # Penalti arus kas negatif
        fhi_score = max(min(int(base_score), 100), 10) # Mengunci jangkauan skor di 10 - 100
        
        # --- RENDER DASHBOARD CONSULTING STYLE ---
        st.markdown("<p class='kpi-title'>Ringkasan Eksekutif & Hasil Audit AI</p>", unsafe_allow_html=True)
        
        # Row 1: KPI Dashboard Block Modern
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        
        with c_kpi1:
            st.markdown(f"""
                <div class='consulting-card'>
                    <div class='kpi-title'>Financial Health Index (FHI)</div>
                    <div class='kpi-value'>{fhi_score} <span style='font-size:16px; font-weight:400; color:#64748b;'>/ 100</span></div>
                    <div class='kpi-benchmark'>Skor ideal manajemen aset modern &ge; 75.0</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_kpi2:
            status_html = ""
            if kondisi == 'Growth':
                status_html = f"<div class='kpi-value' style='color:#059669;'>{kondisi} <span class='badge-growth'>Sangat Sehat</span></div>"
            elif kondisi == 'Stable':
                status_html = f"<div class='kpi-value' style='color:#2563eb;'>{kondisi} <span class='badge-stable'>Optimal</span></div>"
            else:
                status_html = f"<div class='kpi-value' style='color:#dc2626;'>{kondisi} <span class='badge-warning'>Risiko Tinggi</span></div>"
                
            st.markdown(f"""
                <div class='consulting-card'>
                    <div class='kpi-title'>Prediksi Model Klasifikasi AI</div>
                    {status_html}
                    <div class='kpi-benchmark'>Berdasarkan pengenalan pola algoritma Random Forest</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_kpi3:
            st.markdown(f"""
                <div class='consulting-card'>
                    <div class='kpi-title'>Retensi Kapasitas Modal</div>
                    <div class='kpi-value'>IDR {net_cash_flow:,.0f}</div>
                    <div class='kpi-benchmark'>Arus kas bersih pasca kewajiban operasional</div>
                </div>
                """, unsafe_allow_html=True)

        # Row 2: Deep Analysis (Insight Strategis Berdasarkan Data Riil)
        st.markdown("<br><p class='kpi-title'>Temuan Utama & Arahan Strategis</p>", unsafe_allow_html=True)
        
        col_analysis, col_breakdown = st.columns([3, 2])
        
        with col_analysis:
            with st.container(border=True):
                st.markdown("#### **Formulasi Kebijakan Anggaran**")
                
                # Insight Dinamis Tingkat Lanjut (Insightful & Contextual)
                if kondisi == 'Growth':
                    st.markdown(f"""
                    * **Analisis Optimalisasi:** Struktur keuangan entitas saat ini berada pada tahap ekspansi yang sangat sehat dengan rasio pengeluaran di angka **{expense_ratio * 100:.1f}%**. Kelebihan likuiditas ini merupakan peluang strategis.
                    * **Rekomendasi Penempatan Modal:** Konsultan menyarankan untuk mengamankan minimal **IDR {income * 0.30:,.0f} (30%)** ke dalam portofolio instrumen pertumbuhan agresif guna mengimbangi laju inflasi jangka panjang.
                    """)
                elif kondisi == 'Stable':
                    st.markdown(f"""
                    * **Analisis Konsolidasi:** Entitas berada dalam posisi keseimbangan internal finansial yang baik. Rasio leverage/utang Anda saat ini terkontrol di angka **{debt_pressure * 100:.1f}%**. Namun, ruang akselerasi modal masih tertahan oleh pengeluaran bulanan.
                    * **Rekomendasi Pemeliharaan:** Prioritas mutlak dialokasikan sebesar **IDR {income * 0.20:,.0f} (20%)** ke instrumen pasar uang yang likuid guna memantapkan ketahanan dana darurat sebelum beralih ke ekspansi portofolio sekunder.
                    """)
                else:
                    st.markdown(f"""
                    * **Analisis Defisit & Kerentanan:** Sistem mendeteksi adanya anomali serius pada struktur manajemen arus kas Anda. Rasio belanja bulanan Anda menyentuh angka kritis **{expense_ratio * 100:.1f}%**, dikombinasikan dengan tekanan beban cicilan utang sebesar **{debt_pressure * 100:.1f}%**.
                    * **Rekomendasi Tindakan Korektif:** Diperlukan langkah efisiensi darurat. Amankan alokasi minimal terkunci **IDR {income * 0.10:,.0f} (10%)** untuk perlindungan likuiditas dasar, serta lakukan pemangkasan langsung pos pengeluaran tersier non-operasional minimum sebesar 25% dari posisi saat ini.
                    """)

        with col_breakdown:
            with st.container(border=True):
                st.markdown("#### **Matriks Pembanding Industri**")
                
                # Desain tabel informasi internal yang bersih ala firma riset
                metrics_data = {
                    "Metrik Tata Kelola": ["Rasio Belanja Bulanan", "Tekanan Utang (Debt Ratio)", "Ketersediaan Kas Sisa"],
                    "Nilai Entitas Anda": [f"{expense_ratio * 100:.1f}%", f"{debt_pressure * 100:.1f}%", f"IDR {net_cash_flow:,.0f}"],
                    "Ambang Batas Ideal": ["≤ 70.0%", "≤ 30.0%", "Surplus (> IDR 0)"]
                }
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Sistem mengalami hambatan teknis saat melakukan komputasi matriks: {e}")

else:
    # State awal aplikasi yang elegan, meniru landing page portal riset eksekutif
    st.markdown("<hr style='border: 0.5px solid #f1f5f9;'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
            <div style='padding: 20px 10px;'>
                <h4 style='margin-top:0; color:#0f172a; font-weight:600;'>Selamat Datang di Portal Analisis Makro FINARY</h4>
                <p style='color:#475569; font-size: 14px; line-height: 1.6;'>
                    Sistem siap mengeksekusi penilaian komparatif kesehatan keuangan entitas Anda menggunakan model klasifikasi buatan <i>Random Forest Optimizer</i>. 
                    Silakan isi metrik laporan neraca pendapatan dan beban Anda pada panel kontrol sebelah kiri, kemudian tekan tombol <b>"Jalankan Diagnostik Finansial"</b> untuk menerbitkan laporan kesimpulan eksekutif.
                </p>
            </div>
            """, unsafe_allow_html=True)
