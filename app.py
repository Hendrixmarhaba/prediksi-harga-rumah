import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Judul aplikasi
st.title("🏡 Aplikasi Prediksi Harga Rumah")
st.write("Masukkan data rumah untuk memprediksi harganya (contoh model regresi sederhana).")

# Input fitur rumah
luas_tanah = st.number_input("Luas Tanah (m²)", min_value=20, max_value=1000, value=100)
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=1000, value=80)
jumlah_kamar = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
jarak_pusat_kota = st.slider("Jarak ke Pusat Kota (km)", 0, 50, 10)

# Data training sederhana (contoh)
data = {
    'luas_tanah': [100, 150, 200, 250, 300],
    'luas_bangunan': [80, 120, 160, 200, 240],
    'jumlah_kamar': [3, 4, 4, 5, 6],
    'jarak_pusat_kota': [10, 8, 6, 4, 2],
    'harga': [500, 700, 900, 1100, 1300]  # juta rupiah
}
df = pd.DataFrame(data)

# Membuat model regresi
X = df[['luas_tanah', 'luas_bangunan', 'jumlah_kamar', 'jarak_pusat_kota']]
y = df['harga']
model = LinearRegression()
model.fit(X, y)

# Prediksi harga berdasarkan input pengguna
input_data = np.array([[luas_tanah, luas_bangunan, jumlah_kamar, jarak_pusat_kota]])
prediksi = model.predict(input_data)[0]

# Tampilkan hasil
st.subheader("💰 Hasil Prediksi")
st.write(f"Perkiraan harga rumah: **Rp {prediksi:.2f} juta**")

st.caption("Model ini hanya contoh sederhana berbasis Linear Regression.")
