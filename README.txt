# 📈 Prediksi Saham BBCA dengan LSTM dan Streamlit

Proyek ini menggunakan model LSTM untuk memprediksi harga saham BBCA berdasarkan data historis dari tahun 2019 hingga 2025.

## 🔧 Fitur Utama
- Prediksi historis vs aktual (grafik)
- Forecast harga saham 1–12 bulan ke depan
- Ringkasan harga prediksi tiap bulan
- UI interaktif via Streamlit

## Struktur Proyek
BBCA-LSTM-PROJECT/
├── app.py
├── data/
│ └── bbca_stock_2019-2025.csv
├── model/
│ └── lstm_model.h5
├── utils/
│ └── preprocessing.py
├── requirements.txt
└── README.md

## 🚀 Cara Menjalankan Lokal

Clone repo ini:
   ```bash
   git clone https://github.com/username/bbca-lstm-project.git
   cd bbca-lstm-project
   pip install -r requirements.txt
   streamlit run app.py


---

## Langkah Deploy ke Streamlit Cloud
### 📌 Syarat:
- Sudah punya akun GitHub & Streamlit (gratis)
- Sudah push semua file ke GitHub

### 🚀 Langkah:
1. Login ke: [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Klik **"New App"**
3. Hubungkan ke repositori GitHub Anda
4. Pilih `main` branch dan `app.py` sebagai file utama
5. Klik **Deploy** 🎉

---

Kalau Anda belum punya file `.gitignore`, saya juga bisa bantu buatkan agar file besar atau environment tidak ikut terpush ke GitHub. Mau?
