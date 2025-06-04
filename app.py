# === [IMPORT] ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.preprocessing import load_and_clean_data

# === [LAYOUT CONFIG] ===
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham BBCA')
st.markdown("---")

# === [PENJELASAN UTAMA] ===
st.markdown("""
### Tentang Prediksi Ini
Model ini menggunakan algoritma **Long Short-Term Memory (LSTM)** untuk memprediksi harga saham BBCA berdasarkan data historis tahun 2019 hingga 2025.  
Model dilatih menggunakan data harga penutupan dan divalidasi menggunakan metrik evaluasi seperti RMSE, MAE, dan MAPE.
""")

# === [LOAD DATA & MODEL] ===
df = load_and_clean_data('data/bbca_stock_2019-2025.csv')
model = load_model("model/lstm_model.h5")
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']].values)

# === [SEQUENCE FUNCTION] ===
def create_sequences(data, seq_len=60):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(x), np.array(y)

X, y = create_sequences(scaled_close)

# === [PREDIKSI HISTORIS] ===
predicted = model.predict(X)
predicted_actual = scaler.inverse_transform(predicted)
actual_actual = scaler.inverse_transform(y)
dates = df['Date'].iloc[60:].reset_index(drop=True)

# === [GRAFIK HISTORIS] ===
st.subheader('Prediksi Historis vs Aktual')

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(dates, actual_actual, label='Actual', linewidth=1.5)
ax1.plot(dates, predicted_actual, label='Predicted', linewidth=1.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax1.set_xlabel('Bulan')
ax1.set_ylabel('Harga Saham')
ax1.legend()
ax1.grid(True)
plt.xticks(rotation=0)
st.pyplot(fig1)

# === [EVALUASI AKURASI] ===
rmse = np.sqrt(mean_squared_error(actual_actual, predicted_actual))
mae = mean_absolute_error(actual_actual, predicted_actual)
mape = np.mean(np.abs((actual_actual - predicted_actual) / actual_actual)) * 100

st.subheader("Evaluasi Model")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RMSE", f"{rmse:,.2f}")
with col2:
    st.metric("MAE", f"{mae:,.2f}")
with col3:
    st.metric("MAPE", f"{mape:.2f}%")

st.markdown("---")

# === [CTA: Prediksi Forecasting] ===
st.header("Prediksi Harga Saham BBCA ke Depan")
st.markdown("""
Gunakan fitur di bawah ini untuk memproyeksikan harga saham BBCA hingga 12 bulan ke depan berdasarkan pola historis.
""")

bulan = st.selectbox(
    "Pilih jumlah bulan yang ingin diprediksi:",
    options=[1, 3, 6, 9, 12],
    format_func=lambda x: f"{x} bulan"
)
hari_prediksi = bulan * 30

# === [FORECASTING] ===
future_input = scaled_close[-60:].reshape(1, 60, 1)
future_preds = []

for _ in range(hari_prediksi):
    next_pred = model.predict(future_input)[0][0]
    future_preds.append(next_pred)
    future_input = np.append(future_input[:, 1:, :], [[[next_pred]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# === [PLOT: Forecast Mingguan] ===
# Buat label mingguan
mingguan_label = [f"{i+1}" for i in range(len(future_prices) // 7)]
mingguan_data = [future_prices[i * 7][0] for i in range(len(mingguan_label))]

# Tentukan frekuensi tampilan label X
if bulan == 1 or bulan == 3:
    xtick_interval = 1  # tampilkan semua minggu
elif bulan == 6 or bulan == 9 or bulan == 12:
    xtick_interval = 4  # tampilkan tiap 4 minggu
else:
    xtick_interval = 1

# Plot grafik
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(mingguan_label, mingguan_data, marker='o', linestyle='-')
ax2.set_title(f'Forecast Harga Saham BBCA {bulan} Bulan ke Depan (Per Minggu)')
ax2.set_xlabel('Minggu ke-')
ax2.set_ylabel('Harga Prediksi')
ax2.grid(True)

# Atur label sumbu X sesuai interval
ax2.set_xticks(np.arange(0, len(mingguan_label), step=xtick_interval))
ax2.set_xticklabels([mingguan_label[i] for i in range(0, len(mingguan_label), xtick_interval)])
st.pyplot(fig2)

# === [RINGKASAN PREDIKSI PER BULAN] ===
st.subheader("Ringkasan Prediksi per Bulan")

bulan_terprediksi = future_prices.reshape(-1, 30)
for i, bulan_harga in enumerate(bulan_terprediksi, start=1):
    harga_akhir = bulan_harga[-1]
    st.write(f"Akhir Bulan ke-{i}: Rp {harga_akhir:,.0f}")
