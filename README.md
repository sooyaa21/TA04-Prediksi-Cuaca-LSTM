# Prediksi Suhu Cuaca Menggunakan LSTM

## Deskripsi Proyek
Proyek ini merupakan aplikasi web untuk memprediksi suhu cuaca menggunakan metode **Long Short-Term Memory (LSTM)**. Model memanfaatkan data suhu historis untuk memperkirakan suhu pada pengamatan berikutnya.

Aplikasi dikembangkan menggunakan Python dan Flask, serta menampilkan hasil prediksi melalui dashboard yang dilengkapi grafik dan informasi evaluasi model.

## Tujuan
- Menerapkan metode LSTM untuk prediksi data suhu cuaca.
- Mengolah data historis menjadi urutan data (sequence).
- Mengintegrasikan model hasil pelatihan ke dalam aplikasi web.
- Menampilkan hasil prediksi dan evaluasi model.

## Dataset
Dataset yang digunakan adalah **Daily Weather Dataset** dari Kaggle.

Sumber dataset:  
https://www.kaggle.com/datasets/mdforhadmia/daily-weather-dataset/code

Data difilter berdasarkan satu titik koordinat, yaitu LAT 20.5 dan LON 88.12, sehingga diperoleh 392 data pengamatan.

## Metode
Metode yang digunakan adalah Long Short-Term Memory (LSTM), dengan tahapan:
1. Pengumpulan dan pemilihan data.
2. Pembersihan dan pengolahan data.
3. Normalisasi menggunakan MinMaxScaler.
4. Pembentukan sequence menggunakan 30 data sebelumnya.
5. Pelatihan model LSTM selama 50 epoch.
6. Evaluasi menggunakan RMSE dan MAE.
7. Integrasi model ke aplikasi web Flask.

## Hasil Evaluasi
| Metrik | Hasil |
|---|---:|
| RMSE | 1.2235 |
| MAE | 0.9252 |

## Teknologi yang Digunakan
- Python
- TensorFlow dan Keras
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML dan CSS

## Fitur Aplikasi
- Menampilkan data suhu historis.
- Melakukan prediksi suhu berikutnya.
- Menampilkan grafik perbandingan suhu aktual dan prediksi.
- Menampilkan nilai evaluasi RMSE dan MAE.

## Google Colab
Notebook pelatihan model:  
https://colab.research.google.com/drive/15R8tnHmjuTsWebDyMnAF-rrD4FVctbfu?usp=sharing

## Cara Menjalankan Aplikasi
1. Clone repository ini.
2. Buat dan aktifkan virtual environment.
3. Instal library yang diperlukan.
4. Jalankan aplikasi Flask menggunakan perintah `python app.py`.
5. Buka `http://127.0.0.1:5000` pada browser.

## Penulis
**Danar Iswara**  
Program Studi Teknik Informatika  
Universitas Bale Bandung
