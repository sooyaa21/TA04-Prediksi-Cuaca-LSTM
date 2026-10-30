
from flask import Flask, render_template
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# =====================================
# 1. MEMUAT MODEL DAN SCALER
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model_lstm_weather.keras")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler_weather.pkl")
)


# =====================================
# 2. NILAI EVALUASI MODEL
# =====================================

RMSE = 1.2235
MAE = 0.9252


# =====================================
# 3. MEMBACA DATASET CUACA
# =====================================

DATA_PATH = os.path.join(BASE_DIR, "weather_data.csv")

df = pd.read_csv(DATA_PATH)

if "DATE" not in df.columns:
    df["DATE"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MONTH"],
            day=df["DAY"]
        )
    )
else:
    df["DATE"] = pd.to_datetime(df["DATE"])

df["TEMPERATURE"] = pd.to_numeric(
    df["TEMPERATURE"],
    errors="coerce"
)

df = df.dropna(
    subset=["DATE", "TEMPERATURE"]
)

df = df.sort_values(
    "DATE"
).reset_index(drop=True)


# =====================================
# 4. FUNGSI PREDIKSI SUHU BERIKUTNYA
# =====================================

def get_prediction():

    temperatures = df["TEMPERATURE"].values.reshape(-1, 1)

    if len(temperatures) < 30:
        raise ValueError(
            "Dataset harus memiliki minimal 30 data suhu."
        )

    last_30 = temperatures[-30:]

    scaled_data = scaler.transform(last_30)

    input_data = scaled_data.reshape(1, 30, 1)

    prediction_scaled = model.predict(
        input_data,
        verbose=0
    )

    prediction_actual = scaler.inverse_transform(
        prediction_scaled
    )

    return float(prediction_actual[0][0])


# =====================================
# 5. DATA PERBANDINGAN AKTUAL DAN PREDIKSI
# =====================================

def get_comparison_data():

    temperatures = df["TEMPERATURE"].values.reshape(-1, 1)

    if len(temperatures) <= 30:
        return [], [], []

    scaled_data = scaler.transform(temperatures)

    X = []
    y = []

    for i in range(30, len(scaled_data)):
        X.append(scaled_data[i-30:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.8)

    X_test = X[split_index:]
    y_test = y[split_index:]

    X_test = X_test.reshape(
        X_test.shape[0], 30, 1
    )

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions_actual = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual_values = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    test_dates = df["DATE"].iloc[
        30 + split_index:
    ]

    dates = test_dates.dt.strftime(
        "%d-%m-%Y"
    ).tolist()

    actual = np.round(
        actual_values, 2
    ).tolist()

    predicted = np.round(
        predictions_actual, 2
    ).tolist()

    return dates, actual, predicted


# =====================================
# 6. FUNGSI MENAMPILKAN DASHBOARD
# =====================================

def render_dashboard(prediction=None, error=None):

    history = df.tail(30)

    dates = history["DATE"].dt.strftime(
        "%d-%m-%Y"
    ).tolist()

    temperatures = history["TEMPERATURE"].round(
        2
    ).tolist()

    comparison_dates, actual, predicted = get_comparison_data()

    return render_template(
        "index.html",
        dates=dates,
        temperatures=temperatures,
        total_data=len(df),
        prediction=prediction,
        error=error,
        comparison_dates=comparison_dates,
        actual=actual,
        predicted=predicted,
        rmse=RMSE,
        mae=MAE
    )


# =====================================
# 7. HALAMAN UTAMA
# =====================================

@app.route("/")
def index():

    return render_dashboard()


# =====================================
# 8. PROSES PREDIKSI
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        result = get_prediction()

        return render_dashboard(
            prediction=round(result, 2)
        )

    except Exception as e:

        return render_dashboard(
            error=str(e)
        )


# =====================================
# 9. MENJALANKAN APLIKASI
# =====================================

if __name__ == "__main__":
    app.run(debug=True)
