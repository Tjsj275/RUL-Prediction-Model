import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("models/rul_lstm_model_fd001.keras")

def predict_rul(file):
    df = pd.read_csv(file.name)
    sensor_cols = [c for c in df.columns if "sensor_" in c]

    if len(df) < 30:
        return "Upload at least 30 rows.", None

    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    preds = []
    for i in range(30, len(df)):
        X = np.expand_dims(df[sensor_cols].values[i-30:i], axis=0)
        preds.append(model.predict(X)[0][0])

    # Create degradation plot
    fig, ax = plt.subplots()
    ax.plot(preds)
    ax.set_title("RUL Degradation Trend")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted RUL")

    rul = int(preds[-1])

    if rul > 100:
        status = "🟢 Healthy"
    elif rul > 40:
        status = "🟡 Moderate Degradation"
    else:
        status = "🔴 Critical – Maintenance Required"

    return f"Predicted RUL: {rul} cycles\nStatus: {status}", fig


gr.Interface(
    fn=predict_rul,
    inputs=gr.File(label="Upload Sensor CSV"),
    outputs=[gr.Textbox(label="Result"), gr.Plot(label="Degradation Trend")],
    title="Industrial Machine RUL Monitoring System"
).launch()
