# 🛠 Predictive Maintenance System – RUL Prediction

This project is an end-to-end **Predictive Maintenance System** built using **Python**, **TensorFlow / Keras**, and **Gradio**, and deployed on **Hugging Face Spaces**.

It predicts the **Remaining Useful Life (RUL)** of industrial machines from multivariate time-series sensor data such as vibration, temperature, pressure, and voltage.

---

## ✨ Features

- Deep Learning based **RUL Prediction** using LSTM networks  
- **Multivariate Time-Series Modeling** with sliding windows  
- **Frequency-domain Feature Extraction (FFT)** for vibration analysis  
- **Machine Degradation Trend Visualization**  
- Interactive **Gradio Web Dashboard**   

---

## 🧠 Model Details

- **Model Used:** LSTM (Recurrent Neural Network)  
- **Input Window Size:** 30 time-steps × 21 sensors  
- **Loss Function:** Mean Squared Error (MSE)  
- **Evaluation Metric:** RMSE  
- **Best RMSE:** ~13 cycles  

---

## 📂 Datasets

- **NASA C-MAPSS FD001** – Remaining Useful Life Benchmark  
- **AI4I 2020 Dataset** – Synthetic Predictive Maintenance  
- **SKAB Dataset** – Multivariate Anomaly Detection  

---

## 🗂 Project Structure
```
Predictive-Maintenance-System/
│
├── data/
│   └── cmapss/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rul_modeling.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_evaluation_analysis.ipynb
│
├── models/
│   └── rul_lstm_model_fd001.keras
│
├── gradio_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶ Run Locally

```bash
pip install -r requirements.txt
python gradio_app.py
```

---
