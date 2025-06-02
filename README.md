# Blood Glucose Prediction using LSTM

This project implements an end-to-end deep learning pipeline to forecast future blood glucose levels using LSTM models. It includes both univariate (glucose-only) and multivariate (glucose + insulin + meal + time encoding) models. The models were trained on synthetic and pre-processed real-time patient data to predict next-hour glucose values.

---

## 📅 Features

* LSTM-based time series forecasting
* Univariate model using glucose history
* Multivariate model with insulin dose, meal timing, and time-of-day encoding
* Sequence construction with sliding windows
* Real-time prediction interface via command-line script
* Modular and reproducible training pipeline

---

## 📊 Project Structure

```
glucose-prediction-lstm/
├── data-*.csv                  # Raw or synthetic time-series data
├── Diabetes (miultivar + Timeseries)-2.ipynb  # Jupyter notebook for training and feature extraction
├── glucose_predictor.py        # CLI tool for running predictions
├── univariate_input.csv        # Sample input for univariate model
├── multivariate_input.csv      # Sample input for multivariate model
├── univariate_lstm_model.h5    # Trained univariate LSTM model
├── multivariate_lstm_model.h5  # Trained multivariate LSTM model
├── univariate_scaler.pkl       # Scaler for univariate input
├── multivariate_scaler.pkl     # Scaler for multivariate input
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
```

---

## 📈 How to Reproduce

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/glucose-prediction-lstm.git
cd glucose-prediction-lstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Launch and run all cells in the Jupyter notebook to train the models:

```bash
jupyter notebook Diabetes\ \(miultivar\ +\ Timeseries\)-2.ipynb
```

---

## 📉 Predict Next Glucose Value

### 🔢 Univariate Model

```bash
python glucose_predictor.py \
  --input_csv univariate_input.csv \
  --model univariate_lstm_model.h5 \
  --scaler univariate_scaler.pkl
```

### 🔢 Multivariate Model

```bash
python glucose_predictor.py \
  --input_csv multivariate_input.csv \
  --model multivariate_lstm_model.h5 \
  --scaler multivariate_scaler.pkl
```

Expected output:

```
Predicted glucose for next hour: 145.67 mg/dL
```

---

## 🚀 Future Work

* Streamlit interface for user-friendly input
* Real-time ingestion of live glucose sensor data
* Model evaluation on benchmark datasets

---

## 📄 License

This project is licensed under the MIT License.

---

## ✍️ Author

Eswar Kamisetti
Graduate Student, Health Informatics
Indiana University

---

> For any questions or collaboration, please open an issue or contact via GitHub.
