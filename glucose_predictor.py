import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

class GlucosePredictor:
    def __init__(self, model_path, scaler_path, window_size=24):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size

    def preprocess(self, df):
        df = df.copy()
        if 'hour_sin' not in df.columns and 'glucose' in df.columns:
            df['hour'] = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df.drop(columns='hour', inplace=True)

        # Ensure correct feature order
        expected_cols = self.scaler.feature_names_in_
        df = df[expected_cols]
        scaled = self.scaler.transform(df)

        return scaled

    def create_sequence(self, data):
        return data[-self.window_size:].reshape(1, self.window_size, data.shape[1])

    def predict_next(self, df):
        data = self.preprocess(df)
        X = self.create_sequence(data)
        y_scaled = self.model.predict(X)

        # Pad with other features for inverse transform
        if X.shape[2] > 1:
            padded = np.concatenate([y_scaled, X[0, -1, 1:].reshape(1, -1)], axis=1)
        else:
            padded = y_scaled

        y_inv = self.scaler.inverse_transform(padded)[:, 0]
        return float(y_inv[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--model", required=True, help="Path to trained model (.h5)")
    parser.add_argument("--scaler", required=True, help="Path to scaler (.pkl)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, parse_dates=True, index_col=0)
    predictor = GlucosePredictor(args.model, args.scaler)
    prediction = predictor.predict_next(df)
    print(f"Predicted glucose for next hour: {prediction:.2f} mg/dL")
