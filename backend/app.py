from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta

app = Flask(__name__)

# Load the trained model
model_path = 'D:/Hack-O-Hire/models/model.joblib'
clf = joblib.load('D:\Hack-O-Hire\models\model.joblib')

# Assuming xx, yy, Z, threshold are calculated as per your provided code
xx, yy = np.meshgrid(np.linspace(0, 11, 200), np.linspace(0, 180000, 200))
Z = np.zeros_like(xx)  # Placeholder for Z
threshold = 0  # Placeholder for threshold value

# Define filter_features and preprocess_data functions


def filter_features(df):
    df = df[['date', 'account_id', 'type', 'amount']]
    df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
    return df


def preprocess_data(df):
    df = df[df['type'] == 'WITHDRAWAL']
    df.sort_values(by='account_id', inplace=True)
    df['sum_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).sum())
    df['count_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).count())
    return df


@app.route('/predict', methods=['POST'])
def predict_anomalies():
    # Load CSV data
    df = pd.read_csv('trans.csv')

    # Apply filtering and preprocessing steps
    df = filter_features(df)
    df = preprocess_data(df)

    # Predict anomalies
    X = df[['count_5days', 'sum_5days']]
    y_pred = clf.predict(X)
    y_scores = clf.decision_function(X)

    # Generate plot
    fig, subplot = plt.subplots(1, 1)
    # Add your contour plot code here using xx, yy, Z, threshold
    # Assuming Z, xx, yy are defined elsewhere as per your provided code
    # Save the plot as a PNG file
    plot_filename = 'anomaly_plot.png'
    fig.savefig(plot_filename)

    # Save predictions as a JSON file
    predictions_filename = 'anomaly_predictions.json'
    predictions_df = pd.DataFrame({'y_pred': y_pred, 'y_scores': y_scores})
    predictions_df.to_json(predictions_filename, orient='records')

    return jsonify({'y_pred': y_pred.tolist(), 'y_scores': y_scores.tolist(), 'plot_filename': plot_filename, 'predictions_filename': predictions_filename})


if __name__ == '__main__':
    app.run(debug=True)
