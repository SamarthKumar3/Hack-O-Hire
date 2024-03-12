from flask import Flask, request, jsonify
import pandas as pd
from pyod.models.iforest import IForest
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
# Assuming df_withdrawals is the preprocessed DataFrame and clf is the trained model
anomaly_proportion = 0.001
clf_name = 'Anomaly Detection - Isolation Forest'
clf = IForest(contamination=anomaly_proportion)

# Assuming xx, yy, Z, threshold are calculated as per your provided code
xx, yy = np.meshgrid(np.linspace(0, 11, 200), np.linspace(0, 180000, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])*-1
Z = Z.reshape(xx.shape)
threshold = (df_withdrawals.loc[df_withdrawals['y_pred'] == 1, 'y_scores'].min(
)*-1)/2 + (df_withdrawals.loc[df_withdrawals['y_pred'] == 0, 'y_scores'].max()*-1)/2


@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.get_json()
    # Assuming data contains a CSV file with the same structure as df_withdrawals
    df = pd.read_csv(data['csv_file'])

    # Apply preprocessing steps
    df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
    df['type'] = df['type'].replace(
        {'PRIJEM': 'CREDIT', 'VYDAJ': 'WITHDRAWAL', 'VYBER': 'NOT SURE'})
    df = df.query('type == "WITHDRAWAL"').sort_values(
        by=['account_id', 'date']).set_index('date')
    df['sum_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).sum())
    df['count_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).count())

    return jsonify({'message': 'Preprocessing completed.'})


@app.route('/predict', methods=['POST'])
def predict_anomalies():
    data = request.get_json()
    X = pd.DataFrame(data['data'], columns=['count_5days', 'sum_5days'])

    # Predict anomalies
    y_pred = clf.predict(X)
    y_scores = clf.decision_function(X)

    # Generate plot
    fig, subplot = plt.subplots(1, 1)
    subplot.contourf(xx, yy, Z, levels=np.linspace(
        Z.min(), threshold, 10), cmap=plt.cm.Blues_r)
    a = subplot.contour(xx, yy, Z, levels=[
                        threshold], linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    b = subplot.scatter(X[y_pred == 0]['count_5days'], X[y_pred == 0]
                        ['sum_5days'], c='white', s=20, edgecolor='k')
    c = subplot.scatter(X[y_pred == 1]['count_5days'], X[y_pred == 1]
                        ['sum_5days'], c='black', s=20, edgecolor='r')
    subplot.axis('tight')
    subplot.legend([a.collections[0], b, c], [
                   'learned decision function', 'inliers', 'outliers'], loc='upper right')
    subplot.set_title(clf_name)
    subplot.set_xlim((0, 11))
    subplot.set_ylim((0, 180000))
    subplot.set_xlabel("5-day count of withdrawal transactions.")
    subplot.set_ylabel("5-day sum of withdrawal transactions")

    # Save the plot as a PNG file
    plot_filename = 'anomaly_plot.png'
    fig.savefig(plot_filename)

    return jsonify({'y_pred': y_pred.tolist(), 'y_scores': y_scores.tolist(), 'plot_filename': plot_filename})


if __name__ == '__main__':
    app.run(debug=True)
