import streamlit as st
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib


def filter_bank_data(df=pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['Debit'], inplace=True)
    df = df[df['Debit'] != ' ']
    df = df.assign(type='Debit')
    df.rename(columns={'Debit': 'amount'}, inplace=True)
    df['amount'] = df['amount'].replace(',', '', regex=True).astype(float)
    df.drop(columns=['Txn Date'], inplace=True)
    df['account-id'] = df['account-id'].apply(lambda s: int(s.split()[2]))
    df.drop(columns=['Credit'], inplace=True)
    df.rename(columns={'Date': 'date',
              'account-id': 'account_id'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df.set_index('date', inplace=True)
    df['sum_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).sum())
    df['count_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).count())
    df_new = df
    return df_new


def check_amount_and_sum(df=pd.DataFrame) -> pd.DataFrame:
    df['amount'] = df['amount'].replace(',', '', regex=True).astype(float)
    df['sum_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).sum())
    df['count_5days'] = df.groupby('account_id')['amount'].transform(
        lambda s: s.rolling(timedelta(days=5)).count())
    df_new = df

    return df_new


def visualize_anomalies_with_date(df):
    # Create a subplot with Plotly
    fig = make_subplots(rows=1, cols=1)

    # Plot non-anomalous data
    fig.add_trace(go.Scatter(x=df[df['y_pred'] == 0].index,
                             y=df[df['y_pred'] == 0]['sum_5days'] / 1000,
                             mode='markers',
                             marker=dict(color='blue', symbol='circle'),
                             name='Non-Anomalous'),
                  row=1, col=1)

    # Highlight anomalous data
    fig.add_trace(go.Scatter(x=df[df['y_pred'] == 1].index,
                             y=df[df['y_pred'] == 1]['sum_5days'] / 1000,
                             mode='markers',
                             marker=dict(color='red', symbol='x'),
                             name='Anomalous'),
                  row=1, col=1)

    # Set layout options
    fig.update_layout(title='Anomalies Detected Over Time',
                      xaxis_title='Date',
                      yaxis_title='Sum of 5-day withdrawals (in thousands)',
                      showlegend=True,
                      hovermode='x unified',
                      xaxis=dict(
                          # Stretching x-axis by 5 days
                          range=[df.index.min(), df.index.max() + \
                                 pd.DateOffset(days=5)]
                      ))

    # Annotate anomalies with their exact values
    for index, row in df[df['y_pred'] == 1].iterrows():
        fig.add_annotation(x=index, y=row['sum_5days'] / 1000,
                           text=f"Anomaly\n{row['sum_5days'] / 1000:.2f}k",
                           showarrow=True,
                           arrowhead=1,
                           arrowcolor='red',
                           arrowwidth=1,
                           ax=20,
                           ay=-40,
                           font=dict(size=10, color='red'))

    # Show plot
    st.plotly_chart(fig)


def visualize_anomalies_3d(df):
    # Create figure
    fig = go.Figure()

    # Extract data for inliers and outliers
    inliers = df[df['y_pred'] == 0]
    outliers = df[df['y_pred'] == 1]

    # Plot inliers
    fig.add_trace(go.Scatter3d(
        x=inliers['count_5days'],
        y=inliers['amount'],
        z=inliers['sum_5days'],
        mode='markers',
        marker=dict(
            color='white',
            size=5,
            line=dict(
                color='black',
                width=0.5
            )
        ),
        name='Inliers'
    ))

    # Plot outliers
    fig.add_trace(go.Scatter3d(
        x=outliers['count_5days'],
        y=outliers['amount'],
        z=outliers['sum_5days'],
        mode='markers',
        marker=dict(
            color='black',
            size=5,
            line=dict(
                color='red',
                width=0.5
            )
        ),
        name='Outliers'
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='5-day count of withdrawal transactions',
            yaxis_title='Transaction Amount',
            zaxis_title='Sum of 5-day withdrawals',
            aspectmode='cube'
        ),
        title='Anomalies Detected'
    )

    # Show plot
    st.plotly_chart(fig)


def main():
    st.title('Anomaly Detection in Bank Transaction Data')

    # Sidebar for uploading CSV and prediction
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        st.sidebar.write('File uploaded successfully!')

        # Predict button
        if st.sidebar.button('Predict'):
            # Load data
            df = pd.read_csv(uploaded_file)

            # Filter and preprocess data
            df_new = filter_bank_data(df)
            df_new = check_amount_and_sum(df_new)

            # Load the Isolation Forest model from the joblib file
            loaded_model = joblib.load('training_notebooks\clf.joblib')

            # Extract the relevant features from the new data
            X_new = df_new[['count_5days', 'sum_5days']]

            # Get the predictions and anomaly scores for the new data
            y_pred_new = loaded_model.predict(X_new)
            y_scores_new = loaded_model.decision_function(X_new)

            # Now you can use y_pred_new and y_scores_new for further analysis or visualization
            df_new['y_pred'] = y_pred_new
            df_new['y_scores'] = y_scores_new

            # Display anomalies
            st.subheader('Anomalies Detected Over Time')
            visualize_anomalies_with_date(df_new)

            st.subheader('Anomalies Detected (3D Visualization)')
            visualize_anomalies_3d(df_new)


if __name__ == "__main__":
    main()
