def predict(df=pd.DataFrame) -> pd.DataFrame:
    import joblib

# Load the Isolation Forest model from the joblib file
    loaded_model = joblib.load('training_notebooks\clf.joblib')

# Assuming you have new data stored in a DataFrame called 'new_data'
# Extract the relevant features from the new data
    X_new = df[['count_5days', 'sum_5days']]

# Get the predictions and anomaly scores for the new data
    y_pred_new = loaded_model.predict(X_new)
    y_scores_new = loaded_model.decision_function(X_new)


# Now you can use y_pred_new and y_scores_new for further analysis or visualization
    df['y_pred'] = y_pred_new
    df['y_scores'] = y_scores_new

    df_new = df

    return df_new


df_results = predict(df_new)
print(df_results)