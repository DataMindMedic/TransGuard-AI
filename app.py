import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Paths
model_path = "/Users/michealomotosho/Documents/EDUCATION DOCUMENTS/DATA SCIENCE SELF PROJECT/TransGuard-AI/models/random_forest_model.pkl"
data_path = "/Users/michealomotosho/Documents/EDUCATION DOCUMENTS/DATA SCIENCE SELF PROJECT/TransGuard-AI/data/Processed/transactions_cleaned.csv"

# Load model
model = joblib.load(model_path)

# Load processed data
data = pd.read_csv(data_path)

# Drop target column
feature_cols = data.drop('Class', axis=1).columns.tolist()

st.title("TransGuard AI - Real-Time Fraud Detection")

st.write("Simulating real-time incoming transactions...")

num_samples = st.slider("Number of new transactions to generate", 1, 1000, 200)

# Sample and reset index
sampled = data.sample(n=num_samples, replace=True, random_state=None).reset_index(drop=True)

# Predict on sampled transactions
preds = model.predict(sampled[feature_cols])
probs = model.predict_proba(sampled[feature_cols])[:, 1]

# Add predictions to the dataframe
sampled['Fraud Prediction'] = preds
sampled['Fraud Probability'] = probs

# Option 1 — add emoji for clarity
sampled['Fraud Flag'] = np.where(sampled['Fraud Prediction']==1, '🚨 FRAUD', '✅ Legit')

# Show selected columns
columns_to_show = ['Fraud Flag', 'Fraud Probability'] + feature_cols

st.write("## Live Transaction Feed")

# Simpler display — no styling
st.dataframe(sampled[columns_to_show], use_container_width=True)
