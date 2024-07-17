import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statistics as stat
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import os
import pickle


# Function to calculate additional measures
def calculate_additional_measures(data):
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    median_val = data.median()
    mode_val = stat.mode(data)
    range_val = max_val - min_val
    auc_val = np.trapz(data, dx=1)
    std_dev = data.std()
    kurt_val = kurtosis(data)
    skew_val = skew(data)
    bumps = len(np.where(np.diff(np.sign(np.diff(data))) > 0)[0])
    var_val = np.var(data)
    cov_val = np.std(data) / np.mean(data)
    iqr_val = np.percentile(data, 75) - np.percentile(data, 25)

    measures = {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'range': range_val,
        'auc': auc_val,
        'std_dev': std_dev,
        'kurtosis': kurt_val,
        'skewness': skew_val,
        'bumps': bumps,
        'variance': var_val,
        'cov': cov_val,
        'iqr': iqr_val
    }

    return measures

# Load the trained model
@st.cache_data
def load_model():
    with open('mccb_model.pkl', 'rb') as model:
        model = pickle.load(model)
    return model

# Process CSV function
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[:3], axis=1)
    df['Timestamp'] = (df.index + 1) / 20
    data = df[df['Timestamp'] < 50010]['Data']
    stats = calculate_additional_measures(data)
    stats = pd.DataFrame([stats])
    return stats

# Streamlit app layout
st.title("MCCB")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = process_csv(uploaded_file)
    st.write("Data Measures:")
    st.write(data)

    model = load_model()
    prediction = model.predict(data)

    st.write(f"Predicted Condition: {prediction[0]}")
