import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statistics as stat
from scipy.stats import kurtosis, skew
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
    return df, stats

# Extract maximum amplitude in the specified timestamp ranges
def extract_amplitude(df, on_start, on_end, off_start, off_end):
    on_operation = df[(df['Timestamp'] >= on_start) & (df['Timestamp'] <= on_end)]['Data']
    off_operation = df[(df['Timestamp'] >= off_start) & (df['Timestamp'] <= off_end)]['Data']
    on_max = on_operation.max()
    off_max = off_operation.max()
    return on_max, off_max

# Calculate operation times
def calculate_operation_times(df, on_start, on_end, off_start, off_end):
    try:
        # For on operation
        on_df = df[(df['Timestamp'] >= on_start) & (df['Timestamp'] <= on_end)]
        if not on_df.empty:
            on_start_indices = on_df[(on_df['Data'] > 200) | (on_df['Data'] < -200)].index
            on_end_indices = on_df[(on_df['Data'] > 2000) | (on_df['Data'] < -2000)].index
            if len(on_start_indices) > 0 and len(on_end_indices) > 0:
                start_index_on = on_start_indices[0]
                end_index_on = on_end_indices[0]
                start_time_on = df.loc[start_index_on, 'Timestamp']
                end_time_on = df.loc[end_index_on, 'Timestamp']
                total_time_seconds_on = (end_time_on - start_time_on) / 1000
            else:
                total_time_seconds_on = np.nan
        else:
            total_time_seconds_on = np.nan

        # For off operation
        off_df = df[(df['Timestamp'] >= off_start) & (df['Timestamp'] <= off_end)]
        if not off_df.empty:
            off_start_indices = off_df[(off_df['Data'] > 200) | (off_df['Data'] < -200)].index
            off_end_indices = off_df[(off_df['Data'] > 2000) | (off_df['Data'] < -2000)].index
            if len(off_start_indices) > 0 and len(off_end_indices) > 0:
                start_index_off = off_start_indices[0]
                end_index_off = off_end_indices[0]
                start_time_off = df.loc[start_index_off, 'Timestamp']
                end_time_off = df.loc[end_index_off, 'Timestamp']
                total_time_seconds_off = (end_time_off - start_time_off) / 1000
            else:
                total_time_seconds_off = np.nan
        else:
            total_time_seconds_off = np.nan

        return total_time_seconds_on, total_time_seconds_off
    except Exception as e:
        st.error(f"An error occurred while calculating operation times: {e}")
        return np.nan, np.nan


# Map predictions to labels, health status, and images
def map_prediction(prediction):
    mapping = {
        'All Springs attached': ("All springs working", "Health - OK", "100%.png"),
        'One Inner spring removed': ("15% spring function loss", "Health - OK", "85%.png"),
        'One Entire set removed': ("30% spring function loss", "Health - Need Maintenance", "70%.png"),
        'One Entire set and Second Inner spring removed': ("45% spring function loss", "Health - Need Maintenance", "55%.png"),
        'Two Entire sets removed': ("60% spring function loss", "Health - Need Maintenance", "40%orless.png")
    }
    return mapping[prediction]

# Streamlit app layout
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    .title {color: #1f77b4; font-size: 36px; font-weight: bold;}
    .subtitle {color: #ff7f0e; font-size: 24px; font-weight: bold;}
    .metric {font-size: 18px; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">MCCB Mechanism Health Prediction</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df, data = process_csv(uploaded_file)

    model = load_model()
    prediction = model.predict(data)[0]  # Get the first prediction

    st.markdown('<div class="subtitle">MCCB Health:</div>', unsafe_allow_html=True)
    st.markdown("---")
    condition, health_status, image_path = map_prediction(prediction)
    st.write(f"Mechanism Condition: {condition}")
    st.write(f"Health Status: {health_status}")
    st.image(image_path, caption='Spring Function')

    # Extract and display maximum amplitude for on and off operations
    on_max, off_max = extract_amplitude(df, 4000, 6000, 44000, 46000)
    st.markdown('<div class="subtitle">For ON Operation:</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.write(f"Vibration Amplitude: {on_max} mV")
    # Calculate and display operation times for on operations
    total_time_seconds_on, total_time_seconds_off = calculate_operation_times(df, 5000, 6000, 45100, 46000)
    total_time_seconds_on = round(total_time_seconds_on, 4)
    total_time_seconds_off = round(total_time_seconds_off, 4)
    if not np.isnan(total_time_seconds_on):
        st.write(f"Total Operation Time: {total_time_seconds_on} s")
    else:
        st.write("Probably fault in the data or faulty file.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">For OFF Operation:</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.write(f"Vibration Amplitude: {off_max} mV")
    # Display operation times for off operations
    if not np.isnan(total_time_seconds_off):
        st.write(f"Total Operation Time: {total_time_seconds_off} s")
    else:
        st.write("Probably fault in the data or faulty file.")
    st.markdown('</div>', unsafe_allow_html=True)
