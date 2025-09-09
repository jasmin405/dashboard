# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="EEG Cognitive Load Dashboard", layout="wide")
st.title("EEG Cognitive Load Dashboard")

# --- Step 1: Load CSV from GitHub repo ---
csv_file = "EEG_Eye_State_Classification.csv"  # replace with your GitHub raw file link if needed
df = pd.read_csv(csv_file)

st.write("### Raw EEG Data")
st.dataframe(df.head())

# --- Step 2: Compute a simple cognitive load score ---
# Example: average of all numeric columns (replace with actual EEG metrics)
numeric_cols = df.select_dtypes(include=np.number).columns
df['EEG_Score'] = df[numeric_cols].mean(axis=1)
avg_score = df['EEG_Score'].mean()

# Determine cognitive load level
if avg_score < 30:
    load_level = 'Low'
elif avg_score < 70:
    load_level = 'Medium'
else:
    load_level = 'High'

st.write(f"### Estimated Cognitive Load: **{load_level}**")

# --- Step 3: Visualizations based on load ---
st.write("### Visualizations")

# --- Low Load: Basic line plots ---
if load_level == 'Low':
    st.write("#### Basic EEG Line Plot")
    for col in numeric_cols:
        st.line_chart(df[col])

# --- Medium Load: Line + Histogram + Scatter ---
elif load_level == 'Medium':
    st.write("#### EEG Line Plots")
    for col in numeric_cols:
        st.line_chart(df[col])

    st.write("#### Histograms")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("#### Scatter Plot Example")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=ax)
        st.pyplot(fig)

# --- High Load: Full Analysis ---
else:
    st.write("#### EEG Line Plots")
    for col in numeric_cols:
        st.line_chart(df[col])
    
    st.write("#### Histograms")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("#### Scatter Plot Matrix")
    fig = sns.pairplot(df[numeric_cols])
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

