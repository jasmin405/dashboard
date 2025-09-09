import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1️⃣ Load EEG CSV
# ---------------------------
DATA_PATH = "EEG_Eye_State_Classification.csv"  # CSV already in repo
df = pd.read_csv(DATA_PATH)

# ---------------------------
# 2️⃣ Feature Extraction & Cognitive Load Classification
# ---------------------------
# Use all EEG channels except 'eyeDetection'
X_features = df.drop(columns=["eyeDetection"])
df["EEG_mean"] = X_features.mean(axis=1)

# Define thresholds for Low / Medium / High
low_thr = df["EEG_mean"].quantile(0.33)
high_thr = df["EEG_mean"].quantile(0.66)

def classify_load(val):
    if val <= low_thr:
        return "Low"
    elif val <= high_thr:
        return "Medium"
    else:
        return "High"

df["Load_Level"] = df["EEG_mean"].apply(classify_load)

# ---------------------------
# 3️⃣ Train Random Forest Model
# ---------------------------
y = df["Load_Level"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_features, y)

# ---------------------------
# 4️⃣ Predict Current Load (last row)
# ---------------------------
current_features = X_features.iloc[-1]  # simulate "current" EEG
pred_load = clf.predict([current_features])[0]

# ---------------------------
# 5️⃣ Streamlit Dashboard
# ---------------------------
st.set_page_config(page_title="Cognitive Load Dashboard", layout="wide")
st.title("🧠 Cognitive Load-Aware Dashboard")
st.sidebar.write(f"### Current Cognitive Load: {pred_load}")

# ---------------------------
# 6️⃣ Adaptive Visualization
# ---------------------------
if pred_load == "Low":
    st.subheader("📊 Detailed Dashboard (Low Load)")
    st.plotly_chart(px.line(df, y="EEG_mean", title="EEG Mean Activity"))
    st.plotly_chart(px.histogram(df, x="EEG_mean", title="Distribution of EEG"))
    # Scatter using first two EEG channels for simplicity
    first_two_cols = X_features.columns[:2]
    st.plotly_chart(px.scatter(df, x=first_two_cols[0], y=first_two_cols[1],
                                color="Load_Level", title=f"{first_two_cols[0]} vs {first_two_cols[1]}"))

elif pred_load == "Medium":
    st.subheader("📈 Moderate Dashboard (Medium Load)")
    st.plotly_chart(px.line(df, y="EEG_mean", title="EEG Mean Activity"))
    st.plotly_chart(px.box(df, y="EEG_mean", title="EEG Activity Summary"))

else:
    st.subheader("✅ Simplified Dashboard (High Load)")
    st.metric("Average EEG", round(df["EEG_mean"].mean(), 2))
    st.metric("High Load Samples", (df["Load_Level"]=="High").sum())

# ---------------------------
# 7️⃣ Optional: Show full EEG data table
# ---------------------------
with st.expander("Show Raw EEG Data"):
    st.dataframe(df)

