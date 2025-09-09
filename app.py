import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("EEG_Eye_State_Classification.csv")

# Preprocess: extract features and define thresholds
X = df.drop(columns=["eyeDetection"])
df["EEG_mean"] = X.mean(axis=1)

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

# Train Random Forest
X = df.drop(columns=["eyeDetection", "Load_Level"])
y = df["Load_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict current load (last row for simulation)
current_features = X.iloc[-1]
pred_load = clf.predict([current_features])[0]

# Streamlit Dashboard
st.title("🧠 Cognitive Load-Aware Dashboard")
st.sidebar.write(f"### Current Cognitive Load: {pred_load}")

if pred_load == "Low":
    st.subheader("📊 Detailed Dashboard (Low Load)")
    st.plotly_chart(px.line(df.head(200), y="EEG_mean", title="EEG Mean Activity"))
    st.plotly_chart(px.histogram(df, x="EEG_mean", title="Distribution of EEG"))
    st.plotly_chart(px.scatter(df, x="AF3", y="F7", color="Load_Level", title="AF3 vs F7"))

elif pred_load == "Medium":
    st.subheader("📈 Moderate Dashboard (Medium Load)")
    st.plotly_chart(px.line(df.head(200), y="EEG_mean", title="EEG Mean Activity"))
    st.plotly_chart(px.box(df, y="EEG_mean", title="EEG Activity Summary"))

else:
    st.subheader("✅ Simplified Dashboard (High Load)")
    st.metric("Average EEG", round(df["EEG_mean"].mean(), 2))
    st.metric("High Load Samples", (df["Load_Level"]=="High").sum())
