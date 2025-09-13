# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "malicious_login_lr.joblib"

def generate_synthetic_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    failed_attempts = np.random.poisson(1, n_samples)
    time_of_login = np.random.randint(0, 24, n_samples)
    geo_distance = np.random.exponential(200, n_samples)
    device_known = np.random.choice([0,1], n_samples, p=[0.3,0.7])

    suspicious_time = ((time_of_login < 6) | (time_of_login > 22)).astype(int)
    long_distance = (geo_distance > 1000).astype(int)
    malicious_score = (
        0.25*failed_attempts +
        0.35*suspicious_time +
        0.2*long_distance +
        0.4*(device_known == 0)
    )

    prob = 1 / (1 + np.exp(- (malicious_score - 0.5)))
    labels = (prob > np.random.rand(n_samples)).astype(int)

    df = pd.DataFrame({
        "failed_attempts": failed_attempts,
        "time_of_login": time_of_login,
        "geo_distance": geo_distance,
        "device_known": device_known,
        "malicious": labels
    })
    return df

def train_and_save_model(df, path=MODEL_PATH):
    X = df[["failed_attempts","time_of_login","geo_distance","device_known"]]
    y = df["malicious"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    joblib.dump((model, X_train, X_test, y_train, y_test), path)
    return model, X_train, X_test, y_train, y_test

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        model, X_train, X_test, y_train, y_test = joblib.load(path)
        return model, X_train, X_test, y_train, y_test
    return None

def predict(model, features_df):
    prob = model.predict_proba(features_df)[0,1]
    pred = int(prob >= 0.5)
    return prob, pred

def feature_contributions(model, feature_values):
    coefs = model.coef_[0]
    contributions = coefs * feature_values
    cont_df = pd.DataFrame({
        "feature": ["failed_attempts","time_of_login","geo_distance","device_known"],
        "value": feature_values,
        "coef": coefs,
        "contribution": contributions
    }).sort_values(by="contribution", key=abs, ascending=False)
    return cont_df

st.set_page_config(page_title="Malicious Login Detector", layout="centered")
st.title("🔐 Malicious Login Detector — Logistic Regression Demo")
st.write("Interactive demo: enter login details to check if an attempt is likely malicious.")

model_bundle = load_model()
if model_bundle is None:
    with st.spinner("Training model on synthetic dataset (this runs only once)..."):
        df = generate_synthetic_data(5000)
        model, X_train, X_test, y_train, y_test = train_and_save_model(df)
    st.success("Model trained and saved.")
else:
    model, X_train, X_test, y_train, y_test = model_bundle

if st.checkbox("Show sample training data / EDA"):
    df = generate_synthetic_data(1000)
    st.dataframe(df.head(10))
    st.write("Distribution of labels:")
    fig, ax = plt.subplots()
    sns.countplot(x="malicious", data=df, ax=ax)
    ax.set_xticklabels(["Legit (0)", "Malicious (1)"])
    st.pyplot(fig)

st.subheader("Input login attempt")
col1, col2 = st.columns(2)
with col1:
    failed_attempts = st.number_input("Failed attempts before success", min_value=0, max_value=50, value=0, step=1)
    time_of_login = st.slider("Login hour (0-23)", 0, 23, 14)
with col2:
    geo_distance = st.number_input("Geo distance from last known location (km)", min_value=0.0, max_value=20000.0, value=10.0, step=1.0, format="%.1f")
    device_known = st.selectbox("Device known?", options=["Yes","No"])
    device_known = 1 if device_known=="Yes" else 0

if st.button("Check login"):
    input_df = pd.DataFrame([{
        "failed_attempts": failed_attempts,
        "time_of_login": time_of_login,
        "geo_distance": geo_distance,
        "device_known": device_known
    }])
    prob, pred = predict(model, input_df)
    st.metric(label="Malicious probability", value=f"{prob:.2f}", delta=None)
    if pred == 1:
        st.error("🚨 Predicted: Malicious login")
    else:
        st.success("✅ Predicted: Legitimate login")

    cont_df = feature_contributions(model, input_df.iloc[0].values)
    st.write("Approximate feature contributions (coef * value). Positive → more malicious:")
    st.table(cont_df[["feature","value","coef","contribution"]].assign(contribution=lambda x: x["contribution"].round(3)))

    st.write("**Threshold tuning**: try different thresholds if you want to prioritize fewer false negatives or fewer false positives.")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5)
    custom_pred = int(prob >= thr)
    st.write(f"With threshold={thr:.2f} → predicted class = {custom_pred}")

if st.checkbox("Show model evaluation (on internal test split)"):
    yhat = model.predict(X_test)
    st.write("Classification report:")
    st.text(classification_report(y_test, yhat))
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, yhat), annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

st.markdown("---")
st.markdown("**Notes / next steps:**\n\n"
            "- Replace synthetic data with real login logs for production.\n"
            "- Add monitoring, drift detection, and SHAP explainability for interpretability.\n"
            "- Consider session-based models for sequence-based detection.")
