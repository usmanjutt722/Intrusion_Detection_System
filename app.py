# --- Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# --- Load Trained Models ---
rf_model = joblib.load("best_rf.pkl")
xgb_model = joblib.load("best_xgb.pkl")
fnn_model = load_model("best_fnn.keras")
scaler = joblib.load("scaler.pkl")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Intrusion Detection", layout="wide")
st.title("🔐 AI-Powered Intrusion Detection System (IDS)")

model_choice = st.sidebar.selectbox("📊 Select the ML Model:", ["RF & XGB Together", "FNN Separately"])
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV test data", type=["csv"])
st.sidebar.info("Expected: 1 row = 1 network session")

# --- Helper Function for Results and Plots ---
def display_results(data, predictions, model_name):
    result_df = data.copy()
    result_df["Prediction"] = predictions

    st.subheader(f"📄 {model_name} Prediction Results")
    styled_df = result_df.style.applymap(lambda x: "background-color: red" if x == 1 else "", subset=["Prediction"])
    st.dataframe(styled_df)

        # --- Class Distribution Plot (Safe for 1-class predictions) ---
    class_counts = pd.Series(predictions).value_counts().sort_index()
    class_dict = {0: 0, 1: 0}
    for label in class_counts.index:
        class_dict[label] = class_counts[label]

    fig_bar = px.bar(
        x=["Normal", "Attack"],
        y=[class_dict[0], class_dict[1]],
        labels={"x": "Class", "y": "Count"},
        title=f"{model_name} - Prediction Class Distribution",
        color_discrete_sequence=["#c38c00", "#ff7e4b"]
    )
    st.plotly_chart(fig_bar)


    # --- Confusion Matrix ---
    if "Label" in data.columns:
        y_true = data["Label"].values
        cm = confusion_matrix(y_true, predictions)
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Normal", "Attack"],
            y=["Normal", "Attack"],
            colorscale= "viridis",
            showscale=True
        )
        fig_cm.update_layout(title=f"{model_name} - Confusion Matrix")
        st.plotly_chart(fig_cm)

# --- Main Logic ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data.drop(columns=["Label"], errors="ignore")

    if model_choice == "RF & XGB Together":
        expected_features = rf_model.feature_names_in_
        if not set(expected_features).issubset(X.columns):
            st.error("❌ Uploaded data is missing required RF/XGB features.")
        else:
            X_selected = X[expected_features]

            rf_pred = rf_model.predict(X_selected)
            display_results(data, rf_pred, "Random Forest")

            xgb_pred = xgb_model.predict(X_selected)
            display_results(data, xgb_pred, "XGBoost")

    elif model_choice == "FNN Separately":
        if scaler is None:
            st.error("❌ Scaler not found. Required for FNN.")
        elif X.shape[1] != scaler.mean_.shape[0]:
            st.error("❌ FNN expects data with different number of features.")
        else:
            X_scaled = scaler.transform(X)
            fnn_prob = fnn_model.predict(X_scaled)
            fnn_pred = (fnn_prob >= 0.5).astype(int).flatten()
            display_results(data, fnn_pred, "FNN")

else:
    st.info("📌 Please upload a valid CSV test file to continue.")
