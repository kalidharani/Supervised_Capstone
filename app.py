import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn & Revenue Prediction",
    page_icon="📊",
    layout="centered"
)

# -------------------------------
# Load Models & Scalers
# -------------------------------
clf_model = joblib.load("churn_classification_model.pkl")
clf_scaler = joblib.load("classification_scaler.pkl")

reg_model = joblib.load("monthly_charges_regression_model.pkl")
reg_scaler = joblib.load("regression_scaler.pkl")

# -------------------------------
# App Title
# -------------------------------
st.title("📊 Customer Churn & Revenue Prediction System")

st.write(
    "This app predicts whether a telecom customer is likely to churn "
    "and estimates their monthly charges using machine learning models."
)

st.markdown("---")

# -------------------------------
# Helper
# -------------------------------
def yn(val):
    return 1 if val == "Yes" else 0

# -------------------------------
# User Inputs (Meaningful)
# -------------------------------
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)

phone = st.selectbox("Phone Service", ["No", "Yes"])
multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# -------------------------------
# FULL INPUT DATA (MASTER)
# -------------------------------
full_input = pd.DataFrame([{
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': yn(senior),
    'Partner': yn(partner),
    'Dependents': yn(dependents),
    'tenure': tenure,
    'PhoneService': yn(phone),
    'MultipleLines': ["No", "Yes", "No phone service"].index(multiple),
    'InternetService': ["DSL", "Fiber optic", "No"].index(internet),
    'OnlineSecurity': ["No", "Yes", "No internet service"].index(security),
    'OnlineBackup': ["No", "Yes", "No internet service"].index(backup),
    'DeviceProtection': ["No", "Yes", "No internet service"].index(device),
    'TechSupport': ["No", "Yes", "No internet service"].index(support),
    'StreamingTV': ["No", "Yes", "No internet service"].index(tv),
    'StreamingMovies': ["No", "Yes", "No internet service"].index(movies),
    'Contract': ["Month-to-month", "One year", "Two year"].index(contract),
    'PaperlessBilling': yn(paperless),
    'PaymentMethod': [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ].index(payment),
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': 0   # dummy placeholder (required for regression scaler)
}])

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):

    # ----- Classification -----
    X_clf = full_input.drop(columns=['Churn'])
    X_clf_scaled = clf_scaler.transform(X_clf)
    churn_pred = clf_model.predict(X_clf_scaled)[0]

    # ----- Regression -----
    X_reg = full_input.drop(columns=['MonthlyCharges'])
    X_reg_scaled = reg_scaler.transform(X_reg)
    revenue_pred = reg_model.predict(X_reg_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Results")

    if churn_pred == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.info(f"💰 Estimated Monthly Charges: ₹ {revenue_pred:.2f}")
