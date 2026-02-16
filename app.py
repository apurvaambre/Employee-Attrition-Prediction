import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Employee Attrition Prediction Dashboard", layout="wide")
st.title("👩‍💼 Employee Attrition Prediction Dashboard")
st.markdown("Explore model results and predict employee attrition with the best-performing model.")

# --- Load Models and Files ---
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

scaler = joblib.load("models/scaler.pkl")
best_model = joblib.load("models/best_model.pkl")

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "🏆 Feature Importance"])

# =========================================
# TAB 1: Prediction
# =========================================
with tab1:
    st.header("🔮 Predict Employee Attrition")
    st.markdown("### Enter Employee Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=65, step=1)
        BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        DistanceFromHome = st.number_input("Distance From Home (km)", min_value=0, max_value=100, step=1)
        Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])

    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                            "Manufacturing Director", "Healthcare Representative", 
                                            "Manager", "Sales Representative", "Human Resources", "Research Director"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=30000, step=100)
        OverTime = st.selectbox("OverTime", ["Yes", "No"])

    with col3:
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=40, step=1)
        YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, step=1)
        JobSatisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
        WorkLifeBalance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4])
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])

    # Combine inputs
    input_dict = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "OverTime": OverTime,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "JobSatisfaction": JobSatisfaction,
        "WorkLifeBalance": WorkLifeBalance,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns from training
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_names]
    input_scaled = scaler.transform(input_encoded)

    if st.button("Predict Attrition", type="primary"):
        pred = best_model.predict(input_scaled)

        # --- Determine correct probability mapping automatically ---
        proba = None
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(input_scaled)[0]
            classes = list(best_model.classes_)

            # Try to detect which class corresponds to "leaving"
            leave_labels = [1, "Yes", "Attrition", "Leave", "True"]
            stay_labels = [0, "No", "Stay", "False"]

            leave_idx = None
            for lbl in leave_labels:
                if lbl in classes:
                    leave_idx = classes.index(lbl)
                    break

            if leave_idx is None:
                # fallback: assume the class with the higher mean probability
                # in training was the positive class, so index 1
                leave_idx = 1 if len(classes) > 1 else 0

            proba = probs[leave_idx]

        # --- Show results correctly ---
        st.markdown("### 🧭 Prediction Result")

        if pred[0] in [1, "Yes", "Attrition", "Leave", "True"]:
            if proba is not None:
                st.error(f"⚠️ The employee is **likely to leave**. Attrition Probability: **{proba:.2f}**")
            else:
                st.error("⚠️ The employee is **likely to leave**.")
        else:
            if proba is not None:
                st.success(f"✅ The employee is **likely to stay**. Retention Probability: **{1 - proba:.2f}**")
            else:
                st.success("✅ The employee is **likely to stay**.")



# =========================================
# TAB 2: Analysis
# =========================================
with tab2:
    st.header("📊 Model Analysis")

    # --- Accuracy Comparison ---
    if os.path.exists("models/model_performance.csv"):
        perf_df = pd.read_csv("models/model_performance.csv")
        fig = px.bar(
            perf_df,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Model Accuracy Comparison"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(yaxis=dict(range=[0, 1]), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Confusion Matrices ---
    st.subheader("🔹 Confusion Matrices")
    cols = st.columns(2)
    confusion_files = [
        ("Logistic Regression", "models/Logistic_Regression_confusion_matrix.png"),
        ("Random Forest", "models/Random_Forest_confusion_matrix.png"),
        ("XGBoost", "models/XGBoost_confusion_matrix.png"),
        ("SVM", "models/SVM_confusion_matrix.png")
    ]
    for i, (name, path) in enumerate(confusion_files):
        if os.path.exists(path):
            with cols[i % 2]:
                st.subheader(name)
                st.image(Image.open(path), caption=f"{name} Confusion Matrix", use_container_width=True)

    # --- ROC & AUC Curves ---
    st.subheader("📈 ROC & AUC Curves")
    cols = st.columns(2)
    roc_files = [
        ("Logistic Regression", "models/Logistic_Regression_roc_curve.png"),
        ("Random Forest", "models/Random_Forest_roc_curve.png"),
        ("XGBoost", "models/XGBoost_roc_curve.png"),
        ("SVM", "models/SVM_roc_curve.png")
    ]
    for i, (name, path) in enumerate(roc_files):
        if os.path.exists(path):
            with cols[i % 2]:
                st.subheader(name)
                st.image(Image.open(path), caption=f"{name} ROC Curve", use_container_width=True)

# =========================================
# TAB 3: Feature Importance
# =========================================
with tab3:
    st.header("🏆 Key Features Affecting Attrition")

    if hasattr(best_model, "feature_importances_"):
        # Tree-based models
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": best_model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="coolwarm", ax=ax, legend=False)
        ax.set_title("Top 10 Most Important Features (Tree-Based Importance)", fontsize=13, fontweight="bold")
        st.pyplot(fig)

    elif hasattr(best_model, "coef_"):
        # Linear models
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(best_model.coef_[0])
        }).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="mako", ax=ax, legend=False)
        ax.set_title("Top 10 Most Important Features (Model Coefficients)", fontsize=13, fontweight="bold")
        st.pyplot(fig)

    else:
        # Permutation Importance
        from sklearn.inspection import permutation_importance
        st.subheader("🔍 Calculating Permutation Feature Importance...")
        with st.spinner("This might take a few seconds..."):
            sample_df = pd.DataFrame(input_encoded).sample(n=min(200, len(input_encoded)), replace=True)
            sample_scaled = scaler.transform(sample_df)
            result = permutation_importance(best_model, sample_scaled,
                                            best_model.predict(sample_scaled),
                                            n_repeats=5, random_state=42)
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": result.importances_mean
            }).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="viridis", ax=ax, legend=False)
        ax.set_title("Top 10 Most Important Features (Permutation Importance)", fontsize=13, fontweight="bold")
        st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Apurva Ambre")