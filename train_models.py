import json
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, classification_report
)
import math

# --- Load Dataset ---
df = pd.read_csv('data/employee_attrition.csv', delimiter = '\t')  # Change to your dataset path

# --- Drop unnecessary columns ---
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

# --- Encode categorical variables ---
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Define features and target ---
selected_features = [
    'Age',
    'DistanceFromHome',
    'MonthlyIncome',
    'OverTime',
    'JobSatisfaction',
    'WorkLifeBalance',
    'Education',
    'TotalWorkingYears',
    'YearsAtCompany',
    'JobLevel',
    'Gender',
    'MaritalStatus'
]

X = df[selected_features]

y = df['Attrition']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# Save the training feature names
feature_names = list(X.columns)
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

# --- Define models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

accuracy_scores = {}
conf_matrices = {}

# --- Train & Evaluate ---
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # --- Calculate metrics ---
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    # Error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    # Store metrics
    accuracy_scores[name] = acc
    conf_matrices[name] = cm

    # Print neatly
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # acc = accuracy_score(y_test, y_pred)
    # accuracy_scores[name] = acc
    # conf_matrices[name] = confusion_matrix(y_test, y_pred)

    # print(f"Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrices[name], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"models/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # --- ROC & AUC ---
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1
    else:
        try:
            y_proba = model.decision_function(X_test_scaled)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # normalize
        except:
            y_proba = None

    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        # Plot ROC Curve
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"models/{name.replace(' ', '_')}_roc_curve.png")
        plt.close()

        print(f"AUC Score: {auc_score:.4f}")
    else:
        print("ROC/AUC not available for this model.")


# --- Save accuracy comparison to CSV ---
perf_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])
perf_df.to_csv('models/model_performance.csv', index=False)
print("\n✅ Model performance saved to models/model_performance.csv")

# --- Plot Comparison Chart ---
plt.figure(figsize=(7, 5))
sns.barplot(data=perf_df, x='Model', y='Accuracy', hue='Model', palette='cool', legend=False)
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
for i, v in enumerate(perf_df['Accuracy']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', color='black')
plt.tight_layout()
plt.savefig('models/model_accuracy_comparison.png')
plt.close()
print("✅ Accuracy comparison chart saved in models/")

# --- Save Best Model ---
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
joblib.dump(best_model, f"models/best_model.pkl")

print(f"\n🏆 Best model saved: {best_model_name} with accuracy {accuracy_scores[best_model_name]:.4f}")
print("All confusion matrices and charts saved in 'models/' folder.")