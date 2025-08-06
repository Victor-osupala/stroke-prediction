# ===========================
# Stroke Prediction Training
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
import joblib
import os

# ==== Step 1: Load Dataset ====
df = pd.read_csv("stroke_dataset_10000.csv")

# ==== Step 2: Preprocessing ====
# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'stroke':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split features and label
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ==== Step 3: Feature Selection (Chi-Square) ====
selector = SelectKBest(score_func=chi2, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# ==== Step 4: Feature Scaling ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ==== Step 5: Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ==== Step 6: Train MLP Classifier ====
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# ==== Step 7: Evaluation ====
y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)
report_dict = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# ==== Step 8: Save Model and Components ====
os.makedirs("models", exist_ok=True)

joblib.dump(mlp, "models/mlp_stroke_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(selector, "models/chi2_selector.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(selected_features.tolist(), "models/selected_features.pkl")

# ==== Step 9: Save Evaluation Report as Text ====
with open("models/evaluation_report.txt", "w") as f:
    f.write("==== STROKE PREDICTION MODEL EVALUATION ====\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(report)
    f.write("\n=== Confusion Matrix ===\n")
    f.write(np.array2string(conf_matrix))

print("✅ Model and evaluation text report saved.")

# ==== Step 10: Save Evaluation Plots ====
# Confusion Matrix Plot
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay(conf_matrix).plot(ax=ax_cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.close()

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig_roc = plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("models/roc_curve.png")
plt.close()

print("📊 Evaluation images (Confusion Matrix & ROC Curve) saved to 'models/' folder.")
