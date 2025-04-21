import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# ------------------------- Load Dataset -------------------------
print("📂 Loading dataset...")
df = pd.read_csv("creditcard.csv")

# ------------------------- Preprocess -------------------------
print("🧼 Preprocessing data...")
X = df.drop("Class", axis=1)
y = df["Class"]

# ------------------------- Train-Test Split -------------------------
print("✂️ Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------- SMOTE Oversampling -------------------------
print("🧪 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ------------------------- Feature Scaling -------------------------
print("🔧 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ------------------------- Train Model -------------------------
print("🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_resampled)

# ------------------------- Make Predictions -------------------------
print("🔍 Making predictions...")
y_probs = model.predict_proba(X_test_scaled)[:, 1]
risk_scores = np.round(y_probs * 100, 2)
risk_flags = (risk_scores > 30).astype(int)

# ------------------------- Assign Transaction IDs -------------------------
print("🆔 Assigning Transaction IDs...")
X_test_reset = X_test.reset_index(drop=True)
X_test_reset["TransactionID"] = [f"TX{i:06d}" for i in range(len(X_test_reset))]

# ------------------------- Prepare Output Data -------------------------
print("📊 Preparing results...")
results = pd.DataFrame({
    "TransactionID": X_test_reset["TransactionID"],
    "ActualClass": y_test.reset_index(drop=True),
    "RiskScore(%)": risk_scores,
    "Flagged": np.where(risk_flags == 1, "Yes", "No")
})

# ------------------------- Save Risky Transactions -------------------------
print("💾 Saving flagged risky transactions...")
risky = results[results["Flagged"] == "Yes"]
risky.to_csv("flagged_transactions.csv", index=False)

# ------------------------- Print Sample -------------------------
print("\n🔎 Sample Predictions:")
print(results.head(10).to_string(index=False))

# ------------------------- SHAP Explainability -------------------------
print("\n🧠 Explaining Model Predictions with SHAP (using summary)...")

# Use a small subset to keep SHAP fast
shap_subset = X_test_scaled[:10]
shap_feature_names = X.columns

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_subset)

# Choose SHAP values for the positive class if available
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ------------------------- SHAP Summary Plot -------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_subset, feature_names=shap_feature_names, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Top Predictors)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()

print("📈 SHAP summary plot saved as 'shap_summary_plot.png'")

# ------------------------- Accuracy and Confusion Matrix -------------------------
print("\n📊 Model Performance Evaluation:")

# Accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ------------------------- Classification Report -------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------- ROC AUC Score -------------------------
roc_auc = roc_auc_score(y_test, y_probs)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# ------------------------- Summary -------------------------
print("\n✅ Process completed successfully.")