import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

# Load dataset
df = pd.read_csv("data/features.csv")

print("\n✅ Dataset loaded. Total rows:", len(df))

# Select input features and labels
X = df[["cx_norm", "cy_norm", "size_norm", "face_visible"]]
y = df["label"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (very important for ML)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("\n✅ CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("model", exist_ok=True)
dump(model, "model/cheat_model.joblib")
dump(scaler, "model/scaler.joblib")

print("\n✅ Model saved to: model/cheat_model.joblib")
print("✅ Scaler saved to: model/scaler.joblib")
