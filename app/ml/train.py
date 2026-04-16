import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import joblib
import os

MODEL_DIR = "app/ml/models"

def train_diabetes_model():
    print("Training diabetes risk model...")
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.2f}")
    print(f"  AUC      : {roc_auc_score(y_test, y_proba):.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, f"{MODEL_DIR}/diabetes_model.joblib")
    joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/diabetes_features.joblib")
    print("  Saved to app/ml/models/diabetes_model.joblib")

if __name__ == "__main__":
    train_diabetes_model()