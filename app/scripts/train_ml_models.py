import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib, os

# Diabetes model (Pima Indians dataset — download from UCI)
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric="logloss"))
])
pipeline.fit(X_train, y_train)

os.makedirs("app/ml/models", exist_ok=True)
joblib.dump(pipeline, "app/ml/models/diabetes_model.joblib")
joblib.dump(X.columns.tolist(), "app/ml/models/diabetes_features.joblib")

print(f"Accuracy: {pipeline.score(X_test, y_test):.2f}")