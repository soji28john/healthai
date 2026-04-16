import joblib, shap, numpy as np
from pathlib import Path

MODEL_DIR = Path("app/ml/models")

def predict_diabetes_risk(features: dict) -> dict:
    pipeline = joblib.load(MODEL_DIR / "diabetes_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "diabetes_features.joblib")

    X = np.array([[features.get(f, 0) for f in feature_names]])

    proba = pipeline.predict_proba(X)[0][1]  # probability of positive class

    # SHAP explanation
    try:
        model = pipeline.named_steps["model"]
        scaler = pipeline.named_steps["scaler"]

        X_scaled = scaler.transform(X)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)
    
        values = shap_values.values[0]
    
        top_factors = sorted(
            zip(feature_names, values),
            key=lambda x: abs(x[1]),
            reverse=True
    )[:3]

        explanation = "Your risk score is mainly influenced by: " + ", ".join(
            f"{name} ({'increases' if val > 0 else 'decreases'} risk)"
            for name, val in top_factors
    )
    except Exception:
        explanation = "Risk influenced by multiple health factors."
    return {
        "risk_score": round(float(proba), 3),
        "risk_level": "HIGH" if proba > 0.7 else "MODERATE" if proba > 0.4 else "LOW",
        "explanation": explanation,
        "disclaimer": "This is a statistical estimate, not a diagnosis. Consult your doctor."
    }