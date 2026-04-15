# tests/test_ml_predict.py
import pytest
from app.ml.predict import predict_diabetes_risk
from unittest.mock import patch, MagicMock
import numpy as np

SAMPLE_FEATURES = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
}

class TestMLPredict:
    def test_output_has_required_keys(self):
        with patch("app.ml.predict.joblib") as mock_joblib:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_pipeline.named_steps = {
                "scaler": MagicMock(transform=lambda x: x),
                "model": MagicMock()
            }
            mock_joblib.load.return_value = mock_pipeline

            result = predict_diabetes_risk(SAMPLE_FEATURES)

            assert "risk_score" in result
            assert "risk_level" in result
            assert "explanation" in result
            assert "disclaimer" in result

    def test_risk_score_is_between_0_and_1(self):
        with patch("app.ml.predict.joblib") as mock_joblib:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = np.array([[0.4, 0.6]])
            mock_pipeline.named_steps = {
                "scaler": MagicMock(transform=lambda x: x),
                "model": MagicMock()
            }
            mock_joblib.load.return_value = mock_pipeline

            result = predict_diabetes_risk(SAMPLE_FEATURES)
            assert 0.0 <= result["risk_score"] <= 1.0

    def test_high_probability_maps_to_high_risk_level(self):
        with patch("app.ml.predict.joblib") as mock_joblib:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
            mock_pipeline.named_steps = {
                "scaler": MagicMock(transform=lambda x: x),
                "model": MagicMock()
            }
            mock_joblib.load.return_value = mock_pipeline

            result = predict_diabetes_risk(SAMPLE_FEATURES)
            assert result["risk_level"] == "HIGH"

    def test_disclaimer_always_present(self):
        with patch("app.ml.predict.joblib") as mock_joblib:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
            mock_pipeline.named_steps = {
                "scaler": MagicMock(transform=lambda x: x),
                "model": MagicMock()
            }
            mock_joblib.load.return_value = mock_pipeline

            result = predict_diabetes_risk(SAMPLE_FEATURES)
            assert "not a diagnosis" in result["disclaimer"].lower()