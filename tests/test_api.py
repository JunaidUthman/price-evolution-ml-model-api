import os
import unittest
from unittest.mock import patch, MagicMock

# Set required settings before imports
os.environ["HF_TOKEN"] = "fake-token"
os.environ["HF_REPO_ID"] = "fake-repo/fake-model"
os.environ["MODEL_FILENAME"] = "model.pkl"

import numpy as np

# Mock the underlying XGBoost model
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1])
mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])

# Patch huggingface_hub and joblib before model_service is loaded
with patch("huggingface_hub.hf_hub_download", return_value="/fake/path"), \
     patch("joblib.load", return_value=mock_model):
    from fastapi.testclient import TestClient
    from main import app
    from app.services.model_service import model_service
    model_service.model = mock_model

class TestProductPriceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs", data)
        self.assertIn("health", data)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_predict_success(self):
        payload = {
            "prix_usd": 150.0,
            "prix_lag_1": 145.0,
            "prix_lag_3": 140.0,
            "prix_lag_7": 135.0,
            "volatilite_7j": 5.0
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["trend_id"], 1)
        self.assertEqual(data["trend_label"], "STABLE")
        self.assertEqual(data["probabilities"], [0.1, 0.8, 0.1])
        self.assertEqual(data["model_version"], "fake-repo/fake-model")

    def test_predict_validation_error(self):
        # Missing required fields should return 422 Unprocessable Entity
        payload = {"prix_usd": 150.0}
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)

if __name__ == '__main__':
    unittest.main()
