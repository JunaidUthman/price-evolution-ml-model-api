import os
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from app.core.config import get_settings
from app.schemas.prediction import PriceFeatures
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class ModelService:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.labels = {0: "BAISSE", 1: "STABLE", 2: "HAUSSE"}

    def load_model(self):
        """Downloads the model from HF and loads it into memory."""
        try:
            logger.info(f"Downloading model from {settings.HF_REPO_ID}...")
            self.model_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.MODEL_FILENAME,
                token=settings.HF_TOKEN
            )
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load model: {e}")

    def predict(self, features: PriceFeatures):
        if self.model is None:
            self.load_model()
        
        # Prepare data for prediction
        data = pd.DataFrame([features.dict()])
        
        # XGBoost prediction
        probabilities = self.model.predict_proba(data)[0].tolist()
        trend_id = int(self.model.predict(data)[0])
        
        return {
            "trend_id": trend_id,
            "trend_label": self.labels[trend_id],
            "probabilities": probabilities,
            "model_version": settings.HF_REPO_ID
        }

model_service = ModelService()
model_service.load_model()
