from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PriceFeatures, PredictionResponse
from app.services.model_service import model_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_trend(features: PriceFeatures):
    """
    Predict the price trend (BAISSE, STABLE, HAUSSE) based on input features.
    """
    try:
        prediction = model_service.predict(features)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_service.model is not None}
