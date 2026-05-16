from pydantic import BaseModel, Field
from typing import List, Optional

class PriceFeatures(BaseModel):
    prix_usd: float = Field(..., description="Current price in USD")
    prix_lag_1: float = Field(..., description="Price 1 period ago")
    prix_lag_3: float = Field(..., description="Price 3 periods ago")
    prix_lag_7: float = Field(..., description="Price 7 periods ago")
    volatilite_7j: float = Field(..., description="Price volatility over the last 7 periods")

class PredictionResponse(BaseModel):
    trend_id: int = Field(..., description="0: BAISSE, 1: STABLE, 2: HAUSSE")
    trend_label: str = Field(..., description="Label of the predicted trend")
    probabilities: List[float] = Field(..., description="Probabilities for each class [BAISSE, STABLE, HAUSSE]")
    model_version: str = Field(..., description="Model repository ID used for prediction")
