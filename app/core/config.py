from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    APP_NAME: str = "Product Price Evolution API"
    HF_TOKEN: str
    HF_REPO_ID: str
    MODEL_FILENAME: str = "xgboost_trend_model.pkl"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
