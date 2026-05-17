# Product Price Evolution API

## Overview
The Product Price Evolution api is a FastAPI-based machine learning service designed to predict future price trends for products. By analyzing a product's current price, its historical price lags, and its recent volatility, the API uses a pre-trained XGBoost classification model to forecast whether the price is likely to decrease (`BAISSE`), remain stable (`STABLE`), or increase (`HAUSSE`) over the next 7 days.

The service automatically downloads the latest trained model from the Hugging Face Hub on startup, ensuring that it is always serving predictions using the most up-to-date intelligence.

## Features
- **FastAPI Framework**: High-performance, asynchronous REST API.
- **Machine Learning Integration**: Serves an XGBoost classification model using `scikit-learn` and `joblib`.
- **Hugging Face Hub Support**: Dynamically fetches the serialized model file from a configured Hugging Face repository.
- **Health Checks**: Built-in `/api/v1/health` endpoint for monitoring service status.
- **Swagger Documentation**: Interactive API documentation automatically available at `/docs`.

## Project Structure
```text
.
├── app
│   ├── api
│   │   └── routes.py         # API Endpoints (Health, Predict)
│   ├── core
│   │   └── config.py         # Application settings and environment variables
│   ├── schemas
│   │   └── prediction.py     # Pydantic models for request/response validation
│   └── services
│       └── model_service.py  # Logic for downloading & loading the ML model
├── main.py                   # FastAPI application entry point
├── test_api.py               # Integration test script for API verification
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables configuration
```

## Setup and Installation

### Prerequisites
- Python 3.9+
- A Hugging Face account and Access Token (if the model repository is private or to avoid rate limits).

### 1. Clone the repository
Navigate to the project directory:
```bash
cd product-price-api
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Environment Variables Configuration
Create a `.env` file in the root directory based on the following structure. Replace `HF_TOKEN` with your Hugging Face access token:
```env
HF_TOKEN=your_huggingface_access_token
HF_REPO_ID=JunaidUTH/product-price-evolution
MODEL_FILENAME=xgboost_trend_model.pkl
PORT=8000
HOST=0.0.0.0
```

### 5. Run the Application
Start the FastAPI server using `uvicorn`:
```bash
python main.py
```
Alternatively, you can run uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`. 
Interactive API docs can be accessed at `http://localhost:8000/docs`.

## API Endpoints Detailed

### 1. Predict Price Trend
**Endpoint:** `POST /api/v1/predict`

This endpoint accepts product pricing features and returns the predicted future price trend along with the probabilities for each class.

**Request Payload (JSON):**
- `prix_usd` (float): The current price of the product in USD.
- `prix_lag_1` (float): The price of the product 1 period (day) ago.
- `prix_lag_3` (float): The price of the product 3 periods (days) ago.
- `prix_lag_7` (float): The price of the product 7 periods (days) ago.
- `volatilite_7j` (float): The standard deviation of the price over the last 7 periods, representing price volatility.

**Example Request:**
```json
{
  "prix_usd": 150.50,
  "prix_lag_1": 150.50,
  "prix_lag_3": 145.00,
  "prix_lag_7": 140.00,
  "volatilite_7j": 4.5
}
```

**Response Payload (JSON):**
- `trend_id` (integer): The numeric identifier of the predicted trend (`0`, `1`, or `2`).
- `trend_label` (string): The human-readable label (`BAISSE`, `STABLE`, or `HAUSSE`).
- `probabilities` (list of floats): The confidence probabilities for each class `[BAISSE, STABLE, HAUSSE]`.
- `model_version` (string): The Hugging Face repository ID of the model used for the prediction.

**Example Response:**
```json
{
  "trend_id": 2,
  "trend_label": "HAUSSE",
  "probabilities": [0.05, 0.15, 0.80],
  "model_version": "JunaidUTH/product-price-evolution"
}
```

### 2. Health Check
**Endpoint:** `GET /api/v1/health`

Verifies that the API is running and that the machine learning model has been successfully downloaded and loaded into memory.

**Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Testing the API
An automated testing script is provided to verify the functionality of the API. It spins up the server in the background, hits the health and prediction endpoints, and prints the results.

Run the test script:
```bash
python test_api.py
```
This will test the model prediction pipeline from end-to-end to ensure the model responds correctly to valid feature inputs.

## How the Machine Learning Model Works

The model served by this API is an **XGBoost Classifier** designed specifically to forecast a product's price trajectory. It operates on a 7-day prediction horizon. 

### Data Extraction and Feature Engineering
The model is trained on historical scraping data. For each product, the training pipeline computes:
1. **Time-shifted Prices (Lags):** It looks at what the price was 1, 3, and 7 days prior (`prix_lag_1`, `prix_lag_3`, `prix_lag_7`). These features help the model understand short-term and medium-term price momentum.
2. **Volatility:** It calculates a rolling 7-day standard deviation of the price (`volatilite_7j`). High volatility might indicate frequent repricing strategies or unstable inventory.

### The Target Variable (Prediction Goal)
To train the model, the pipeline looks exactly 7 days into the future to calculate the percentage variation in price. Based on this future variation, it assigns one of three target classes:
- **0 - BAISSE (Decrease):** The price dropped by more than 1% (`< -0.01`).
- **1 - STABLE:** The price changed by less than 1% (`>= -0.01` and `<= 0.01`).
- **2 - HAUSSE (Increase):** The price increased by more than 1% (`> 0.01`).

### Training Process
- The historical dataset is split chronologically: the oldest 80% of data is used to train the XGBoost algorithm, while the most recent 20% is held back to test the model's accuracy on unseen, future-like data.
- The model is configured with a `multi:softprob` objective, meaning it calculates the *probability* for each of the three possible trends rather than outputting a single definitive answer. This provides valuable confidence metrics alongside the final prediction.
- Once training is complete and validated, the model is serialized and automatically uploaded to a Hugging Face repository, where it is dynamically retrieved by this API for inference.
