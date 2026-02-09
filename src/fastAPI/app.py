import os
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO)
#logging.info("Model loaded successfully")

# -----------------------------
# Load .env
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes_best_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(TRACKING_URI)
model = None

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Diabetes Regression API", version="0.1.0")

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float

# -----------------------------
# Startup: load model
# -----------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        logging.info(f"✅ Loaded model from {model_uri}")
        print(f"✅ Loaded model from {model_uri}")
    except Exception as e:
        print("❌ Failed to load model")
        print(e)
        logging.error("❌ Failed to load model")
        logging.exception(e)
        
        model = None

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "FastAPI ML service is running"}

@app.get("/health")
def health():
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@app.get("/version")
def version():
    return {"model_name": MODEL_NAME, "model_stage": MODEL_STAGE}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    logging.info(f"Received request: {request.features}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
