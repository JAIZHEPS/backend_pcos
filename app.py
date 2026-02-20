import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------
# 1. SETUP & ASSET LOADING
# ---------------------------------------------------------
app = FastAPI(
    title="PCOS Diagnostic API",
    description="Backend API for predicting PCOS based on 43 clinical features.",
    version="1.0.0"
)

# Global variables for the model and scaler
MODEL_PATH = "pcos_random_forest.pkl"
SCALER_PATH = "pcos_scaler.pkl"

try:
    # We load these globally so they stay in memory (warm start)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Check how many features the model expects based on the scaler
    EXPECTED_FEATURES = scaler.n_features_in_
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    EXPECTED_FEATURES = 43 # Default fallback

# ---------------------------------------------------------
# 2. DATA MODELS (SCHEMA)
# ---------------------------------------------------------
class PatientData(BaseModel):
    # This expects a JSON like: {"features": [28, 24.5, ... 43 times]}
    features: List[float]

# ---------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def health_check():
    """Confirms the API is alive and the model is loaded."""
    if model and scaler:
        return {"status": "online", "model_loaded": True, "expected_features": EXPECTED_FEATURES}
    return {"status": "error", "message": "Model files missing on server"}

@app.post("/predict")
def predict_pcos(patient: PatientData):
    """
    Receives 43 features, scales them, and returns PCOS prediction.
    """
    # Validation: Ensure exactly the right amount of data is sent
    if len(patient.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {EXPECTED_FEATURES} features, but received {len(patient.features)}."
        )

    try:
        # Convert to numpy and reshape for a single prediction
        input_data = np.array(patient.features).reshape(1, -1)
        
        # Apply the SAME scaling used during training
        scaled_data = scaler.transform(input_data)
        
        # Get prediction (0 or 1)
        prediction = model.predict(scaled_data)[0]
        
        # Get probability (Confidence)
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probabilities) * 100)
        
        return {
            "pcos_detected": bool(prediction),
            "confidence_percentage": round(confidence, 2),
            "interpretation": "Positive" if prediction == 1 else "Negative"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)