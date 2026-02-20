import joblib
import pickle
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
    description="Backend API for predicting PCOS based on 9 clinical features.",
    version="1.1.0"
)

# Global variables for the model and scaler (Updated to match your filenames)
MODEL_PATH = "pcos_model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    # Using pickle.load as we saved with pickle in the previous step
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # Dynamically detect how many features the model expects
    EXPECTED_FEATURES = scaler.n_features_in_
    print(f"Model loaded. Expecting {EXPECTED_FEATURES} features.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    EXPECTED_FEATURES = 9 # Fallback for your simplified dataset

# ---------------------------------------------------------
# 2. DATA MODELS (SCHEMA)
# ---------------------------------------------------------
class PatientData(BaseModel):
    # This expects a JSON like: 
    # {"features": [30, 70.5, 165, 25.8, 12.0, 0, 0, 1, 1]}
    features: List[float]

# ---------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def health_check():
    """Confirms the API is alive and the model is loaded."""
    if model and scaler:
        return {
            "status": "online", 
            "model_loaded": True, 
            "expected_features": EXPECTED_FEATURES,
            "feature_order": ["Age", "Weight", "Height", "BMI", "Hb", "WeightGain", "HairLoss", "Pimples", "Exercise"]
        }
    return {"status": "error", "message": "Model files missing on server"}

@app.post("/predict")
def predict_pcos(patient: PatientData):
    """
    Receives clinical features, scales them, and returns PCOS prediction.
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not initialized on server.")

    # Validation: Ensure exactly the right amount of data is sent (9 features)
    if len(patient.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {EXPECTED_FEATURES} features, but received {len(patient.features)}."
        )

    try:
        # 1. Convert to numpy and reshape for a single prediction
        input_data = np.array(patient.features).reshape(1, -1)
        
        # 2. Apply the SAME scaling used during training
        scaled_data = scaler.transform(input_data)
        
        # 3. Get prediction (0 or 1)
        prediction = int(model.predict(scaled_data)[0])
        
        # 4. Get probability (Confidence)
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probabilities) * 100)
        
        return {
            "pcos_detected": bool(prediction),
            "confidence_percentage": round(confidence, 2),
            "interpretation": "Positive" if prediction == 1 else "Negative",
            "recommendation": "Consult a doctor for further clinical validation." if prediction == 1 else "Results suggest low risk."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # To run this: save as main.py and run 'uvicorn main:app --reload'
    uvicorn.run(app)