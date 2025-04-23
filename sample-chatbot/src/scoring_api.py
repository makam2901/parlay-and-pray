# src/scoring_api.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "parlay-and-pray-heart-disease")
MODEL_NAME = os.getenv("MODEL_NAME", "parlay_model")

# --- Load Model (Load once on startup) ---
model = None

def load_model():
    global model
    if model is None:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            # Get the latest Production or Staging version, fallback to latest overall
            latest_version = None
            for stage in ["Production", "Staging"]:
                 versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
                 if versions:
                     latest_version = max(versions, key=lambda v: int(v.version))
                     break # Found preferred stage

            if not latest_version: # Fallback to any latest version
                versions = client.get_latest_versions(MODEL_NAME, stages=[])
                if not versions:
                    raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")
                latest_version = max(versions, key=lambda v: int(v.version))

            model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model '{MODEL_NAME}' version {latest_version.version} from stage '{latest_version.current_stage}'")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Depending on strategy, you might want the app to fail startup
            # raise RuntimeError(f"Failed to load MLflow model: {e}")
            model = None # Ensure model is None if loading failed
    return model

# --- Pydantic Model for Request ---
class ScoringRequest(BaseModel):
    vector: List[float]

# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_model() # Load model when the API starts

@app.post("/score")
async def score(request: ScoringRequest):
    loaded_model = load_model() # Get loaded model (or attempt reload if failed initially)

    if loaded_model is None:
         raise HTTPException(status_code=503, detail="Model is not available")

    if len(request.vector) != 13: # Or match the exact number of features your model expects
        raise HTTPException(status_code=400, detail=f"Invalid input vector length. Expected 13 features, got {len(request.vector)}")

    try:
        input_df = pd.DataFrame([request.vector])
        # If your model expects specific column names, assign them:
        # input_df.columns = ["age", "sex", ...]
        prediction = loaded_model.predict(input_df)
        # Assuming prediction is a numpy array, get the first element
        result = prediction[0].item() # Use .item() to convert numpy type to python native type
        print(f"Input: {request.vector}, Prediction: {result}")
        return {"prediction": result}
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

@app.get("/")
def read_root():
    return {"message": "Scoring API is running"}