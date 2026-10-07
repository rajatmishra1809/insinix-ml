from pathlib import Path
import json
from enum import Enum

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import RoverPolicy

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rover_model.pth"
SCALER_PATH = BASE_DIR / "scaler.json"

app = FastAPI(
    title="INSINIX ML Rover Controller Server",
    description="Real-time PyTorch inference server for INSINIX autonomous rover steering & obstacle avoidance policy",
    version="1.0.0"
)

# Enable CORS for browser simulator client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActionRecommendation(str, Enum):
    PROCEED = "PROCEED"
    SLOW_DOWN = "SLOW_DOWN"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class RoverStateRequest(BaseModel):
    lidar: list[float] = Field(..., min_length=16, max_length=16, description="16 normalized LiDAR ray distances [0, 1]")
    goal_angle: float = Field(..., description="Normalized goal angle [-1, 1]")
    goal_distance: float = Field(..., description="Normalized goal distance [0, 1]")
    speed: float = Field(..., description="Normalized current rover speed [0, 1]")
    heading: float = Field(..., description="Normalized heading angle [-1, 1]")
    payload_weight: float = Field(..., description="Normalized payload weight [0, 1]")
    friction: float = Field(..., description="Normalized terrain friction [0, 1]")


# Global model & scaler references
model = None
feature_mean = None
feature_std = None
model_load_error = None


def load_model_and_scaler():
    global model, feature_mean, feature_std, model_load_error

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        model_load_error = f"Model file ({MODEL_PATH.name}) or Scaler file ({SCALER_PATH.name}) not found. Please run train.py first."
        print(f"WARNING: {model_load_error}")
        return False

    try:
        scaler = json.loads(SCALER_PATH.read_text())
        feature_mean = np.asarray(scaler["mean"], dtype=np.float32)
        feature_std = np.asarray(scaler["std"], dtype=np.float32)

        model_instance = RoverPolicy(input_size=22)
        model_instance.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
        model_instance.eval()
        model = model_instance
        model_load_error = None
        print("SUCCESS: INSINIX ML model and feature scaler loaded.")
        return True
    except Exception as e:
        model_load_error = str(e)
        print(f"ERROR loading model: {e}")
        return False


@app.on_event("startup")
def startup_event():
    load_model_and_scaler()


def apply_safety_layer(lidar_readings: list[float], raw_target_speed: float, emergency_dist: float = 0.10, slowdown_dist: float = 0.20):
    """
    Evaluates LiDAR readings against safety thresholds and overrides unsafe raw network outputs.
    """
    if not lidar_readings:
        return ActionRecommendation.EMERGENCY_STOP, 0.0

    nearest_obstacle = float(min(lidar_readings))

    if nearest_obstacle < emergency_dist:
        return ActionRecommendation.EMERGENCY_STOP, 0.0

    if nearest_obstacle < slowdown_dist or raw_target_speed < 0.20:
        safe_speed = float(min(raw_target_speed, 0.35))
        return ActionRecommendation.SLOW_DOWN, safe_speed

    return ActionRecommendation.PROCEED, float(raw_target_speed)


@app.get("/")
def root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "input_features": 22,
        "outputs": ["steering", "target_speed", "recommendation"],
        "error": model_load_error
    }


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "model_loaded": model is not None,
        "status": "online" if model is not None else "degraded",
        "error": model_load_error
    }


@app.post("/reload")
def reload_model():
    success = load_model_and_scaler()
    return {
        "reloaded": success,
        "model_loaded": model is not None,
        "error": model_load_error
    }


@app.post("/predict")
def predict(state: RoverStateRequest):
    if model is None:
        # Try reloading once in case user just trained
        if not load_model_and_scaler():
            raise HTTPException(
                status_code=503,
                detail=f"ML Model unavailable. {model_load_error}"
            )

    # Validate inputs
    for idx, dist in enumerate(state.lidar):
        if np.isnan(dist) or np.isinf(dist):
            state.lidar[idx] = 1.0

    features = np.array(
        state.lidar + [
            state.goal_angle,
            state.goal_distance,
            state.speed,
            state.heading,
            state.payload_weight,
            state.friction,
        ],
        dtype=np.float32,
    )

    if len(features) != 22:
        raise HTTPException(status_code=400, detail=f"Expected 22 state vector features, got {len(features)}")

    # Standardize inputs using fitted scaler
    normalized_features = (features - feature_mean) / feature_std
    tensor_input = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)

    # PyTorch Inference
    with torch.no_grad():
        raw_output = model(tensor_input)[0].numpy()

    steering = float(np.clip(raw_output[0], -1.0, 1.0))
    target_speed = float(np.clip(raw_output[1], 0.0, 1.0))

    # Apply safety layer
    recommendation, safe_speed = apply_safety_layer(state.lidar, target_speed)

    return {
        "steering": round(steering, 4),
        "target_speed": round(safe_speed, 4),
        "recommendation": recommendation.value,
        "raw_speed": round(target_speed, 4)
    }
