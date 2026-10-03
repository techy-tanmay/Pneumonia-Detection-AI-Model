
import os
from pathlib import Path
import random

MODEL_MODE = os.getenv("MODEL_MODE", "mock").lower()
MODEL_PATH = Path(__file__).resolve().parent / "models" / "densenet121_best.pth"

_model = None


def load_model():
    
    raise NotImplementedError(
        "The trained model adapter has not been connected. Set MODEL_MODE=mock for UI development."
    )


def predict_image(image_path: Path) -> dict:
    if MODEL_MODE == "mock":
       
        pneumonia_probability = round(random.uniform(0.15, 0.90), 4)
        normal_probability = round(1.0 - pneumonia_probability, 4)
        prediction = (
            "Pneumonia" if pneumonia_probability >= 0.5 else "Normal"
        )
        return {
            "prediction": prediction,
            "pneumonia_probability": pneumonia_probability,
            "normal_probability": normal_probability,
            "model_mode": "mock_demo",
            "explanation": (
                "DEMO DATA ONLY: these scores are randomly generated and are not produced by an ML model."
            ),
            "heatmap_url": None,
        }

    if MODEL_MODE == "model":
        
        raise NotImplementedError(
            "Replace this adapter with the team's finalized model inference code."
        )

    raise ValueError("MODEL_MODE must be 'mock' or 'model'.")
