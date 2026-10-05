"""
PneumoVision Model Architecture Loader
Reconstructs the exact Faster R-CNN ResNet50-FPN v2 architecture trained on the RSNA Pneumonia Detection dataset.
Supports both primary detection inference and secondary Keras classification models.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger("pneumovision.model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "rsna_fasterrcnn_best.pth")
META_PATH = os.path.join(MODELS_DIR, "rsna_fasterrcnn_meta.json")
REGISTRY_PATH = os.path.join(MODELS_DIR, "models_registry.json")
FINAL_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "pneumonia_final_best.keras")
FINAL_CLF_THRESHOLD = 0.85  # Optimized on validation set


def load_model_metadata() -> Dict[str, Any]:
    """Load metadata saved during model training."""
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {META_PATH}: {e}")
    # Default fallback matching training configuration
    return {
        "model_version": "v2",
        "image_size": 512,
        "num_classes": 2,
        "score_threshold": 0.75,
        "val_score": 0.7944
    }


def load_models_registry() -> Dict[str, Any]:
    """Load catalog of all available models in the HackFest suite."""
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read registry: {e}")
    return {}


def build_faster_rcnn_model(weights_path: Optional[str] = None, device: str = DEVICE):
    """
    Reconstructs the EXACT Faster R-CNN ResNet50-FPN v2 architecture:
    - min_size=512, max_size=512
    - box_score_thresh=0.05
    - box_detections_per_img=30
    - FastRCNNPredictor(in_features, 2)
    - Loads state_dict from trained rsna_fasterrcnn_best.pth
    """
    if weights_path is None:
        weights_path = BEST_MODEL_PATH

    logger.info(f"Building Faster R-CNN ResNet50-FPN v2 on device: {device}")
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        min_size=512,
        max_size=512,
        box_score_thresh=0.05,
        box_detections_per_img=30,
        trainable_backbone_layers=5
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Trained model weights not found at: {weights_path}. "
            "Please ensure rsna_fasterrcnn_best.pth is placed in backend/models/"
        )

    logger.info(f"Loading state_dict from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    load_res = model.load_state_dict(state_dict)
    logger.info(f"State dict loaded successfully: {load_res}")

    model.to(device)
    model.eval()
    return model


# Classification model cache
_CLASSIFICATION_MODELS_CACHE = {}
_FINAL_CLF_MODEL = None


def load_final_classification_model():
    """Load the single curated best classification model for automatic dual-engine use."""
    global _FINAL_CLF_MODEL
    if _FINAL_CLF_MODEL is not None:
        return _FINAL_CLF_MODEL

    import keras as _keras
    if not os.path.exists(FINAL_CLF_MODEL_PATH):
        # Fall back to best_final_model in classification folder
        fallback = os.path.join(MODELS_DIR, "classification", "best_models", "best_final_model.keras")
        if os.path.exists(fallback):
            logger.info(f"Loading classification model from fallback: {fallback}")
            _FINAL_CLF_MODEL = _keras.models.load_model(fallback)
        else:
            logger.warning("No final classification model found. Dual-engine will be disabled.")
            return None
    else:
        logger.info(f"Loading final classification model from: {FINAL_CLF_MODEL_PATH}")
        _FINAL_CLF_MODEL = _keras.models.load_model(FINAL_CLF_MODEL_PATH)

    return _FINAL_CLF_MODEL


def get_classification_model(model_filename: str):
    """
    Load a Keras classification model by filename (for API backward compat).
    If model_filename is 'auto' or 'pneumonia_final_best', loads the curated best model.
    """
    if model_filename in ('auto', 'pneumonia_final_best', 'pneumonia_final_best.keras'):
        return load_final_classification_model()

    if model_filename in _CLASSIFICATION_MODELS_CACHE:
        return _CLASSIFICATION_MODELS_CACHE[model_filename]

    import keras as _keras

    # Search for model file in classification folder
    clf_base = os.path.join(MODELS_DIR, "classification")
    target_path = None
    for root, _, files in os.walk(clf_base):
        for f in files:
            if f == model_filename or f == f"{model_filename}.keras":
                target_path = os.path.join(root, f)
                break
        if target_path:
            break

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(f"Classification model '{model_filename}' not found under {clf_base}")

    logger.info(f"Loading Keras model from: {target_path}")
    clf_model = _keras.models.load_model(target_path)
    _CLASSIFICATION_MODELS_CACHE[model_filename] = clf_model
    return clf_model
