"""
PneumoVision Inference Engine
Performs real forward inference using the trained Faster R-CNN ResNet50-FPN v2 model.
Maps 512x512 coordinates back to original image resolution.
Calculates approximate anatomical localization (Right/Left Lung, Upper/Mid/Lower Zone).
Supports optional secondary classification validation.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from .model import DEVICE, load_model_metadata, get_classification_model


def get_anatomical_zone(x: float, y: float, w: float, h: float, orig_w: int, orig_h: int) -> str:
    """
    Computes radiological anatomical localization:
    Note: Radiographs are viewed in standard anatomical position:
    - Image Left (x < 0.48) corresponds to the Patient's RIGHT Lung.
    - Image Right (x > 0.52) corresponds to the Patient's LEFT Lung.
    - 0.48 <= x <= 0.52 corresponds to Central / Perihilar / Mediastinal zone.
    Vertical zones:
    - y < 0.35: Upper Zone
    - 0.35 <= y <= 0.65: Mid Zone
    - y > 0.65: Lower Zone / Basal
    """
    cx = (x + w / 2.0) / float(orig_w)
    cy = (y + h / 2.0) / float(orig_h)

    # Lateral determination
    if cx < 0.45:
        lateral = "Right Lung"
    elif cx > 0.55:
        lateral = "Left Lung"
    else:
        lateral = "Central / Perihilar"

    # Vertical determination
    if cy < 0.35:
        zone = "Upper Zone"
    elif cy > 0.65:
        zone = "Lower Zone (Base)"
    else:
        zone = "Mid Zone"

    if lateral.startswith("Central"):
        return f"{lateral} ({zone})"
    return f"{lateral} {zone}"


def summarize_localization(zones: List[str]) -> str:
    """Combines individual region localizations into a concise radiological summary."""
    if not zones:
        return "None detected"

    unique_zones = list(dict.fromkeys(zones))
    has_right = any("Right Lung" in z for z in unique_zones)
    has_left = any("Left Lung" in z for z in unique_zones)

    if has_right and has_left:
        return f"Bilateral Opacities ({', '.join(unique_zones)})"
    return ", ".join(unique_zones)


def run_detection_inference(
    model: torch.nn.Module,
    tensor_3ch: torch.Tensor,
    orig_dims: Tuple[int, int],
    score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs Faster R-CNN detection on the preprocessed 3-channel [3, 512, 512] tensor.
    Filters detections using the specified or trained score_threshold.
    Converts bounding boxes from 512x512 back to original image coordinates.
    """
    metadata = load_model_metadata()
    thresh = score_threshold if (score_threshold is not None and score_threshold > 0) else float(metadata.get("score_threshold", 0.75))

    orig_w, orig_h = orig_dims
    scale_x = orig_w / 512.0
    scale_y = orig_h / 512.0

    # Ensure batch dimension [1, 3, 512, 512]
    if tensor_3ch.dim() == 3:
        input_tensor = tensor_3ch.unsqueeze(0).to(DEVICE)
    else:
        input_tensor = tensor_3ch.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        predictions = model(input_tensor)
    inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    detections = []
    zone_list = []

    for box, score, label in zip(boxes, scores, labels):
        # Class 1 = pneumonia-associated opacity
        if label == 1 and score >= thresh:
            x1, y1, x2, y2 = box
            # Clip to 512 range
            x1 = max(0.0, min(512.0, float(x1)))
            y1 = max(0.0, min(512.0, float(y1)))
            x2 = max(0.0, min(512.0, float(x2)))
            y2 = max(0.0, min(512.0, float(y2)))

            # Rescale to original resolution
            orig_x = round(x1 * scale_x, 1)
            orig_y = round(y1 * scale_y, 1)
            orig_width = round((x2 - x1) * scale_x, 1)
            orig_height = round((y2 - y1) * scale_y, 1)

            zone = get_anatomical_zone(orig_x, orig_y, orig_width, orig_height, orig_w, orig_h)
            zone_list.append(zone)

            detections.append({
                "confidence": round(float(score), 4),
                "confidence_percent": f"{round(float(score) * 100.0, 1)}%",
                "anatomical_zone": zone,
                "box": {
                    "x": orig_x,
                    "y": orig_y,
                    "width": orig_width,
                    "height": orig_height
                },
                "model_coord_box": {
                    "x": round(x1, 1),
                    "y": round(y1, 1),
                    "width": round(x2 - x1, 1),
                    "height": round(y2 - y1, 1)
                }
            })

    # Sort detections descending by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    num_det = len(detections)
    if num_det > 0:
        finding = "Pneumonia-associated opacity detected"
        finding_description = "AI model detected a region of interest consistent with pneumonia-associated opacity."
        top_conf = detections[0]["confidence"]
        top_conf_percent = f"{round(top_conf * 100.0, 1)}%"
        localization = summarize_localization(zone_list)
    else:
        finding = "No pneumonia-associated opacity detected"
        finding_description = "No regions of interest exceeding the confidence threshold were detected in this radiograph."
        top_conf = 0.0
        top_conf_percent = "0.0%"
        localization = "None detected"

    return {
        "finding": finding,
        "finding_description": finding_description,
        "confidence": top_conf,
        "confidence_percent": top_conf_percent,
        "num_detections": num_det,
        "localization": localization,
        "detections": detections,
        "inference_time_ms": inference_time_ms,
        "applied_threshold": thresh
    }


def run_classification_inference(model_id: str, tensor_3ch: torch.Tensor) -> Dict[str, Any]:
    """
    Runs classification model (e.g. DenseNet-121 fine-tuned) on 224x224 input.
    """
    try:
        clf_model = get_classification_model(model_id)
        if clf_model is None:
            return {"model_id": model_id, "error": "Model not available", "status": "unavailable"}

        # Convert tensor [3, 512, 512] in [0, 1] to [1, 224, 224, 3] for Keras model
        np_img = tensor_3ch.permute(1, 2, 0).cpu().numpy()  # [512, 512, 3]
        pil_img = Image.fromarray((np_img * 255.0).astype(np.uint8))
        resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)

        # Standard Keras DenseNet input preprocessing [0, 255] float32
        x = np.array(resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)  # [1, 224, 224, 3]

        t0 = time.perf_counter()
        pred = clf_model(x)
        clf_time = round((time.perf_counter() - t0) * 1000.0, 1)

        if hasattr(pred, "detach"):
            pred_arr = pred.detach().cpu().numpy()
        elif hasattr(pred, "numpy"):
            pred_arr = pred.numpy()
        else:
            pred_arr = np.array(pred)

        val = float(pred_arr.flatten()[0])
        val = max(0.0, min(1.0, val))
        return {
            "model_id": model_id,
            "probability": round(val, 4),
            "probability_percent": f"{round(val * 100.0, 1)}%",
            "classification_time_ms": clf_time,
            "status": "success"
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "error": str(e),
            "status": "failed"
        }


def combine_analysis_verdict(
    detection_result: Dict[str, Any],
    clf_result: Optional[Dict[str, Any]],
    clf_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Combines Faster R-CNN object detection with DenseNet-121 whole-image classification
    to formulate a robust, dual-validated radiological finding.
    """
    num_det = detection_result.get("num_detections", 0)
    det_conf = detection_result.get("confidence", 0.0)
    det_localization = detection_result.get("localization", "None detected")

    clf_prob = 0.0
    clf_status = "unavailable"
    if clf_result and clf_result.get("status") == "success":
        clf_prob = float(clf_result.get("probability", 0.0))
        clf_status = "success"

    clf_is_positive = (clf_prob >= clf_threshold)

    # Both agree on positive
    if num_det > 0 and clf_is_positive:
        is_positive = True
        finding = "Pneumonia-associated opacity detected"
        finding_description = (
            f"Dual-engine validation: Faster R-CNN localized {num_det} focal opacity region(s) "
            f"and DenseNet-121 classifier indicated pneumonia features ({round(clf_prob * 100.0, 1)}% probability)."
        )
        primary_confidence = max(det_conf, clf_prob)
        localization = det_localization

    # Detection localized opacity, classifier below threshold
    elif num_det > 0:
        is_positive = True
        finding = "Pneumonia-associated opacity detected"
        finding_description = (
            f"Faster R-CNN localized {num_det} focal region(s) of interest ({det_localization}). "
            f"DenseNet-121 classification probability: {round(clf_prob * 100.0, 1)}%."
        )
        primary_confidence = det_conf
        localization = det_localization

    # Detection found no focal boxes, but classifier detected diffuse pneumonia pattern
    elif clf_is_positive:
        is_positive = True
        finding = "Pneumonia pattern detected (Classification)"
        finding_description = (
            f"DenseNet-121 whole-image classifier detected pneumonia features with {round(clf_prob * 100.0, 1)}% probability "
            f"(threshold: {int(clf_threshold * 100)}%), though focal opacity did not exceed localized bounding box threshold."
        )
        primary_confidence = clf_prob
        localization = "Diffuse / Non-focal pattern"

    # Both indicate normal / no pneumonia
    else:
        is_positive = False
        finding = "No pneumonia-associated opacity detected"
        finding_description = (
            "No focal opacity regions or diffuse pneumonia characteristics were detected above decision thresholds."
        )
        primary_confidence = 0.0
        localization = "None detected"

    return {
        "is_positive": is_positive,
        "finding": finding,
        "finding_description": finding_description,
        "confidence": round(primary_confidence, 4),
        "confidence_percent": f"{round(primary_confidence * 100.0, 1)}%",
        "localization": localization,
        "num_detections": num_det,
        "clf_probability": round(clf_prob, 4),
        "clf_probability_percent": f"{round(clf_prob * 100.0, 1)}%",
        "clf_is_positive": clf_is_positive,
        "clf_status": clf_status,
        "clf_threshold": clf_threshold
    }

