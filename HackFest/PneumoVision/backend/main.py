"""
PneumoVision FastAPI Application
Full-stack backend serving chest radiograph inference, model registry, report generation, and frontend UI.
"""

import os
import io
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .model import (
    DEVICE, build_faster_rcnn_model, load_model_metadata, load_models_registry,
    load_final_classification_model, FINAL_CLF_THRESHOLD
)
from .preprocessing import preprocess_xray_file
from .inference import run_detection_inference, run_classification_inference, combine_analysis_verdict
from .report import generate_html_report, generate_text_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pneumovision.main")

app = FastAPI(
    title="PneumoVision API",
    description="AI-Assisted Chest X-ray Analysis and Pneumonia-Associated Opacity Localization",
    version="1.0.0"
)

# Enable CORS for local testing & integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
model = None
clf_model = None
metadata = {}
models_registry = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "sample_images")


def init_app():
    """Initializes and loads trained detection and classification models."""
    global model, clf_model, metadata, models_registry
    if model is None:
        logger.info("Initializing PneumoVision backend...")
        metadata = load_model_metadata()
        models_registry = load_models_registry()
        try:
            model = build_faster_rcnn_model(device=DEVICE)
            logger.info("Faster R-CNN detection model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}")
            model = None

    if clf_model is None:
        try:
            clf_model = load_final_classification_model()
            logger.info("DenseNet-121 classification model loaded.")
        except Exception as e:
            logger.warning(f"Classification model load deferred/failed: {e}")
            clf_model = None


@app.on_event("startup")
def startup_event():
    init_app()


@app.get("/health")
def health_check():
    """Health check endpoint returning system status, device, and model metadata."""
    if model is None:
        init_app()
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "classification_loaded": clf_model is not None,
        "device": DEVICE,
        "detection_model": "Faster R-CNN ResNet50-FPN v2",
        "classification_model": "DenseNet-121 (pneumonia_final_best)",
        "score_threshold": metadata.get("score_threshold", 0.75),
        "clf_threshold": FINAL_CLF_THRESHOLD,
        "val_score": metadata.get("val_score", 0.7944),
        "image_size": metadata.get("image_size", 512)
    }


@app.get("/models")
def get_models():
    """Returns primary localization engine & curated classification engine info."""
    if model is None:
        init_app()
    return {
        "primary_detection": {
            "id": "rsna_fasterrcnn_v2",
            "name": "Faster R-CNN ResNet50-FPN v2 (RSNA Best)",
            "architecture": "Faster R-CNN (ResNet50-FPN v2 Backbone)",
            "task": "Pneumonia-Associated Opacity Localization & Bounding Box Detection",
            "classes": ["Background", "Pneumonia-Associated Opacity"],
            "trained_image_size": 512,
            "score_threshold": metadata.get("score_threshold", 0.75),
            "val_score": metadata.get("val_score", 0.7944),
            "device": DEVICE,
            "weights_file": "rsna_fasterrcnn_best.pth"
        },
        "curated_classification": {
            "id": "pneumonia_final_best",
            "name": "DenseNet-121 Final Best Classifier",
            "architecture": "DenseNet-121 (Sigmoid)",
            "task": "Whole-Image Pneumonia Classification & Confirmation",
            "operating_threshold": FINAL_CLF_THRESHOLD,
            "weights_file": "pneumonia_final_best.keras",
            "input_size": 224
        },
        "classification_suite": models_registry.get("classification_models", [])
    }


@app.get("/sample_images")
def list_sample_images():
    """Returns list of bundled sample chest X-rays for quick evaluation."""
    if not os.path.exists(SAMPLES_DIR):
        return {"samples": []}

    samples = []
    for f in os.listdir(SAMPLES_DIR):
        lower = f.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom")):
            fpath = os.path.join(SAMPLES_DIR, f)
            samples.append({
                "filename": f,
                "size_bytes": os.path.getsize(fpath),
                "is_dicom": lower.endswith((".dcm", ".dicom")),
                "type": "Opacity Present" if "opacity" in lower or "pneumonia" in lower else "Normal / Control"
            })
    return {"samples": samples}


@app.get("/sample_images/{filename}")
def get_sample_image_file(filename: str):
    """Serve a sample image file."""
    clean_name = os.path.basename(filename)
    fpath = os.path.join(SAMPLES_DIR, clean_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Sample image not found")
    media_type = "application/dicom" if clean_name.lower().endswith((".dcm", ".dicom")) else "image/png"
    return FileResponse(fpath, media_type=media_type)


@app.post("/predict")
async def predict_xray(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    Main radiograph analysis endpoint:
    - Preprocesses radiograph (DICOM / PNG / JPG)
    - Runs Faster R-CNN opacity localization
    - Automatically runs DenseNet-121 classifier
    - Synthesizes dual-engine diagnostic verdict
    """
    global model
    if model is None:
        init_app()
    if model is None:
        raise HTTPException(status_code=503, detail="Detection model is not loaded. Check backend logs.")

    # Validate file extension
    filename = file.filename or "unknown.png"
    lower_name = filename.lower()
    allowed_exts = (".jpg", ".jpeg", ".png", ".dcm", ".dicom")
    if not any(lower_name.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{filename}'. Supported formats: JPG, JPEG, PNG, DICOM (.dcm)"
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Preprocess
        tensor_3ch, orig_dims, display_b64, fmt, meta = preprocess_xray_file(content, filename)

        # 1. Detection inference (Faster R-CNN)
        inf_result = run_detection_inference(
            model=model,
            tensor_3ch=tensor_3ch,
            orig_dims=orig_dims,
            score_threshold=threshold
        )

        # 2. Classification inference (DenseNet-121)
        clf_result = run_classification_inference("pneumonia_final_best", tensor_3ch)

        # 3. Combine findings into unified verdict
        verdict = combine_analysis_verdict(inf_result, clf_result, FINAL_CLF_THRESHOLD)

        total_latency = round(
            inf_result.get("inference_time_ms", 0.0) + clf_result.get("classification_time_ms", 0.0),
            1
        )

        clf_summary = {
            "model_name": "DenseNet-121 Final Best",
            "probability": clf_result.get("probability", 0.0),
            "probability_percent": clf_result.get("probability_percent", "0.0%"),
            "threshold": FINAL_CLF_THRESHOLD,
            "threshold_percent": f"{int(FINAL_CLF_THRESHOLD * 100)}%",
            "is_positive": verdict["clf_is_positive"],
            "status": clf_result.get("status", "unknown"),
            "latency_ms": clf_result.get("classification_time_ms", 0.0)
        }

        response = {
            "success": True,
            "filename": filename,
            "is_positive": verdict["is_positive"],
            "finding": verdict["finding"],
            "finding_description": verdict["finding_description"],
            "confidence": verdict["confidence"],
            "confidence_percent": verdict["confidence_percent"],
            "num_detections": inf_result["num_detections"],
            "localization": verdict["localization"],
            "inference_time_ms": total_latency,
            "applied_threshold": inf_result["applied_threshold"],
            "detection": inf_result,
            "classification": clf_summary,
            "secondary_classification": clf_result,
            "image": {
                "width": orig_dims[0],
                "height": orig_dims[1],
                "format": fmt,
                "metadata": meta
            },
            "detections": inf_result["detections"],
            "display_image": display_b64,
            "disclaimer": "This system is a research and decision-support prototype and is not a medical diagnosis."
        }
        return response

    except Exception as e:
        logger.error(f"Inference error on {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")


@app.post("/predict_dual")
async def predict_dual_engine(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    secondary_model: Optional[str] = Form("best_final_model")
):
    """
    Dual-engine analysis endpoint (backward compatible):
    Runs unified analysis. If a non-default secondary model is requested, runs that model.
    """
    base_res = await predict_xray(file=file, threshold=threshold)

    # If another secondary model was specifically requested
    if secondary_model and secondary_model not in ("none", "best_final_model", "pneumonia_final_best", "auto"):
        try:
            await file.seek(0)
            content = await file.read()
            tensor_3ch, _, _, _, _ = preprocess_xray_file(content, file.filename or "xray.png")
            custom_clf = run_classification_inference(secondary_model, tensor_3ch)
            base_res["secondary_classification"] = custom_clf
        except Exception as e:
            logger.warning(f"Custom secondary classification failed: {e}")

    return base_res


@app.post("/report/download")
async def download_report(request: Request):
    """Generates a downloadable HTML or plain text report from analysis JSON."""
    data = await request.json()
    report_format = data.get("format", "html").lower()

    if report_format == "text":
        text_content = generate_text_report(data)
        return PlainTextResponse(
            content=text_content,
            headers={"Content-Disposition": f"attachment; filename=PneumoVision_Report_{data.get('filename', 'xray')}.txt"}
        )
    else:
        html_content = generate_html_report(data)
        return HTMLResponse(
            content=html_content,
            headers={"Content-Disposition": f"attachment; filename=PneumoVision_Report_{data.get('filename', 'xray')}.html"}
        )


# Serve frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index(request: Request):
        """Serve the PneumoVision frontend dashboard."""
        accept = request.headers.get("accept", "")
        if "text/html" in accept or "*/*" in accept:
            index_path = os.path.join(FRONTEND_DIR, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
        return {
            "name": "PneumoVision API",
            "description": "AI-Assisted Chest X-ray Analysis and Pneumonia-Associated Opacity Localization",
            "status": "online",
            "docs": "/docs",
            "frontend": "/"
        }
else:
    @app.get("/")
    def root():
        return {
            "name": "PneumoVision API",
            "description": "AI-Assisted Chest X-ray Analysis and Pneumonia-Associated Opacity Localization",
            "status": "online",
            "docs": "/docs"
        }
