from pathlib import Path
import uuid
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference import predict_image
from preprocessing import validate_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Pneumonia AI Screening Prototype",
    description="Hackathon prototype only; not for clinical diagnosis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@app.get("/")
def root():
    return {
        "message": "Pneumonia AI API is running",
        "docs": "/docs",
        "mode": "mock" if __import__("os").getenv("MODEL_MODE", "mock") == "mock" else "model",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG, JPEG, or WEBP image. DICOM is not enabled in this starter.",
        )

   
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 10 MB.")

    image_id = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / image_id
    with image_path.open("wb") as destination:
        destination.write(contents)

    try:
        quality = validate_image(image_path)
        #if not quality["is_usable"]:
            #return {
               # "case_id": image_id,
                #"prediction": None,
                #"pneumonia_probability": None,
                #"normal_probability": None,
                #"model_mode": "not_run",
                #"quality": quality,
                #"message": "Image quality checks did not pass. Upload a clearer image.",
                #"medical_notice": "Prototype only. This result is not a diagnosis.",
            #}

        result = predict_image(image_path)
        return {
            "case_id": image_id,
            **result,
            "quality": quality,
            "medical_notice": (
                "Research prototype only—not a medical diagnosis. "
                "Do not use this result to make treatment decisions. "
                "Consult a qualified healthcare professional."
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        
        raise HTTPException(status_code=500, detail="Prediction failed. Check the backend logs.")
