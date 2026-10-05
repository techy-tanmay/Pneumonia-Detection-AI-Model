# PneumoVision — AI-Assisted Chest X-ray Analysis and Pneumonia-Associated Opacity Localization

**KLE Tech HackFest 2026 Project**  
*A Research and Clinical Decision-Support Prototype*

---

## 1. Project Overview

**PneumoVision** is a full-stack, deep learning-powered medical imaging web application developed for analyzing chest radiographs (CXR) and accurately localizing pneumonia-associated opacities.

Instead of outputting a generic black-box prediction or stating that "the patient has pneumonia," PneumoVision utilizes an **RSNA-trained Faster R-CNN (ResNet50-FPN v2)** object detector to locate exact regions of interest consistent with pneumonia-associated opacities, rescales coordinates to the native patient image resolution, identifies anatomical zones (e.g., *Right Lung Mid Zone*, *Bilateral Opacities*), and provides an interactive clinical decision-support interface.

In addition to the primary Faster R-CNN detection engine, PneumoVision integrates a structured suite of **18 trained Keras classification models** (Fine-tuned DenseNet-121, ImageNet transfer stages, Viral-Aware models, Zoom-Crop augmented models, and Domain-Adapted BN variants) for dual-engine verification.

---

## 2. System Architecture

```mermaid
flowchart TD
    User([Clinician / User]) -->|Uploads DICOM / PNG / JPG| Frontend[PneumoVision Frontend Dashboard]
    Frontend -->|POST /predict or /predict_dual| FastAPI[FastAPI Backend Server]
    FastAPI --> Preproc[Preprocessing Engine]
    
    subgraph Preprocessing_Pipeline [Exact Radiograph Preprocessing]
        Preproc --> DcmCheck{DICOM or Raster?}
        DcmCheck -->|DICOM| DcmRead[pydicom Read + MONOCHROME1 Inversion]
        DcmCheck -->|PNG/JPG| PILRead[Grayscale Conversion]
        DcmRead --> ClipNorm[1st - 99th Percentile Normalization]
        PILRead --> ClipNorm
        ClipNorm --> Resize[Bilinear Resize to 512x512]
        Resize --> Tensor[Grayscale 3-Channel Tensor [3, 512, 512]]
    end

    Tensor --> PrimaryModel[Faster R-CNN ResNet50-FPN v2\nWeights: rsna_fasterrcnn_best.pth]
    PrimaryModel --> DetectionEngine[ROI Filtering & Thresholding\nDefault: 0.75]
    DetectionEngine --> ScaleCoords[Coordinate Rescaling 512 -> Original]
    ScaleCoords --> AnatMap[Anatomical Zone Mapping\ne.g., Right Lung Mid Zone]
    
    FastAPI -.->|Optional Dual-Engine| ClfModel[DenseNet-121 Classifier\n224x224 Input]
    
    AnatMap --> ResultJSON[Formatted Diagnostic JSON Result]
    ClfModel -.-> ResultJSON
    ResultJSON --> Frontend
    Frontend --> Overlay[Interactive Bounding Box Overlay + PAC Tools]
    Frontend --> ReportGen[Printable Clinical HTML/Text Report]
```

---

## 3. Key Features

- **Multi-Format Ingestion**: Supports raw Medical DICOM (`.dcm`), PNG, JPG, and JPEG.
- **Exact Calibrated Preprocessing**:
  - Automatically detects and corrects photometric interpretation (`MONOCHROME1` inversion).
  - 1st and 99th percentile dynamic range clipping and normalization.
  - Generates web-displayable lossless representation directly for DICOM files.
- **Authentic Local Model Weights**:
  - Reconstructs exact `fasterrcnn_resnet50_fpn_v2` with `FastRCNNPredictor(in_features, 2)`.
  - Directly loads `rsna_fasterrcnn_best.pth` (173.4 MB) trained on RSNA Pneumonia Detection Challenge.
  - Zero fake numbers, fake boxes, or mock predictions.
- **Coordinate Rescaling & Anatomical Localization**:
  - Maps model bounding box predictions from 512×512 back to original patient image space.
  - Translates spatial coordinates into radiological terminology (*Right Lung Mid Zone*, *Left Lower Zone / Base*, *Bilateral Opacities*).
- **Interactive PACS-Style Web Viewer**:
  - Dynamic canvas bounding box overlay with confidence percentage tags.
  - Zoom in (`+`), Zoom out (`-`), 1:1 Reset, Invert Grayscale, and Fullscreen viewer.
  - Bi-directional hover highlighting between region list and image boxes.
- **Integrated 19-Model HackFest Suite**:
  - Primary PyTorch Faster R-CNN object detector.
  - 18 categorized Keras classification models (DenseNet best models, ImageNet transfer stages, Viral-aware classifiers, and Zoom-crop variants).
- **Formal Diagnostic Report Generation**:
  - Downloadable, print-ready HTML clinical report and structured plain text export.
- **Local Analysis History**:
  - Client-side `localStorage` analysis log with zero patient PHI persistence.

---

## 4. Directory Structure

```
PneumoVision/
├── backend/
│   ├── main.py                  # FastAPI application & REST endpoints
│   ├── model.py                 # Faster R-CNN architecture & Keras model loader
│   ├── inference.py             # Inference pipeline, coordinate scaling, localization
│   ├── preprocessing.py         # DICOM & raster radiograph preprocessing
│   ├── report.py                # HTML & plain-text report generators
│   ├── requirements.txt         # Python dependencies
│   │
│   ├── models/
│   │   ├── rsna_fasterrcnn_best.pth   # Primary trained Faster R-CNN weights (173.4 MB)
│   │   ├── rsna_fasterrcnn_meta.json  # Training metadata (threshold: 0.75, val: 0.7944)
│   │   ├── models_registry.json       # Catalog of all 19 integrated models
│   │   ├── detection/                 # Detection weights folder
│   │   └── classification/            # Categorized Keras models (18 models)
│   │       ├── best_models/
│   │       ├── imagenet_transfer/
│   │       ├── viral_aware/
│   │       ├── zoom_crop/
│   │       └── experimental_v2/
│   │
│   ├── uploads/
│   └── outputs/
│
├── frontend/
│   ├── index.html               # Medical technology UI dashboard
│   ├── style.css                # Polished responsive design system
│   └── script.js                # Interactive controller & canvas renderer
│
├── sample_images/
│   ├── sample_chest_radiograph.dcm       # Authentic DICOM test file
│   ├── sample_pneumonia_right_opacity.png # Unilateral opacity radiograph
│   ├── sample_bilateral_opacities.jpg     # Bilateral opacity radiograph
│   ├── sample_normal_chest.png           # Normal control radiograph
│   └── README.txt
│
├── README.md
├── .gitignore
└── start_backend.bat            # Windows 1-click execution script
```

---

## 5. Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Model Architecture** | Faster R-CNN (ResNet50-FPN v2 Backbone) |
| **Classes** | 2 (`0`: Background, `1`: Pneumonia-Associated Opacity) |
| **Input Resolution** | 512 × 512 × 3 Grayscale |
| **Trained Score Threshold** | `0.75` (Calibrated in `rsna_fasterrcnn_meta.json`) |
| **Validation Score** | `0.7944` |
| **Framework** | PyTorch 2.14 / Torchvision |
| **Dataset** | RSNA Pneumonia Detection Challenge |
| **Device Execution** | Auto-detects `CUDA` GPU when available; runs seamlessly on `CPU` |

---

## 6. Windows Setup & Installation

### Step 1: Open PowerShell
Open PowerShell and navigate to the project directory:
```powershell
cd "c:\Users\Deepti Patil\Desktop\HackFest\PneumoVision"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

### Step 4: Run the Application
Launch the server with `uvicorn`:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*Or simply double-click `start_backend.bat`.*

### Step 5: Open the Application
Open your web browser and navigate to:
- **Application Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. API Endpoints

### `GET /health`
Returns health check status, device, and model metadata:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_name": "Faster R-CNN ResNet50-FPN v2",
  "score_threshold": 0.75,
  "val_score": 0.7944,
  "image_size": 512
}
```

### `POST /predict`
Runs opacity detection and localization on uploaded radiograph:
- **Parameters**: `file` (multipart/form-data), `threshold` (optional float).
- **Supported Formats**: `.dcm`, `.png`, `.jpg`, `.jpeg`.
- **Response**:
```json
{
  "success": true,
  "filename": "sample_chest_radiograph.dcm",
  "finding": "Pneumonia-associated opacity detected",
  "confidence": 0.874,
  "confidence_percent": "87.4%",
  "num_detections": 1,
  "localization": "Right Lung Mid Zone",
  "inference_time_ms": 1150.4,
  "image": {
    "width": 1024,
    "height": 1024,
    "format": "DICOM"
  },
  "detections": [
    {
      "confidence": 0.874,
      "confidence_percent": "87.4%",
      "anatomical_zone": "Right Lung Mid Zone",
      "box": {
        "x": 480.0,
        "y": 320.0,
        "width": 210.0,
        "height": 180.0
      }
    }
  ]
}
```

### `POST /predict_dual`
Runs both Faster R-CNN localization and a secondary Keras classification model:
- **Parameters**: `file`, `threshold`, `secondary_model` (e.g. `best_final_model`).

### `GET /models`
Returns the complete catalog of 19 integrated models across detection and classification categories.

### `POST /report/download`
Generates a downloadable formal HTML or plain-text diagnostic report.

---

## 8. Safety & Medical Disclaimer

> [!IMPORTANT]
> **Research & Clinical Decision-Support Prototype**  
> PneumoVision is developed strictly for research, academic evaluation, and demonstration purposes. Model predictions do **not** constitute a definitive medical diagnosis. This software must **never** supersede professional clinical assessment, patient history review, or formal radiological evaluation by a licensed healthcare provider.

---

## 9. HackFest Demonstration Flow

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
2. Click one of the **Quick Demo Radiographs** (e.g., `sample_chest_radiograph.dcm` or `Right Opacity Sample`).
3. Observe the uploaded file summary card showing dimensions and format.
4. Click **Analyze X-ray**.
5. Watch the real 5-step processing pipeline: `Uploaded -> Preprocessing -> Detection -> Localization -> Complete`.
6. Inspect the visualized radiograph with genuine Faster R-CNN bounding boxes.
7. Observe the radiological finding, confidence metric, and anatomical zone (*Right Lung Mid Zone*).
8. Use image tools: **Zoom In**, **Invert Grayscale**, and **Fullscreen**.
9. Click **Download Formal Report (HTML)** to view the printable medical documentation.
10. Click **History** in the top navigation bar to view stored session runs.
11. Explore the **Model Suite** section to review the 19 categorized models.
