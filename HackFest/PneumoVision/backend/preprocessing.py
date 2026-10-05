"""
PneumoVision Preprocessing Pipeline
Reproduces the exact training preprocessing for Chest X-ray radiographs:
- DICOM support via pydicom (MONOCHROME1 inversion, 1st-99th percentile clipping & normalization)
- Standard image support (JPG, JPEG, PNG)
- Bilinear resizing to 512x512
- 3-channel grayscale tensor expansion [3, 512, 512]
"""

import io
import base64
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
import pydicom


def preprocess_dicom_bytes(file_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int], str, Dict[str, Any]]:
    """
    Exact preprocessing for DICOM files as specified:
    1. Read with pydicom
    2. Get pixel_array
    3. Convert to float32
    4. If PhotometricInterpretation == 'MONOCHROME1': invert image
    5. Calculate 1st and 99th percentile
    6. Clip and normalize to [0, 1]
    7. Convert to uint8 grayscale
    8. Resize to 512x512 using bilinear interpolation
    9. Convert grayscale image to tensor
    10. Repeat grayscale channel 3 times: [1, H, W] -> [3, H, W]
    """
    dcm = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    pixel_array = dcm.pixel_array.astype(np.float32)

    orig_height, orig_width = pixel_array.shape[:2]

    # Check Photometric Interpretation
    photo_interp = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photo_interp == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # 1st and 99th percentile clipping & normalization
    p1 = np.percentile(pixel_array, 1)
    p99 = np.percentile(pixel_array, 99)
    if p99 > p1:
        clipped = np.clip(pixel_array, p1, p99)
        normalized = (clipped - p1) / (p99 - p1)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.float32)

    # Convert to uint8 grayscale
    uint8_img = (normalized * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")

    # Bilinear resize to 512x512
    resized_pil = pil_img.resize((512, 512), resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0

    # Grayscale tensor repeated 3 times: [3, 512, 512]
    tensor_1ch = torch.from_numpy(resized_arr).unsqueeze(0)  # [1, 512, 512]
    tensor_3ch = tensor_1ch.repeat(3, 1, 1)  # [3, 512, 512]

    # Convert processed PIL image to Base64 PNG for web display
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    display_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    meta = {
        "Modality": getattr(dcm, "Modality", "CR/DX"),
        "PhotometricInterpretation": str(photo_interp),
        "PatientPosition": getattr(dcm, "PatientPosition", "AP/PA"),
        "BodyPartExamined": getattr(dcm, "BodyPartExamined", "CHEST")
    }

    return tensor_3ch, (orig_width, orig_height), display_b64, meta


def preprocess_standard_image_bytes(file_bytes: bytes, file_ext: str = "PNG") -> Tuple[torch.Tensor, Tuple[int, int], str, Dict[str, Any]]:
    """
    Standard image (JPG, JPEG, PNG) preprocessing aligned with training pipeline:
    1. Read with PIL, convert to grayscale
    2. Convert to float32
    3. Calculate 1st and 99th percentile clipping and normalize to [0, 1]
    4. Convert to uint8 grayscale
    5. Resize to 512x512 using bilinear interpolation
    6. Convert to tensor and repeat 3 times: [3, 512, 512]
    """
    pil_raw = Image.open(io.BytesIO(file_bytes))
    orig_width, orig_height = pil_raw.size

    # Convert to grayscale
    gray_pil = pil_raw.convert("L")
    arr = np.array(gray_pil, dtype=np.float32)

    # 1st and 99th percentile clipping & normalization
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    if p99 > p1:
        clipped = np.clip(arr, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
    else:
        norm = arr / 255.0 if np.max(arr) > 1.0 else arr

    uint8_img = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    clean_pil = Image.fromarray(uint8_img, mode="L")

    # Resize to 512x512
    resized_pil = clean_pil.resize((512, 512), resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0

    tensor_1ch = torch.from_numpy(resized_arr).unsqueeze(0)
    tensor_3ch = tensor_1ch.repeat(3, 1, 1)  # [3, 512, 512]

    # Create display base64
    buffer = io.BytesIO()
    clean_pil.save(buffer, format="PNG")
    display_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    meta = {
        "Modality": "Radiograph (Raster)",
        "Format": file_ext.upper(),
        "ColorChannels": "Grayscale Normalized"
    }

    return tensor_3ch, (orig_width, orig_height), display_b64, meta


def preprocess_xray_file(file_bytes: bytes, filename: str) -> Tuple[torch.Tensor, Tuple[int, int], str, str, Dict[str, Any]]:
    """
    Main dispatch function: auto-detects DICOM vs standard raster image.
    Returns:
    - model_tensor: torch.Tensor [3, 512, 512]
    - (orig_width, orig_height)
    - display_b64: data URL for frontend rendering
    - detected_format: "DICOM" | "PNG" | "JPEG"
    - metadata dict
    """
    lower = filename.lower()
    is_dicom = lower.endswith(".dcm") or lower.endswith(".dicom") or file_bytes[:132].endswith(b"DICM")

    if is_dicom:
        tensor_3ch, dims, display_b64, meta = preprocess_dicom_bytes(file_bytes)
        return tensor_3ch, dims, display_b64, "DICOM", meta
    else:
        ext = "PNG" if lower.endswith(".png") else "JPEG"
        tensor_3ch, dims, display_b64, meta = preprocess_standard_image_bytes(file_bytes, ext)
        return tensor_3ch, dims, display_b64, ext, meta
