from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError
import numpy as np


def open_grayscale(path: Path) -> Image.Image:
    try:
        image = Image.open(path).convert("L")
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file could not be read as an image.") from exc


def validate_image(path: Path) -> dict:
    
    image = open_grayscale(path)
    width, height = image.size
    pixels = np.asarray(image, dtype=np.float32)
    mean_brightness = float(pixels.mean())
    contrast_std = float(pixels.std())

    
    try:
        import cv2
        blur_score = float(cv2.Laplacian(pixels, cv2.CV_32F).var())
        blur_check_available = True
    except ImportError:
        blur_score = None
        blur_check_available = False

    checks = {
        "resolution": width >= 150 and height >= 150,
        "brightness": 5 <= mean_brightness <= 250,
        "contrast": contrast_std >= 5,
    }
    if blur_check_available:
        
        checks["blur"] = blur_score >= 2.0

    is_usable = all(checks.values())
    return {
        "width": width,
        "height": height,
        "brightness_mean": round(mean_brightness, 2),
        "contrast_std": round(contrast_std, 2),
        "blur_score": round(blur_score, 2) if blur_score is not None else None,
        "checks": checks,
        "is_usable": is_usable,
        "note": "Technical heuristics only; passing checks does not guarantee diagnostic image quality.",
    }
