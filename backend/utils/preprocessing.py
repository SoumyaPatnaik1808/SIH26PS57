import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Adaptive MMSE Lee despeckling filter.
    Identical math to original baseline to preserve exact YOLO weights alignment.
    """
    img_f = img.astype(np.float32)
    img_mean = uniform_filter(img_f, size)
    img_sqr_mean = uniform_filter(img_f**2, size)
    img_variance = np.maximum(img_sqr_mean - img_mean**2, 0.0)

    overall_variance = np.var(img_f)

    # Avoid division by zero
    weight = img_variance / (img_variance + overall_variance + 1e-8)

    filtered_img = img_mean + weight * (img_f - img_mean)
    return np.clip(filtered_img, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE with clipLimit=3.0 and tileGridSize=(8, 8) for YOLO visual extraction.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_sonar_pipeline(image_bytes: bytes):
    """
    Decoupled Dual-Stream Pipeline:
      1. visual_img (Lee + CLAHE) -> Feeds ONNX YOLO-OBB (100% identical to baseline)
      2. radiometric_img (Lee + Linear Stretch) -> Feeds Physics Decision Engine
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image from bytes")

    # Shared Base: Lee Despeckling
    filtered_img = lee_filter(img, size=7)

    # STREAM A (Visual AI Stream): Non-linear local CLAHE (Bit-for-bit identical for YOLO)
    visual_img = apply_clahe(filtered_img)

    # STREAM B (Radiometric Physics Stream): Global Linear Min-Max Normalization
    # Preserves true physical shadow-to-seabed intensity ratios without local CLAHE warping
    radiometric_img = cv2.normalize(
        filtered_img, np.empty_like(filtered_img), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    return visual_img, radiometric_img