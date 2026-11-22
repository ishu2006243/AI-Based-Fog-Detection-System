"""
Core Fog Detection Engine & Visualization for Django WebApp
"""

import os
import base64
from typing import Tuple, Dict, Any, Union
import cv2
import numpy as np

# Import the core analysis function from detect_fog.py
from detect_fog import analyze_glass_fog


def decode_image_input(image_input: Union[bytes, str, np.ndarray]) -> np.ndarray:
    """
    Decodes an image from bytes, base64 data-URL, file path, or numpy array.
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img

    if isinstance(image_input, str):
        # Base64 data URL
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            data = base64.b64decode(encoded)
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from base64 string.")
            return img
        # File path
        elif os.path.exists(image_input):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            return img
        else:
            raise FileNotFoundError(f"Image path not found: {image_input}")

    raise TypeError(f"Unsupported image type: {type(image_input)}")


def generate_annotated_preview(
    img_bgr: np.ndarray,
    status: str,
    metrics: Dict[str, Any],
) -> str:
    """
    Creates an annotated visualization showing detected droplets/features
    and returns a base64 data-URL string.
    """
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

    # Detect droplet keypoints to draw
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea = 4
    blob_params.maxArea = 1500
    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.20
    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.40
    blob_params.filterByInertia = True
    blob_params.minInertiaRatio = 0.15

    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(gray)

    # Highlight droplets with glowing rings if fogged, or subtle cyan if detected
    ring_color = (0, 220, 255) if status == "fogged" else (100, 200, 100)
    for kp in keypoints:
        center = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        radius = max(3, int(round(kp.size / 2.0)))
        cv2.circle(vis, center, radius, ring_color, 1, cv2.LINE_AA)

    # Top banner with status
    banner_height = 40
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner_bg = (30, 45, 15) if status == "clear" else (15, 30, 55)
    banner[:] = banner_bg

    text = f"Status: {status.upper()}  |  BG Sharpness: {metrics['bg_sharpness']:.1f}  |  Droplets: {metrics['blob_count']}"
    cv2.putText(
        banner,
        text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 160) if status == "clear" else (0, 200, 255),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack([banner, vis])

    # Encode to JPEG base64
    _, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def process_fog_detection(image_input: Union[bytes, str, np.ndarray]) -> Dict[str, Any]:
    """
    Full pipeline: decodes image, runs analysis, generates preview,
    and returns comprehensive response dict.
    """
    img_bgr = decode_image_input(image_input)

    # Run analysis
    status, metrics = analyze_glass_fog(img_bgr)

    # Generate annotated visualization
    annotated_preview = generate_annotated_preview(img_bgr, status, metrics)

    # Generate original preview
    h, w = img_bgr.shape[:2]
    max_prev = 800
    if max(h, w) > max_prev:
        s = max_prev / float(max(h, w))
        prev_mat = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        prev_mat = img_bgr

    _, prev_buf = cv2.imencode(".jpg", prev_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    original_preview = f"data:image/jpeg;base64,{base64.b64encode(prev_buf).decode('utf-8')}"

    return {
        "success": True,
        "status": status,
        "is_fogged": (status == "fogged"),
        "reasons": metrics["reasons"],
        "metrics": {
            "bg_sharpness": round(metrics["bg_sharpness"], 1),
            "res_ratio": round(metrics["res_ratio"], 2),
            "dyn_range": round(metrics["dyn_range"], 1),
            "contrast_std": round(metrics["contrast_std"], 1),
            "blob_count": metrics["blob_count"],
            "blob_density": round(metrics["blob_density"], 1),
            "meniscus_ratio": round(metrics["meniscus_ratio"] * 100, 1),
            "dcp_mean": round(metrics["dcp_mean"], 2),
            "sat_mean": round(metrics["sat_mean"], 1),
            "haze_index": round(metrics["haze_index"], 2),
        },
        "annotated_preview": annotated_preview,
        "original_preview": original_preview,
    }
