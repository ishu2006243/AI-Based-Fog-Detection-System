#!/usr/bin/env python3
"""
Glass Fog Detection Script

Detects whether an image has glass fogged on it (condensation droplets,
dew beading, steam haze, or milky veiling) vs clear glass looking at a scene.

Outputs:
  - 'clear'  : The glass/image is clear.
  - 'fogged' : Fog, condensation droplets, or steam detected on the glass.

Usage:
  # Check a specific image
  python detect_fog.py image3.png
  python detect_fog.py image4.png

  # Check all sample images in the folder
  python detect_fog.py

  # With detailed metrics
  python detect_fog.py image3.png --verbose
"""

import sys
import os
import argparse
import glob
from typing import Tuple, Dict, Any, Union
import cv2
import numpy as np

try:
    from detector.serial_controller import send_fog_alert
except ImportError:
    send_fog_alert = None


def analyze_glass_fog(
    image_input: Union[str, np.ndarray],
    max_dimension: int = 1000,
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyzes an image to determine whether the glass is fogged or clear.

    Args:
        image_input: File path (str) or loaded BGR/grayscale numpy image array.
        max_dimension: Max width/height to resize for consistent scale analysis.

    Returns:
        status: 'fogged' or 'clear'
        metrics: Dictionary containing calculated optical and texture features.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img_mat = cv2.imread(image_input)
        if img_mat is None:
            raise ValueError(f"Could not decode image from file: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_mat = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Normalize image dimensions for consistent droplet and frequency scale
    h, w = img_mat.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        img_mat = cv2.resize(
            img_mat,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        h, w = img_mat.shape[:2]

    total_pixels = float(h * w)

    # Convert color spaces
    if len(img_mat.shape) == 3:
        gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
        bgr = img_mat
    else:
        gray = img_mat
        bgr = cv2.cvtColor(img_mat, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------------------
    # 1. Background Scene Sharpness vs Surface Droplet Residual
    # When glass is fogged with droplets, the droplets sit on the glass surface
    # while the underlying scene behind the glass is diffused/blurred.
    # Applying a median filter removes high-frequency surface droplets,
    # revealing the sharpness of the underlying background scene.
    # -------------------------------------------------------------
    med = cv2.medianBlur(gray, 7)
    med_lap_var = float(cv2.Laplacian(med, cv2.CV_64F).var())

    droplet_residual = cv2.absdiff(gray, med)
    mean_droplet_residual = float(np.mean(droplet_residual))
    # Ratio of surface droplet texture to background sharpness
    res_ratio = float(mean_droplet_residual / (np.sqrt(med_lap_var) + 1e-4))

    # Dynamic Range & Contrast
    p5, p95 = np.percentile(gray, [5, 95])
    dyn_range = float(p95 - p5)
    contrast_std = float(np.std(gray))

    # -------------------------------------------------------------
    # 2. Condensation Droplets & Meniscus Analysis
    # Water droplets on glass form fluid domes with a specular highlight
    # (bright peak) inside a darker boundary (meniscus ring).
    # -------------------------------------------------------------
    k_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    dilated_blackhat = cv2.dilate(
        blackhat,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    meniscus_match = (tophat > 12) & (dilated_blackhat > 12)
    meniscus_ratio = float(np.count_nonzero(meniscus_match) / total_pixels)

    # SimpleBlobDetector for discrete droplet beads
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
    blob_count = len(keypoints)
    blob_density = float(blob_count / (total_pixels / 10000.0))  # per 100x100 block

    # -------------------------------------------------------------
    # 3. Haze / Steam Veiling Light Analysis
    # Fine steam or micro-vapor scatters light, washing out contrast
    # and elevating the Dark Channel Prior (DCP).
    # -------------------------------------------------------------
    min_rgb = np.min(bgr, axis=2)
    dcp_patch_size = 15
    dark_channel = cv2.erode(
        min_rgb,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dcp_patch_size, dcp_patch_size)),
    )
    dcp_mean = float(np.mean(dark_channel) / 255.0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))

    if contrast_std > 8.0:
        haze_index = float(dcp_mean / ((contrast_std / 100.0) + 1e-4))
    else:
        haze_index = 0.0

    # -------------------------------------------------------------
    # 4. Decision Logic
    # -------------------------------------------------------------
    is_fogged = False
    reasons = []

    # Check 1: Clear glass looking at a sharp, rich background scene
    # When glass is clear, the scene behind it is sharp (high med_lap_var)
    # and has wide dynamic range (deep shadows to bright lights).
    if med_lap_var >= 100.0 and dyn_range >= 150.0 and res_ratio < 0.70:
        is_fogged = False

    # Check 2: Condensation droplets on glass
    # Droplets present on surface with diffuse/obscured background
    elif res_ratio >= 0.75 and mean_droplet_residual >= 1.8 and (blob_density >= 10.0 or meniscus_ratio >= 0.03):
        is_fogged = True
        reasons.append(
            f"Glass condensation droplets (surface-to-bg ratio: {res_ratio:.2f}, "
            f"droplet residual: {mean_droplet_residual:.1f}, density: {blob_density:.1f}/10k px)"
        )

    # Check 3: Dense micro-condensation layer / frost (high meniscus texture with blurred background)
    elif meniscus_ratio >= 0.15 and med_lap_var < 100.0:
        is_fogged = True
        reasons.append(
            f"Dense micro-condensation layer on glass (meniscus texture: {meniscus_ratio*100:.1f}%, "
            f"bg sharpness: {med_lap_var:.1f})"
        )

    # Check 4: Steamy / milky fog veil (milky haze, compressed dynamic range, but non-flat image)
    elif contrast_std >= 10.0 and haze_index >= 1.6 and dcp_mean >= 0.45 and dyn_range < 150.0 and \
         (blob_count >= 15 or mean_droplet_residual >= 1.5):
        is_fogged = True
        reasons.append(
            f"Steamy fog veil (haze index: {haze_index:.2f}, DCP: {dcp_mean:.2f}, "
            f"dyn range: {dyn_range:.0f})"
        )

    else:
        # Default clear
        is_fogged = False

    status = "fogged" if is_fogged else "clear"

    metrics = {
        "is_fogged": is_fogged,
        "status": status,
        "reasons": reasons,
        "bg_sharpness": med_lap_var,
        "surface_residual": mean_droplet_residual,
        "res_ratio": res_ratio,
        "dyn_range": dyn_range,
        "contrast_std": contrast_std,
        "dcp_mean": dcp_mean,
        "sat_mean": sat_mean,
        "haze_index": haze_index,
        "blob_count": blob_count,
        "blob_density": blob_density,
        "meniscus_ratio": meniscus_ratio,
    }

    return status, metrics


def detect_fog(image_path: str) -> str:
    """Convenience helper returning 'fogged' or 'clear'."""
    status, _ = analyze_glass_fog(image_path)
    return status


def is_fogged(image_path: str) -> bool:
    """Convenience helper returning True if fogged, False if clear."""
    status, _ = analyze_glass_fog(image_path)
    return status == "fogged"


def main():
    parser = argparse.ArgumentParser(
        description="Detect whether an image has glass fogged on it. Prints 'clear' or 'fogged'."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Path to one or more image files. If omitted, checks all sample images in the current folder.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed diagnostic metrics and detection reasons.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Send '*1' over serial via pySerial if fog is detected.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port to use (e.g. /dev/cu.usbserial-*, COM3). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # Determine target images
    target_files = args.images
    if not target_files:
        # Scan current directory for sample images
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        found = []
        for ext in extensions:
            found.extend(glob.glob(ext))
        # Filter out hidden files
        target_files = [f for f in sorted(found) if not os.path.basename(f).startswith(".")]

        if not target_files:
            print("No images found in the current directory.")
            sys.exit(1)

    # Process images
    single_image_mode = len(target_files) == 1 and bool(args.images)

    for path in target_files:
        try:
            status, metrics = analyze_glass_fog(path)

            serial_info = None
            if status == "fogged" and (args.serial or os.environ.get("SERIAL_PORT")):
                if send_fog_alert:
                    sent, msg = send_fog_alert("*1", port=args.port)
                    serial_info = msg
                else:
                    serial_info = "pyserial not available"

            if single_image_mode and not args.verbose:
                # Direct simple output: print 'clear' or 'fogged'
                print(status)
                if serial_info and args.serial:
                    print(f"Serial: {serial_info}")
            else:
                # Multi-image or verbose display
                if args.verbose:
                    print(f"[{os.path.basename(path)}] -> {status.upper()}")
                    if metrics["reasons"]:
                        for r in metrics["reasons"]:
                            print(f"  * {r}")
                    else:
                        print("  * Surface is clear (scene behind glass is sharp and unobstructed)")
                    print(
                        f"  [Metrics] BG Sharpness: {metrics['bg_sharpness']:.1f}, "
                        f"Surface-to-BG Ratio: {metrics['res_ratio']:.2f}, "
                        f"Dyn Range: {metrics['dyn_range']:.1f}, "
                        f"Droplets: {metrics['blob_count']}, "
                        f"Density: {metrics['blob_density']:.1f}/10k px, "
                        f"Meniscus: {metrics['meniscus_ratio']*100:.1f}%, "
                        f"HazeIdx: {metrics['haze_index']:.2f}\n"
                    )
                else:
                    print(f"{os.path.basename(path)}: {status}")

        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
