import os
import json
import glob
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .fog_engine import process_fog_detection
from .serial_controller import send_fog_alert, get_serial_status

BASE_DIR = settings.BASE_DIR


def index(request):
    """
    Renders the modern Glass Fog Detection Dashboard.
    """
    # Find sample images in root directory
    sample_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in glob.glob(str(BASE_DIR / ext)):
            fname = os.path.basename(p)
            if not fname.startswith("."):
                sample_files.append(fname)

    sample_files = sorted(sample_files)
    context = {
        "sample_files": sample_files,
    }
    return render(request, "detector/index.html", context)


@csrf_exempt
def detect_api(request):
    """
    API endpoint accepting image upload, base64 camera snapshot, or sample image name.
    If fog is detected, sends "*1" over serial via pySerial.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Only POST method is allowed."}, status=405)

    try:
        result = None

        # 1. Check if multipart file upload
        if "image" in request.FILES:
            uploaded_file = request.FILES["image"]
            image_bytes = uploaded_file.read()
            result = process_fog_detection(image_bytes)

        # 2. Check JSON payload (webcam base64 or sample name)
        elif request.body:
            body = request.body.decode("utf-8")
            data = json.loads(body)

            # Camera snapshot base64
            if "image_data" in data and data["image_data"]:
                image_data = data["image_data"]
                result = process_fog_detection(image_data)

            # Sample image request
            elif "sample_name" in data and data["sample_name"]:
                sample_name = os.path.basename(data["sample_name"])
                sample_path = BASE_DIR / sample_name
                if not sample_path.exists():
                    return JsonResponse({"success": False, "error": f"Sample not found: {sample_name}"}, status=404)
                result = process_fog_detection(str(sample_path))

        if result is None:
            return JsonResponse({"success": False, "error": "No image data provided. Upload a file or capture via camera."}, status=400)

        # Trigger serial alert if fog is detected
        if result.get("is_fogged"):
            sent, msg = send_fog_alert("*1")
            result["serial"] = {"sent": sent, "message": msg}

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def serial_status_api(request):
    """
    Returns diagnostic information about the serial port configuration.
    """
    return JsonResponse(get_serial_status())


def sample_image(request, filename):
    """
    Serves a sample image file from the workspace.
    """
    safe_name = os.path.basename(filename)
    file_path = BASE_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return HttpResponseNotFound("Image not found")

    ext = safe_name.split(".")[-1].lower()
    content_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        return HttpResponse(f.read(), content_type=content_type)
