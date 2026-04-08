"""
main.py

FastAPI entry point for the Asset Discrepancy Detection API.

Features:
- Health check
- Image upload
- YOLO inference
- Final detection filtering
- Bounding box rendering
- Base64 output image response

Endpoints:
- GET  /health
- POST /detect
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import base64
import threading

from app.model import load_model, get_model
from app.utils import process_result, draw_boxes

app = FastAPI(
    title="Asset Discrepancy Detection API",
    description="API for detecting BTS assets such as Panel Antenna, RRU, and Microwave Dish.",
    version="1.1.0"
)

# ==============================
# 1. Model Path
# ==============================
MODEL_PATH = "asset-11l-cp03-180.pt"
MODEL_URL = "https://github.com/eldisja1/Asset-Discrepancy-NMSAI/releases/download/v1.1.0/asset-11l-cp03-180.pt"

# ==============================
# 2. Thread Safety Config
# ==============================
# Prevent race conditions during inference
model_lock = threading.Lock()

# Limit the number of concurrent active inferences
batch_semaphore = threading.Semaphore(10)


# ==============================
# 3. Startup Event
# ==============================
@app.on_event("startup")
def startup_event():
    """
    Load the model once when FastAPI starts.
    If the model file is missing, it will be downloaded automatically.
    """
    load_model(MODEL_PATH, MODEL_URL)
    print("Model loaded successfully")


# ==============================
# 4. Health Check
# ==============================
@app.get("/health")
def health():
    """
    Simple endpoint to verify that the API is running.
    """
    return {"status": "ok"}


# ==============================
# 5. Detection Endpoint
# ==============================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and run asset detection.

    Supported file types:
    - image/jpeg
    - image/png
    - image/jpg

    Returns:
        JSONResponse:
        {
            "total_objects": int,
            "counts": dict,
            "detections": list,
            "image_base64": str
        }
    """

    # ==============================
    # A. Validate Content-Type
    # ==============================
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported formats: jpg, jpeg, png."
        )

    # ==============================
    # B. Limit concurrent inference
    # ==============================
    with batch_semaphore:
        try:
            # ==============================
            # C. Read uploaded file as OpenCV image
            # ==============================
            contents = await file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            # ==============================
            # D. Get loaded model
            # ==============================
            model = get_model()

            # ==============================
            # E. Thread-safe inference
            # ==============================
            # Use low confidence first, then apply manual final filtering
            with model_lock:
                results = model(img, conf=0.25, verbose=False)

            result = results[0]

            # ==============================
            # F. Process detection result
            # ==============================
            detections, object_count = process_result(result, model)

            # ==============================
            # G. Draw bounding boxes
            # ==============================
            output_image = draw_boxes(img.copy(), detections)

            # ==============================
            # H. Encode output image to base64
            # ==============================
            success, buffer = cv2.imencode(".jpg", output_image)
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to encode output image"
                )

            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # ==============================
            # I. Return clean final response
            # ==============================
            return JSONResponse({
                "total_objects": sum(object_count.values()),
                "counts": object_count,
                "detections": detections,
                "image_base64": image_base64
            })

        except HTTPException:
            raise

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )