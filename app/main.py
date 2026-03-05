from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import requests
import base64
import threading

from app.model import load_model, get_model
from app.utils import process_result, draw_boxes

app = FastAPI(
    title="Asset Detection API",
    version="2.0.0"
)

MODEL_PATH = "asset-x-120.pt"
MODEL_URL = "https://github.com/eldisja1/asset-discrepancy-nmsai/releases/download/v1.0.0/asset-x-120.pt"

# ==============================
# THREAD SAFETY CONFIG
# ==============================
model_lock = threading.Lock()          # Prevent race condition during inference
batch_semaphore = threading.Semaphore(10)  # Max 10 concurrent at a time


# ==============================
# Download Model Jika Tidak Ada
# ==============================
def download_model():
    print("Downloading model from GitHub Release...")
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code != 200:
        raise RuntimeError("Failed to download model")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Model downloaded successfully")


# ==============================
# Startup Event
# ==============================
@app.on_event("startup")
def startup_event():
    if not os.path.exists(MODEL_PATH):
        download_model()

    load_model(MODEL_PATH)
    print("Model loaded successfully")


# ==============================
# Health Check
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}


# ==============================
# Detection Endpoint
# ==============================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # ===== LIMIT 10 CONCURRENT =====
    with batch_semaphore:

        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        model = get_model()

        # ===== THREAD SAFE INFERENCE =====
        with model_lock:
            results = model(img)

        result = results[0]

        detections, object_count = process_result(result, model)

        # ==============================
        # Draw Bounding Box
        # ==============================
        output_image = draw_boxes(img.copy(), detections)

        # ==============================
        # Encode ke Base64
        # ==============================
        _, buffer = cv2.imencode(".jpg", output_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({
            "total_objects": sum(object_count.values()),
            "counts": object_count,
            "detections": detections,
            "image_base64": image_base64
        })