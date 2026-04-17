"""
main.py

FastAPI entry point for Asset Discrepancy Detection API.

Supports:
- Image upload
- URL (image & video)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import cv2
import base64
import threading
import time

from app.model import load_model, get_model
from app.utils import process_result, draw_boxes

app = FastAPI(
    title="Asset Discrepancy Detection API",
    version="1.2.0"
)

# ================= CONFIG =================
MODEL_PATH = "asset-11l-cp03-180.pt"
MODEL_URL = "https://github.com/eldisja1/Asset-Discrepancy-NMSAI/releases/download/v1.2.0/asset-11l-cp03-180.pt"

CONF_THRESHOLD = 0.25
INTERVAL_SEC = 1.0
MAX_SHOWN = 3
MAX_VIDEO_SECONDS = 10
MAX_FRAMES = 30

# ================= THREAD SAFETY =================
model_lock = threading.Lock()
batch_semaphore = threading.Semaphore(10)

# ================= STARTUP =================
@app.on_event("startup")
def startup_event():
    load_model(MODEL_PATH, MODEL_URL)
    print("Model loaded successfully")

# ================= REQUEST =================
class URLRequest(BaseModel):
    file_url: str

# ================= HEALTH =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= IMAGE PROCESS =================
def process_image(img):
    model = get_model()

    with model_lock:
        results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

    detections, object_count = process_result(results, model)
    output = draw_boxes(img.copy(), detections)

    _, buffer = cv2.imencode(".jpg", output)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return detections, object_count, image_base64

# ================= VIDEO PROCESS =================
def process_video(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return False, {}, []

    model = get_model()

    best_frames = {}
    last_bucket = -1
    frame_count = 0
    start_time = time.time()
    found_any = False

    last_frame = None

    try:
        while cap.isOpened():

            if time.time() - start_time > MAX_VIDEO_SECONDS:
                break

            if frame_count >= MAX_FRAMES:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            last_frame = frame.copy()

            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            bucket = int(ms // (INTERVAL_SEC * 1000))

            if bucket == last_bucket:
                continue
            last_bucket = bucket

            with model_lock:
                results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

            detections, object_count = process_result(results, model)

            if len(detections) > 0:
                found_any = True

                score = sum([d["confidence"] for d in detections])

                if bucket not in best_frames or score > best_frames[bucket]["score"]:
                    drawn = draw_boxes(frame.copy(), detections)

                    best_frames[bucket] = {
                        "score": score,
                        "frame": drawn,
                        "counts": object_count
                    }

    finally:
        cap.release()

    frames_sorted = sorted(
        best_frames.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )[:MAX_SHOWN]

    images = []
    final_counts = {}

    for _, data in frames_sorted:
        _, buffer = cv2.imencode(".jpg", data["frame"])
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        images.append(img_b64)

        for k, v in data["counts"].items():
            final_counts[k] = final_counts.get(k, 0) + v

    if not found_any and last_frame is not None:
        fallback_frame = last_frame.copy()

        text = "CLEAR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        color = (0, 255, 0)

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = fallback_frame.shape[1] - text_w - 20
        y = 40

        cv2.putText(
            fallback_frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness
        )

        _, buffer = cv2.imencode(".jpg", fallback_frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        images = [img_b64]

    return found_any, final_counts, images

# ================= ENDPOINT: UPLOAD =================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    with batch_semaphore:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        detections, counts, image_base64 = process_image(img)

        return JSONResponse({
            "total_objects": sum(counts.values()),
            "counts": counts,
            "detections": detections,
            "image_base64": image_base64
        })

# ================= ENDPOINT: URL =================
@app.post("/detect-url")
def detect_url(data: URLRequest):

    if not data.file_url:
        raise HTTPException(status_code=400, detail="URL required")

    try:
        url = data.file_url.lower()

        # ================= NEW: WEBM HANDLING =================
        if url.endswith(".webm"):
            cap = cv2.VideoCapture(data.file_url, cv2.CAP_FFMPEG)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise HTTPException(status_code=400, detail="Cannot read webm URL")

            detections, counts, image_base64 = process_image(frame)

            return {
                "type": "webm_frame",
                "total_objects": sum(counts.values()),
                "counts": counts,
                "detections": detections,
                "image_base64": image_base64
            }

        # image
        if url.endswith((".jpg", ".jpeg", ".png")):
            cap = cv2.VideoCapture(data.file_url)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise HTTPException(status_code=400, detail="Cannot read image URL")

            detections, counts, image_base64 = process_image(frame)

            return {
                "type": "image",
                "total_objects": sum(counts.values()),
                "counts": counts,
                "detections": detections,
                "image_base64": image_base64
            }

        # video
        found, counts, images = process_video(data.file_url)

        return {
            "type": "video",
            "status": "OBJECT DETECTED" if found else "CLEAR",
            "total_objects": sum(counts.values()),
            "counts": counts,
            "total_images": len(images),
            "images_base64": images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))