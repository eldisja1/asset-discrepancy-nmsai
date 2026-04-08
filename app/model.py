"""
model.py

Handles YOLO model lifecycle:
- Downloading the model if missing
- Loading the model into memory
- Returning the loaded model instance

Purpose:
To ensure the model is downloaded (if necessary) and loaded
only once during application startup, then reused for all requests.
"""

from ultralytics import YOLO
import os
import urllib.request

# Global model instance (singleton-style)
MODEL = None


def download_model_if_missing(model_path: str, model_url: str):
    """
    Download the model file if it does not exist locally.

    Args:
        model_path (str): Local path where the model should be stored.
        model_url (str): Remote URL to download the model from.
    """
    if not os.path.exists(model_path):
        print("Model file not found. Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully")


def load_model(model_path: str, model_url: str = None):
    """
    Load the YOLO model into memory.

    If the model file does not exist locally and a model URL is provided,
    it will be downloaded first.

    Args:
        model_path (str): Path to the YOLO .pt model file.
        model_url (str, optional): URL to download the model if missing.

    Returns:
        YOLO: Loaded YOLO model instance.

    Raises:
        RuntimeError: If the model file does not exist and no URL is provided.
    """
    global MODEL

    if not os.path.exists(model_path):
        if model_url:
            download_model_if_missing(model_path, model_url)
        else:
            raise RuntimeError(
                f"Model file not found: {model_path} and no download URL was provided."
            )

    MODEL = YOLO(model_path)
    return MODEL


def get_model():
    """
    Return the loaded YOLO model instance.

    Raises:
        RuntimeError: If the model has not been loaded yet.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")
    return MODEL