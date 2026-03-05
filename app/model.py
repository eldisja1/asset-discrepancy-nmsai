from ultralytics import YOLO

model = None

def load_model(model_path: str):
    global model
    model = YOLO(model_path)
    return model

def get_model():
    if model is None:
        raise RuntimeError("Model not loaded")
    return model