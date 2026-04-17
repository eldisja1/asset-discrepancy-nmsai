"""
utils.py

Contains:
- Custom class renaming
- Per-class threshold and rule-based filtering
- YOLO result post-processing
- Bounding box drawing
- Detection summary / detail formatting

Purpose:
To ensure the API output remains consistent with the final
validated notebook inference logic.
"""

import cv2

# ==============================
# 1. Custom Class Rename
# ==============================
rename_map = {
    "rru": "RRU",
    "panel_antenna": "Panel_Antenna",
    "microwave_dish": "Microwave_Dish"
}

# ==============================
# 2. Per-class Confidence Threshold
# ==============================
# Final thresholds based on validation tuning results
conf_threshold = {
    "Panel_Antenna": 0.25,
    "RRU": 0.50,
    "Microwave_Dish": 0.50,
}

# ==============================
# 3. Additional Geometry Filter
# ==============================
# Used to reduce false positives,
# especially for Panel_Antenna
min_width = {
    "Panel_Antenna": 40,
}

min_aspect_ratio = {
    "Panel_Antenna": 0.20,
}

# ==============================
# 4. Color Mapping (BGR)
# ==============================
# OpenCV uses BGR format, not RGB
color_map = {
    "Panel_Antenna": (139, 0, 0),   # Dark blue
    "RRU": (128, 0, 128),           # Dark purple
    "Microwave_Dish": (0, 100, 0)   # Dark green
}


# ==============================
# 5. Helper: Normalize Class Name
# ==============================
def get_class_name(box, model):
    """
    Get the original YOLO class name and rename it
    to match the project's final output format.

    Args:
        box: YOLO detection box
        model: Loaded YOLO model

    Returns:
        str: Normalized class name
    """
    class_id = int(box.cls[0])
    original_name = model.names[class_id]
    return rename_map.get(original_name, original_name)


# ==============================
# 6. Helper: Validate Detection
# ==============================
def is_above_threshold(box, model):
    """
    Determine whether a bounding box passes
    the final filtering rules.

    Applied filters:
    1. Per-class confidence threshold
    2. Minimum bounding box width (Panel_Antenna only)
    3. Minimum aspect ratio (Panel_Antenna only)

    Args:
        box: YOLO detection box
        model: Loaded YOLO model

    Returns:
        bool: True if the detection is valid, otherwise False
    """
    class_name = get_class_name(box, model)
    conf = float(box.conf[0])

    # ==============================
    # A. Confidence filter
    # ==============================
    if conf < conf_threshold.get(class_name, 0.25):
        return False

    # ==============================
    # B. Geometry filter for Panel_Antenna
    # ==============================
    if class_name == "Panel_Antenna":
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        aspect = w / h if h > 0 else 0

        if w < min_width.get(class_name, 0):
            return False

        if aspect < min_aspect_ratio.get(class_name, 0):
            return False

    return True


# ==============================
# 7. Process Detection Result
# ==============================
def process_result(result, model):
    """
    Process YOLO inference output into:
    - detection list
    - object count per class

    Only detections that pass the final filtering rules
    will be included.

    Args:
        result: YOLO inference result
        model: Loaded YOLO model

    Returns:
        tuple:
            detections (list[dict])
            object_count (dict)
    """
    detections = []
    object_count = {}

    if result.boxes is None:
        return detections, object_count

    for box in result.boxes:
        # ==============================
        # Apply final filtering logic
        # ==============================
        if not is_above_threshold(box, model):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_name = get_class_name(box, model)

        object_count[class_name] = object_count.get(class_name, 0) + 1

        detections.append({
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })

    return detections, object_count


# ==============================
# 8. Build Detection Detail List
# ==============================
def build_detection_detail(detections):
    """
    Build a detection detail list similar to the
    final notebook output.

    Output is sorted by:
    - class name
    - highest confidence first

    Args:
        detections (list[dict]): Filtered detection results

    Returns:
        list[dict]: Sorted detection detail list
    """
    detail_list = []

    for det in detections:
        detail_list.append({
            "class": det["class"],
            "confidence": det["confidence"]
        })

    detail_list.sort(key=lambda x: (x["class"], -x["confidence"]))
    return detail_list


# ==============================
# 9. Draw Bounding Boxes
# ==============================
def draw_boxes(image, detections):
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: OpenCV image (numpy array)
        detections: List of processed detection results

    Returns:
        image: Image with bounding boxes and labels
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]

        #label = f"{class_name} {confidence:.2f}"
        label = f"{class_name}"

        box_color = color_map.get(class_name, (255, 255, 255))
        text_color = (255, 255, 255)

        box_thickness = 5
        font_scale = 1.2
        font_thickness = 3

        # ==============================
        # Draw bounding box
        # ==============================
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            box_color,
            box_thickness
        )

        # ==============================
        # Measure text size
        # ==============================
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        y_label = max(y1 - 15, text_height + 10)

        # ==============================
        # Draw label background
        # ==============================
        cv2.rectangle(
            image,
            (x1, y_label - text_height - 10),
            (x1 + text_width + 10, y_label),
            box_color,
            -1
        )

        # ==============================
        # Draw label text
        # ==============================
        cv2.putText(
            image,
            label,
            (x1 + 5, y_label - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness
        )

    return image