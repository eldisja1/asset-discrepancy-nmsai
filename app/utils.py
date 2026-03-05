import cv2

# ==============================
# Custom Class Rename
# ==============================
rename_map = {
    "rru": "RRU",
    "panel_antenna": "Panel_Antenna",
    "microwave_dish": "Microwave_Dish"
}

# ==============================
# Color Mapping (BGR)
# ==============================
color_map = {
    "Panel_Antenna": (139, 0, 0),
    "RRU": (128, 0, 128),
    "Microwave_Dish": (0, 100, 0)
}

# ==============================
# Process Detection Result
# ==============================
def process_result(result, model):
    detections = []
    object_count = {}

    if result.boxes is None:
        return detections, object_count

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])

        original_name = model.names[class_id]
        class_name = rename_map.get(original_name, original_name)

        object_count[class_name] = object_count.get(class_name, 0) + 1

        detections.append({
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })

    return detections, object_count


# ==============================
# Draw Bounding Boxes
# ==============================
def draw_boxes(image, detections):

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        label = f'{class_name} {det["confidence"]:.2f}'

        box_color = color_map.get(class_name, (255, 255, 255))
        text_color = (255, 255, 255)

        box_thickness = 5
        font_scale = 1.2
        font_thickness = 3

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            box_color,
            box_thickness
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        y_label = max(y1 - 15, text_height + 10)

        cv2.rectangle(
            image,
            (x1, y_label - text_height - 10),
            (x1 + text_width + 10, y_label),
            box_color,
            -1
        )

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