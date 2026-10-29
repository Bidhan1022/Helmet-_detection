import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "number_plate",
    3: "rider"
}

CLASS_COLORS = {
    "helmet": (34, 197, 94),        # Green
    "no_helmet": (68, 68, 239),     # Red
    "number_plate": (11, 158, 245), # Amber
    "rider": (246, 130, 59)         # Blue
}

DEFAULT_COLOR = (200, 200, 200)

_model_instance: Optional[YOLO] = None
_model_path_resolved: Optional[Path] = None


def find_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p
    
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "models" / "best.pt"

    
def get_model() -> YOLO:
    global _model_instance, _model_path_resolved

    if _model_instance is not None:
        return _model_instance

    model_path = find_model_path()
    _model_path_resolved = model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please ensure your trained YOLOv8 model is placed at 'models/best.pt' "
            "as specified in the project requirements."
        )

    print(f"[Helmet Detection Backend] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    
    _model_instance = model
    return _model_instance


def get_model_info() -> Dict[str, Any]:
    global _model_path_resolved
    path = find_model_path()
    exists = path.exists()
    return {
        "model_file": str(path),
        "exists": exists,
        "is_loaded": _model_instance is not None,
        "classes": CLASS_NAMES,
        "device": "CPU / Available PyTorch Device"
    }


def draw_detections_on_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    scale = max(0.45, min(width, height) / 1000.0)
    thickness = max(2, int(scale * 3))

    for det in detections:
        box = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = det["class"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_name.upper()}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = scale * 0.8
        font_thickness = max(1, int(thickness / 1.5))

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = y1 if y1 - text_h - baseline - 6 >= 0 else y1 + text_h + baseline + 6
        label_x2 = min(width, x1 + text_w + 10)

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        text_baseline_y = label_y2 - baseline - 3 if y1 - text_h - baseline - 6 >= 0 else label_y2 - 4
        cv2.putText(
            annotated,
            label,
            (x1 + 5, text_baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated


def predict_image(
    image_input: Any,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    model = get_model()

    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise ValueError(f"Could not read image file from {image_input}")
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    detections: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "helmet": 0,
        "no_helmet": 0,
        "number_plate": 0,
        "rider": 0
    }

    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = [float(v) for v in box.xyxy[0].tolist()]

                class_name = CLASS_NAMES.get(cls_id, str(cls_id))
                if class_name in counts:
                    counts[class_name] += 1
                else:
                    counts[class_name] = 1

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(c, 2) for c in coords]
                })

    annotated_image = draw_detections_on_image(img_bgr, detections)
    return annotated_image, detections, counts
