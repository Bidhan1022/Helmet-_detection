

import os
import uuid
import base64
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import (
    get_model,
    get_model_info,
    predict_image,
    find_model_path,
    CLASS_NAMES
)
from video_detection import process_video

# Application directories
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = 150 * 1024 * 1024

app = FastAPI(
    title="Helmet Detection API",
    description="Backend API for detecting Helmets, Riders, No-Helmet violations, and Number Plates using pre-trained YOLOv8 (models/best.pt).",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print(" HELMET DETECTION FASTAPI SERVER STARTING (app/backend)")
    print("=" * 60)
    model_path = find_model_path()
    print(f"Target model location: {model_path}")

    if not model_path.exists():
        print("!" * 60)
        print(" [CRITICAL ERROR] models/best.pt WAS NOT FOUND!")
        print(f" Expected location: {model_path}")
        print(" Please place your trained YOLOv8 model at 'models/best.pt'.")
        print("!" * 60)
    else:
        try:
            model = get_model()
            print("[SUCCESS] Loaded YOLOv8 model from:", model_path)
            print("Classes detected:", CLASS_NAMES)
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}: {e}")
    print("=" * 60)


@app.get("/")
def root():
    model_info = get_model_info()
    return {
        "service": "Helmet Detection API",
        "status": "online",
        "version": "1.0.0",
        "model": model_info,
        "classes": CLASS_NAMES,
        "endpoints": {
            "image_prediction": "POST /predict/image",
            "video_prediction": "POST /predict/video",
            "status": "GET /status",
            "outputs": "GET /outputs/{filename}"
        }
    }


@app.get("/status")
def server_status():
    info = get_model_info()
    return {
        "status": "healthy",
        "model_ready": info["exists"],
        "model_loaded": info["is_loaded"],
        "model_path": info["model_file"],
        "classes": info["classes"],
        "allowed_images": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_videos": list(ALLOWED_VIDEO_EXTENSIONS)
    }


@app.post("/predict/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.1, le=0.9, description="IoU NMS threshold")
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image file format '{ext}'. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    model_path = find_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model 'models/best.pt' is missing. Please ensure your trained YOLOv8 model is placed at: {model_path}"
        )

    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds maximum limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB"
        )

    unique_id = uuid.uuid4().hex[:10]
    upload_path = UPLOADS_DIR / f"upload_{unique_id}{ext}"
    output_filename = f"pred_{unique_id}.jpg"
    output_path = OUTPUTS_DIR / output_filename

    try:
        with open(upload_path, "wb") as f:
            f.write(contents)

        annotated_bgr, detections, counts = predict_image(
            upload_path,
            conf_threshold=conf,
            iou_threshold=iou
        )

        cv2.imwrite(str(output_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        has_violations = counts.get("no_helmet", 0) > 0
        total_detections = sum(counts.values())

        return {
            "success": True,
            "filename": file.filename,
            "image_url": f"/outputs/{output_filename}",
            "image_base64": base64_image,
            "detections": detections,
            "counts": counts,
            "total_detections": total_detections,
            "has_violations": has_violations,
            "conf_threshold": conf,
            "iou_threshold": iou
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )
    finally:
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass


@app.post("/predict/video")
async def predict_video_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.1, le=0.9, description="IoU NMS threshold")
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video file format '{ext}'. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    model_path = find_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model 'models/best.pt' is missing. Please ensure your trained YOLOv8 model is placed at: {model_path}"
        )

    contents = await file.read()
    if len(contents) > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video size exceeds maximum limit of {MAX_VIDEO_SIZE_BYTES // (1024 * 1024)}MB"
        )

    unique_id = uuid.uuid4().hex[:10]
    upload_video_path = UPLOADS_DIR / f"upload_video_{unique_id}{ext}"
    output_filename = f"pred_video_{unique_id}.mp4"
    output_video_path = OUTPUTS_DIR / output_filename

    try:
        with open(upload_video_path, "wb") as f:
            f.write(contents)

        metrics = process_video(
            str(upload_video_path),
            str(output_video_path),
            conf_threshold=conf,
            iou_threshold=iou
        )

        return {
            "success": True,
            "filename": file.filename,
            "video_url": f"/outputs/{output_filename}",
            "output_filename": output_filename,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "metrics": metrics,
            "counts": metrics.get("total_detection_instances", {}),
            "peak_counts": metrics.get("peak_in_single_frame", {}),
            "total_frames": metrics.get("total_frames", 0),
            "fps": metrics.get("fps", 0),
            "duration_seconds": metrics.get("duration_seconds", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}"
        )
    finally:
        if upload_video_path.exists():
            try:
                upload_video_path.unlink()
            except Exception:
                pass


@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
