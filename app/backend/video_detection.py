
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import cv2

from inference import get_model, draw_detections_on_image, CLASS_NAMES


def process_video(
    input_path: str,
    output_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Dict[str, Any]:
    model = get_model()

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video stream from {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = str(output_path)
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize VideoWriter for {output_path}")

    frame_idx = 0
    total_counts = {"helmet": 0, "no_helmet": 0, "number_plate": 0, "rider": 0}
    max_counts_in_single_frame = {"helmet": 0, "no_helmet": 0, "number_plate": 0, "rider": 0}
    frames_with_violations = 0
    start_time = time.time()
    frame_samples: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )

            frame_detections: List[Dict[str, Any]] = []
            frame_counts = {"helmet": 0, "no_helmet": 0, "number_plate": 0, "rider": 0}

            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        coords = [float(v) for v in box.xyxy[0].tolist()]

                        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
                        if class_name in frame_counts:
                            frame_counts[class_name] += 1
                        else:
                            frame_counts[class_name] = 1

                        frame_detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 4),
                            "bbox": [round(c, 2) for c in coords]
                        })

            for k, v in frame_counts.items():
                total_counts[k] += v
                if v > max_counts_in_single_frame[k]:
                    max_counts_in_single_frame[k] = v

            if frame_counts["no_helmet"] > 0:
                frames_with_violations += 1

            annotated_frame = draw_detections_on_image(frame, frame_detections)

            hud_text = f"FRAME {frame_idx}/{total_frames} | HELMET: {frame_counts['helmet']} | NO HELMET: {frame_counts['no_helmet']} | RIDER: {frame_counts['rider']} | PLATES: {frame_counts['number_plate']}"
            cv2.rectangle(annotated_frame, (0, 0), (width, 36), (20, 24, 33), -1)
            cv2.putText(
                annotated_frame,
                hud_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            out.write(annotated_frame)

            if frame_idx % max(1, int(fps / 2)) == 0 or frame_idx == 1 or frame_idx == total_frames:
                frame_samples.append({
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 2),
                    "counts": frame_counts
                })

            if progress_callback:
                progress_callback(frame_idx, total_frames, (frame_idx / total_frames) * 100.0)

    finally:
        cap.release()
        out.release()

    elapsed = time.time() - start_time
    duration_seconds = round(frame_idx / fps, 2) if fps > 0 else 0.0

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_output_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(Path(output_path).with_name("h264_" + Path(output_path).name))
        ]
        res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if res.returncode == 0:
            converted_path = str(Path(output_path).with_name("h264_" + Path(output_path).name))
            os.replace(converted_path, output_path)
    except Exception:
        pass

    return {
        "status": "completed",
        "processed_frames": frame_idx,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": duration_seconds,
        "processing_time_seconds": round(elapsed, 2),
        "avg_fps_processing": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "total_detection_instances": total_counts,
        "peak_in_single_frame": max_counts_in_single_frame,
        "frames_with_violations": frames_with_violations,
        "violation_rate_pct": round((frames_with_violations / max(1, frame_idx)) * 100, 1),
        "timeline_samples": frame_samples[:50]
    }
