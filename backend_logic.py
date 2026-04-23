"""
backend_logic.py
Handles all inference, annotation, file I/O, and logging for ThreatVision.
"""

import os
import cv2
import json
import time
import tempfile
import numpy as np
import supervision as sv
from datetime import datetime
from inference import get_model

import config

# ── Ensure output dirs exist ───────────────────────────────────────────────────
for _dir in [config.TEMP_DIR, config.IMG_OUT, config.VID_OUT, config.FRAMES_OUT]:
    os.makedirs(_dir, exist_ok=True)

# ── Model loader (cached after first call) ─────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is None:
        if not config.ROBOFLOW_API_KEY:
            raise EnvironmentError(
                "ROBOFLOW_API_KEY is not set. "
                "Run: export ROBOFLOW_API_KEY=your_key"
            )
        # Point inference cache at local copy — prevents re-download
        os.environ["ROBOFLOW_MODEL_CACHE_DIR"] = config.MODEL_SAVE_PATH
        _model = get_model(model_id=config.MODEL_ID)
    return _model


# ── Everything below is unchanged ─────────────────────────────────────────────

def _make_annotators():
    box   = sv.BoxAnnotator(color=sv.Color(r=220, g=38, b=38), thickness=2)
    label = sv.LabelAnnotator(
        color=sv.Color(r=220, g=38, b=38),
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
    )
    return box, label


def _annotate(frame: np.ndarray, detections: sv.Detections, labels: list) -> np.ndarray:
    box_ann, lbl_ann = _make_annotators()
    frame = box_ann.annotate(scene=frame.copy(), detections=detections)
    frame = lbl_ann.annotate(scene=frame, detections=detections, labels=labels)
    return frame


def _build_labels(results, detections: sv.Detections) -> list:
    labels = []
    for i in range(len(detections)):
        cls  = detections.data.get("class_name", ["unknown"] * len(detections))[i]
        conf = detections.confidence[i] if detections.confidence is not None else 0.0
        labels.append(f"{cls}  {conf:.0%}")
    return labels


def _load_log() -> list:
    if os.path.exists(config.LOG_PATH):
        with open(config.LOG_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_log(entries: list):
    with open(config.LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def append_log(source: str, filename: str, detections: sv.Detections):
    entries = _load_log()
    class_names = list(detections.data.get("class_name", []))
    entries.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":    source,
        "file":      filename,
        "count":     len(detections),
        "classes":   class_names,
    })
    _save_log(entries)


def get_log() -> list:
    return _load_log()


def clear_log():
    _save_log([])


def process_image(uploaded_file, confidence: float) -> dict:
    model = load_model()

    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=config.TEMP_DIR) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    image = cv2.imread(tmp_path)
    if image is None:
        raise ValueError(f"Could not read image: {tmp_path}")

    results    = model.infer(image, confidence=confidence)[0]
    detections = sv.Detections.from_inference(results)
    labels     = _build_labels(results, detections)
    annotated  = _annotate(image, detections, labels)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"result_{ts}{suffix}"
    out_path = os.path.join(config.IMG_OUT, out_name)
    cv2.imwrite(out_path, annotated)

    os.remove(tmp_path)
    append_log("image", uploaded_file.name, detections)

    return {
        "output_path": out_path,
        "count":       len(detections),
        "classes":     list(detections.data.get("class_name", [])),
        "labels":      labels,
    }


def _open_video_writer(out_path: str, fps: float, width: int, height: int):
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] VideoWriter opened with codec: {codec}")
            return writer, out_path
        print(f"[WARNING] Codec {codec} unavailable, trying next...")

    avi_path = out_path.replace(".mp4", ".avi")
    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    writer   = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    if writer.isOpened():
        print(f"[INFO] VideoWriter opened with XVID (.avi)")
        return writer, avi_path

    raise RuntimeError(
        f"All codecs failed.\nPath: {out_path}\n"
        f"Resolution: {width}x{height}, FPS: {fps}\n"
        "Try: sudo apt install ffmpeg"
    )


def _reencode_h264(src_path: str) -> str:
    import subprocess, shutil

    if shutil.which("ffmpeg") is None:
        print("[WARNING] ffmpeg not found — video may not play in browser. Install: sudo apt install ffmpeg")
        return src_path

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    tmp_out = src_path.replace(".mp4", f"_h264_{ts}.mp4").replace(".avi", f"_h264_{ts}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-acodec", "aac",
        tmp_out
    ]

    print(f"[INFO] Re-encoding to H.264: {tmp_out}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and os.path.exists(tmp_out):
        os.remove(src_path)
        print(f"[INFO] Re-encode successful: {tmp_out}")
        return tmp_out
    else:
        print(f"[WARNING] FFmpeg re-encode failed:\n{result.stderr}")
        return src_path


def process_video(uploaded_file, confidence: float, progress_callback=None) -> dict:
    model = load_model()

    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=config.TEMP_DIR) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(config.VID_OUT, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.VID_OUT, f"result_{ts}.mp4")

    print(f"[INFO] Output dir  : {config.VID_OUT}")
    print(f"[INFO] Dir exists  : {os.path.exists(config.VID_OUT)}")
    print(f"[INFO] Output path : {out_path}")
    print(f"[INFO] Resolution  : {width}x{height}  FPS: {fps}  Frames: {total}")

    writer, out_path = _open_video_writer(out_path, fps, width, height)

    all_classes   = []
    total_detects = 0
    frame_idx     = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results    = model.infer(frame, confidence=confidence)[0]
            detections = sv.Detections.from_inference(results)
            labels     = _build_labels(results, detections)
            annotated  = _annotate(frame, detections, labels)

            writer.write(annotated)
            total_detects += len(detections)
            all_classes   += list(detections.data.get("class_name", []))
            frame_idx     += 1

            if progress_callback and total > 0:
                progress_callback(frame_idx / total)
    finally:
        cap.release()
        writer.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"[INFO] File exists  : {os.path.exists(out_path)}")
    print(f"[INFO] File size    : {os.path.getsize(out_path) if os.path.exists(out_path) else 0} bytes")

    out_path = _reencode_h264(out_path)
    append_log("video", uploaded_file.name, sv.Detections.empty())

    return {
        "output_path":      out_path,
        "total_detects":    total_detects,
        "classes":          all_classes,
        "frames_processed": frame_idx,
    }


def webcam_frame(frame: np.ndarray, confidence: float, session_dir: str, last_save_time: float) -> dict:
    model = load_model()

    results    = model.infer(frame, confidence=confidence)[0]
    detections = sv.Detections.from_inference(results)
    labels     = _build_labels(results, detections)
    annotated  = _annotate(frame, detections, labels)

    now        = time.time()
    saved_path = None
    if now - last_save_time >= config.WEBCAM_FRAME_INTERVAL:
        os.makedirs(session_dir, exist_ok=True)
        ts         = datetime.now().strftime("%H%M%S")
        saved_path = os.path.join(session_dir, f"frame_{ts}.jpg")
        cv2.imwrite(saved_path, annotated)
        last_save_time = now
        append_log("webcam", saved_path, detections)

    return {
        "annotated":      annotated,
        "detections":     detections,
        "labels":         labels,
        "count":          len(detections),
        "classes":        list(detections.data.get("class_name", [])),
        "last_save_time": last_save_time,
        "saved_path":     saved_path,
    }


def new_webcam_session_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(config.FRAMES_OUT, f"session_{ts}")