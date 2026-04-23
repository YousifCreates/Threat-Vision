import os

MODEL_ID = "weapondetection-xx3lz-tgh2t/2"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

DEFAULT_CONFIDENCE = 0.4
MIN_CONFIDENCE     = 0.1
MAX_CONFIDENCE     = 1.0

BASE_DIR    = "/media/yousifcreates/339b14bb-e7ae-4348-b1dd-b0b8a896b600/Portfolio/Deep Learning/ThreatVision/ThreatVision"
TEMP_DIR    = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
IMG_OUT     = os.path.join(OUTPUT_DIR, "images")
VID_OUT     = os.path.join(OUTPUT_DIR, "videos")
FRAMES_OUT  = os.path.join(OUTPUT_DIR, "frames")
LOG_PATH    = os.path.join(OUTPUT_DIR, "detection_log.json")

MODEL_DIR       = os.path.join(BASE_DIR, "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "weapondetection_v2", "cache_copy")

WEBCAM_FRAME_INTERVAL = 5
WEBCAM_INDEX          = 0

APP_TITLE    = "ThreatVision"
APP_SUBTITLE = "Real-time Weapon & Threat Detection"
THREAT_COLOR = (0, 0, 255)