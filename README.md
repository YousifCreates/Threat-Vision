# ThreatVision 🎯

Real-time weapon & threat detection powered by Roboflow Inference + Streamlit.

## Project Structure
```
ThreatVision/
├── app.py              # Streamlit UI (tabs: Image / Video / Webcam / Log)
├── backend_logic.py    # Inference, annotation, file I/O, logging
├── config.py           # All constants, paths, model settings
├── requirements.txt
├── temp/               # Temporary upload storage (auto-created)
└── output/
    ├── images/         # Processed images
    ├── videos/         # Processed videos
    └── frames/         # Webcam snapshots (one subfolder per session)
```

## Setup

### 1. Activate your conda environment
```bash
conda activate project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Roboflow API key
```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

### 4. Set your model ID in config.py
```python
MODEL_ID = "your-project/1"   # ← replace this
```

### 5. Run the app
```bash
streamlit run app.py
```

## Features
- **Image tab** – Upload image → annotated result + download
- **Video tab** – Upload video → frame-by-frame analysis + download
- **Webcam tab** – Live detection, auto-saves frames every 5s
- **Log tab** – Full detection history, summary stats, JSON export
- **Confidence slider** – Adjust in sidebar, applies to all modes
