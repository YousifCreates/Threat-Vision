"""
app.py  –  ThreatVision Streamlit Frontend
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from collections import Counter

import config
import backend_logic as bl

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
  }

  /* Background */
  .stApp {
    background-color: #0a0c0f;
    background-image:
      repeating-linear-gradient(0deg,   transparent, transparent 39px, rgba(220,38,38,0.04) 40px),
      repeating-linear-gradient(90deg,  transparent, transparent 39px, rgba(220,38,38,0.04) 40px);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d0f13;
    border-right: 1px solid rgba(220,38,38,0.25);
  }

  /* Title */
  .tv-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    color: #dc2626;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 0;
  }
  .tv-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 1.5rem;
  }

  /* Metric cards */
  .metric-card {
    background: #111318;
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    color: #dc2626;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Status badges */
  .badge-threat {
    display: inline-block;
    background: rgba(220,38,38,0.15);
    border: 1px solid #dc2626;
    color: #dc2626;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
  }
  .badge-clear {
    display: inline-block;
    background: rgba(34,197,94,0.1);
    border: 1px solid #22c55e;
    color: #22c55e;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(220,38,38,0.2);
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    color: #6b7280;
    padding: 10px 24px;
    border: none;
    background: transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #dc2626 !important;
    border-bottom: 2px solid #dc2626 !important;
    background: rgba(220,38,38,0.05) !important;
  }

  /* Buttons */
  .stButton > button {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.5);
    color: #dc2626;
    border-radius: 2px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: rgba(220,38,38,0.25);
    border-color: #dc2626;
    color: #fff;
  }

  /* Sliders */
  [data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #9ca3af;
    letter-spacing: 0.1em;
  }

  /* Log table */
  .log-row {
    display: flex;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.73rem;
    color: #9ca3af;
    align-items: center;
  }
  .log-row:hover { background: rgba(220,38,38,0.04); }
  .log-ts   { color: #4b5563; min-width: 130px; }
  .log-src  { color: #6b7280; min-width: 60px; text-transform: uppercase; }
  .log-count { color: #dc2626; min-width: 30px; text-align: right; }
  .log-file { color: #6b7280; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Divider */
  hr { border-color: rgba(220,38,38,0.15) !important; }

  /* Image captions */
  .stImage > div > div { color: #4b5563 !important; font-size: 0.7rem !important; }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 1px dashed rgba(220,38,38,0.3) !important;
    border-radius: 4px !important;
    background: rgba(220,38,38,0.02) !important;
  }

  /* Progress bar */
  .stProgress > div > div {
    background: #dc2626 !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0c0f; }
  ::-webkit-scrollbar-thumb { background: rgba(220,38,38,0.4); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def threat_badge(count: int) -> str:
    if count > 0:
        return f'<span class="badge-threat">⚠ {count} THREAT{"S" if count != 1 else ""} DETECTED</span>'
    return '<span class="badge-clear">✓ CLEAR</span>'


def render_metrics(count: int, classes: list):
    counts = Counter(classes)
    cols = st.columns(max(len(counts) + 1, 2))
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{count}</div>
          <div class="metric-label">Total Detections</div>
        </div>""", unsafe_allow_html=True)
    for i, (cls, n) in enumerate(counts.items(), 1):
        if i < len(cols):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{n}</div>
                  <div class="metric-label">{cls.upper()}</div>
                </div>""", unsafe_allow_html=True)


def download_button(path: str, label: str, mime: str):
    with open(path, "rb") as f:
        data = f.read()
    st.download_button(
        label=f"⬇  DOWNLOAD {label}",
        data=data,
        file_name=os.path.basename(path),
        mime=mime,
        use_container_width=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tv-title">THREAT<br>VISION</div>', unsafe_allow_html=True)
    st.markdown('<div class="tv-subtitle">Weapon Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")

    confidence = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=int(config.MIN_CONFIDENCE * 100),
        max_value=int(config.MAX_CONFIDENCE * 100),
        value=int(config.DEFAULT_CONFIDENCE * 100),
        step=5,
        format="%d%%",
        help="Detections below this confidence score are ignored.",
    )
    confidence = confidence / 100.0   # convert back to 0.0–1.0 for inference

    st.markdown("---")
    st.markdown(
        f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:#4b5563;">'
        f'MODEL: {config.MODEL_ID}<br>'
        f'CONF:  {confidence:.0%}</p>',
        unsafe_allow_html=True,
    )

    # API key warning
    if not config.ROBOFLOW_API_KEY:
        st.error("⚠ ROBOFLOW_API_KEY not set.\nRun: export ROBOFLOW_API_KEY=xxx")


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown('<div class="tv-title" style="font-size:1.8rem">THREATVISION</div>', unsafe_allow_html=True)
st.markdown('<div class="tv-subtitle">REAL-TIME WEAPON & THREAT DETECTION</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam, tab_log = st.tabs([
    "◈  IMAGE", "▶  VIDEO", "◉  LIVE WEBCAM", "≡  DETECTION LOG"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    st.markdown("#### Upload an image for threat analysis")
    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
    )

    if uploaded:
        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**ORIGINAL**")
            st.image(uploaded, use_container_width=True)

        with st.spinner("Analyzing image..."):
            try:
                result = bl.process_image(uploaded, confidence)

                with col_result:
                    st.markdown("**ANALYZED**")
                    annotated_rgb = bgr_to_rgb(cv2.imread(result["output_path"]))
                    st.image(annotated_rgb, use_container_width=True)

                st.markdown(threat_badge(result["count"]), unsafe_allow_html=True)
                st.markdown("")

                if result["count"] > 0:
                    render_metrics(result["count"], result["classes"])

                download_button(result["output_path"], "IMAGE", "image/jpeg")

            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Processing failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown("#### Upload a video for frame-by-frame threat analysis")
    uploaded_vid = st.file_uploader(
        "Drag & drop or click to upload",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_upload",
    )

    if uploaded_vid:
        st.video(uploaded_vid)

        if st.button("▶  START ANALYSIS", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text  = st.empty()

            def on_progress(pct: float):
                progress_bar.progress(pct)
                status_text.markdown(
                    f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#6b7280;">'
                    f'PROCESSING FRAMES... {pct:.0%}</p>',
                    unsafe_allow_html=True,
                )

            try:
                result = bl.process_video(uploaded_vid, confidence, on_progress)
                progress_bar.progress(1.0)
                status_text.empty()

                st.markdown("---")

                # Side-by-side: original vs processed
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.markdown("**ORIGINAL**")
                    st.video(uploaded_vid)
                with col_proc:
                    st.markdown("**ANALYZED**")
                    with open(result["output_path"], "rb") as vf:
                        st.video(vf.read())

                st.markdown("")

                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{result['total_detects']}</div>
                      <div class="metric-label">Total Detections</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{result['frames_processed']}</div>
                      <div class="metric-label">Frames Processed</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    unique = len(set(result["classes"]))
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{unique}</div>
                      <div class="metric-label">Unique Threats</div>
                    </div>""", unsafe_allow_html=True)

                if result["classes"]:
                    st.markdown("")
                    counts = Counter(result["classes"])
                    for cls, n in counts.most_common():
                        st.markdown(f'<span class="badge-threat">{cls.upper()}  ×{n}</span>&nbsp;',
                                    unsafe_allow_html=True)

                st.markdown("")
                download_button(result["output_path"], "VIDEO", "video/mp4")

            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Processing failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown("#### Live webcam threat detection")
    st.markdown(
        '<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#6b7280;">'
        f'Frames auto-saved every {config.WEBCAM_FRAME_INTERVAL}s → output/frames/</p>',
        unsafe_allow_html=True,
    )

    # Session state
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = ""
    if "last_save" not in st.session_state:
        st.session_state.last_save = 0.0
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("◉  START WEBCAM", use_container_width=True,
                     disabled=st.session_state.webcam_running):
            st.session_state.webcam_running  = True
            st.session_state.session_dir     = bl.new_webcam_session_dir()
            st.session_state.last_save       = time.time()
            st.session_state.frame_count     = 0
            st.session_state.total_detections = 0

    with col_btn2:
        if st.button("■  STOP WEBCAM", use_container_width=True,
                     disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False

    # Live metrics row
    m1, m2, m3 = st.columns(3)
    frame_counter_slot    = m1.empty()
    detection_count_slot  = m2.empty()
    status_slot           = m3.empty()

    frame_placeholder = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not cap.isOpened():
            st.error(f"Could not open webcam (index {config.WEBCAM_INDEX}). "
                     "Change WEBCAM_INDEX in config.py if needed.")
            st.session_state.webcam_running = False
        else:
            try:
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam read failed. Check your device.")
                        break

                    try:
                        out = bl.webcam_frame(
                            frame, confidence,
                            st.session_state.session_dir,
                            st.session_state.last_save,
                        )
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        break

                    st.session_state.last_save        = out["last_save_time"]
                    st.session_state.frame_count     += 1
                    st.session_state.total_detections += out["count"]

                    # Display
                    rgb = bgr_to_rgb(out["annotated"])
                    frame_placeholder.image(rgb, use_container_width=True)

                    frame_counter_slot.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{st.session_state.frame_count}</div>
                      <div class="metric-label">Frames</div>
                    </div>""", unsafe_allow_html=True)

                    detection_count_slot.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{st.session_state.total_detections}</div>
                      <div class="metric-label">Detections</div>
                    </div>""", unsafe_allow_html=True)

                    badge = "🔴 LIVE" if out["count"] > 0 else "🟢 CLEAR"
                    status_slot.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="font-size:1.4rem">{badge}</div>
                      <div class="metric-label">Status</div>
                    </div>""", unsafe_allow_html=True)

            finally:
                cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DETECTION LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("#### Detection history log")

    col_refresh, col_clear = st.columns([1, 1])
    with col_refresh:
        if st.button("↺  REFRESH LOG", use_container_width=True):
            st.rerun()
    with col_clear:
        if st.button("✕  CLEAR LOG", use_container_width=True):
            bl.clear_log()
            st.success("Log cleared.")
            st.rerun()

    log = bl.get_log()

    if not log:
        st.markdown(
            '<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.8rem;color:#4b5563;'
            'text-align:center;padding:2rem;">NO EVENTS LOGGED YET</p>',
            unsafe_allow_html=True,
        )
    else:
        # Summary metrics
        total_events    = len(log)
        total_detects   = sum(e["count"] for e in log)
        all_classes     = [c for e in log for c in e.get("classes", [])]
        top_threat      = Counter(all_classes).most_common(1)[0][0].upper() if all_classes else "—"

        s1, s2, s3 = st.columns(3)
        for col, val, lbl in zip(
            [s1, s2, s3],
            [total_events, total_detects, top_threat],
            ["TOTAL EVENTS", "TOTAL DETECTIONS", "TOP THREAT"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Header row
        st.markdown("""
        <div class="log-row" style="color:#dc2626;border-bottom:1px solid rgba(220,38,38,0.3);">
          <span class="log-ts">TIMESTAMP</span>
          <span class="log-src">SOURCE</span>
          <span class="log-count">#</span>
          <span class="log-file">FILE / CLASSES</span>
        </div>""", unsafe_allow_html=True)

        for entry in reversed(log):
            classes_str = ", ".join(entry.get("classes", [])) or "—"
            fname       = os.path.basename(entry.get("file", "—"))
            st.markdown(f"""
            <div class="log-row">
              <span class="log-ts">{entry['timestamp']}</span>
              <span class="log-src">{entry['source']}</span>
              <span class="log-count">{entry['count']}</span>
              <span class="log-file">{fname} &nbsp;·&nbsp; {classes_str}</span>
            </div>""", unsafe_allow_html=True)

        # Download log as JSON
        st.markdown("")
        import json
        st.download_button(
            label="⬇  DOWNLOAD LOG (JSON)",
            data=json.dumps(log, indent=2),
            file_name=f"threatvision_log_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
        )