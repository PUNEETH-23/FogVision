"""
DashBoard.py
------------
FogVision — AI-Powered ADAS Road Safety Dashboard
Clean light-theme Streamlit interface.

Palette:
  --bg-base:   #F5F5F5
  --bg-panel:  #DFF1F1
  --bg-card:   #BBD5DA
  --accent:    #FF0000  (danger / active)

Video source: uploaded file only (no webcam).
Pipeline:
  1. VideoEngine        → 1 frame/sec from video
  2. fog_density.py     → Dark Channel Prior fog estimate (on raw frame)
  3. dehaze.py          → DehazeModel (on raw frame)
  4. object_detect.py   → YOLOv8 (on dehazed frame)
  5. red_glow.py        → brake-light detection (on dehazed frame)
  6. road_context.py    → GPS reverse-geocode + blackspots
  7. risk_score.py      → weighted risk engine
  8. llm.py             → Ollama LLM recommendation
  9. voice_alert.py     → threaded audio alerts
"""

import html as _html
import re
import time
import cv2
import folium
import streamlit as st
from streamlit_folium import st_folium

from pipeline     import ADASPipeline
from video_engine import VideoEngine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FogVision ADAS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN LIGHT THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

    :root {
        --bg-base:       #0A1128;
        --bg-panel:      #101F42;
        --bg-card:       #1C2D5A;
        --bg-card-inner: #15244C;
        --accent:        #F59E0B;
        --accent-soft:   rgba(245,158,11,0.08);
        --accent-mid:    rgba(245,158,11,0.18);
        --safe:          #10B981;
        --safe-bg:       rgba(16,185,129,0.08);
        --warn:          #F59E0B;
        --warn-bg:       rgba(245,158,11,0.08);
        --danger:        #EF4444;
        --danger-bg:     rgba(239,68,68,0.07);
        --teal-dark:     #06B6D4;
        --teal-mid:      #22D3EE;
        --teal-light:    #00F0FF;
        --text-primary:  #F3F4F6;
        --text-secondary:#E2E8F0;
        --text-muted:    #94A3B8;
        --text-dim:      #64748B;
        --border:        rgba(6,182,212,0.15);
        --border-strong: rgba(6,182,212,0.30);
        --mono:          'DM Mono', monospace;
        --radius:        6px;
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── HEADER ── */
    .adas-header {
        background: var(--bg-panel);
        border: 1.5px solid var(--border-strong);
        border-left: 5px solid var(--accent);
        border-radius: var(--radius);
        padding: 18px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 12px rgba(42,110,118,0.08);
    }
    .adas-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: var(--text-primary);
        margin: 0;
        line-height: 1;
    }
    .adas-title span { color: var(--accent); }
    .adas-subtitle {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        color: var(--text-muted);
        margin-top: 5px;
        text-transform: uppercase;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--bg-card);
        border: 1px solid var(--border-strong);
        border-radius: 100px;
        padding: 6px 14px;
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--safe);
        box-shadow: 0 0 6px var(--safe);
        animation: pulse-dot 1.5s ease-in-out infinite;
        flex-shrink: 0;
    }
    .status-dot-idle {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--text-dim);
        flex-shrink: 0;
    }
    @keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

    /* ── HIGH RISK POPUP BANNER ── */
    .risk-popup-overlay {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 99999;
        background: rgba(139,0,0,0.97);
        border-bottom: 4px solid #FF0000;
        padding: 14px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: risk-popup-flash 0.7s ease-in-out infinite;
        box-shadow: 0 4px 32px rgba(255,0,0,0.45);
    }
    @keyframes risk-popup-flash {
        0%,100%{background:rgba(139,0,0,0.97);}
        50%{background:rgba(200,0,0,0.99);}
    }
    .risk-popup-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .risk-popup-icon {
        font-size: 1.8rem;
        flex-shrink: 0;
    }
    .risk-popup-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        color: #FFFFFF;
        text-transform: uppercase;
        margin: 0;
        line-height: 1;
    }
    .risk-popup-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.14em;
        color: rgba(255,200,200,0.85);
        text-transform: uppercase;
        margin-top: 4px;
    }
    .risk-popup-score {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.06em;
        flex-shrink: 0;
    }
    .risk-popup-score span {
        font-size: 0.9rem;
        font-weight: 400;
        color: rgba(255,200,200,0.8);
    }

    /* ── SECTION TITLES ── */
    .panel-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        color: var(--teal-dark);
        text-transform: uppercase;
        margin-bottom: 10px;
        padding-bottom: 7px;
        border-bottom: 1.5px solid var(--border);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .panel-title::before {
        content: '';
        display: inline-block;
        width: 3px; height: 13px;
        background: var(--accent);
        border-radius: 2px;
        flex-shrink: 0;
    }

    /* ── CARDS ── */
    .panel-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 6px rgba(42,110,118,0.06);
    }

    /* ── METRIC GRID ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    .metric-tile {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 8px;
        text-align: center;
        position: relative;
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    .metric-tile:hover {
        border-color: var(--border-strong);
        box-shadow: 0 2px 10px rgba(42,110,118,0.10);
    }
    .metric-tile::after {
        content: '';
        position: absolute;
        bottom: 0; left: 10%; right: 10%; height: 2px;
        background: var(--bg-card);
        border-radius: 2px;
    }
    .metric-label {
        font-family: var(--mono);
        font-size: 0.58rem;
        letter-spacing: 0.14em;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--teal-dark);
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.6rem;
        color: var(--text-muted);
        margin-top: 2px;
        font-family: var(--mono);
    }

    /* ── RISK BADGE ── */
    .risk-badge {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        padding: 4px 14px;
        border-radius: 4px;
        display: inline-block;
        text-transform: uppercase;
    }
    .risk-LOW    { color: var(--safe);   background: var(--safe-bg);   border: 1.5px solid var(--safe); }
    .risk-MEDIUM { color: var(--warn);   background: var(--warn-bg);   border: 1.5px solid var(--warn); }
    .risk-HIGH   {
        color: #fff;
        background: var(--accent);
        border: 1.5px solid var(--accent);
        animation: risk-blink 0.9s ease-in-out infinite;
    }
    @keyframes risk-blink { 0%,100%{opacity:1;} 50%{opacity:0.7;} }

    /* ── COMPONENT BARS ── */
    .comp-bar-row   { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
    .comp-bar-label { width: 72px; color: var(--text-muted); font-family: var(--mono); text-transform: capitalize; font-size: 0.62rem; }
    .comp-bar-track { flex: 1; background: var(--bg-card); border-radius: 3px; height: 5px; overflow: hidden; }
    .comp-bar-fill  { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
    .comp-bar-val   { width: 30px; text-align: right; font-family: var(--mono); font-size: 0.62rem; color: var(--text-muted); }

    /* ── ALERTS ── */
    .alert-critical {
        background: rgba(255,0,0,0.06);
        border: 1px solid rgba(255,0,0,0.30);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 9px 13px;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 0.83rem;
        color: #8B0000;
        animation: alert-flash 1.1s ease-in-out infinite;
    }
    @keyframes alert-flash {
        0%,100%{background:rgba(255,0,0,0.06);}
        50%{background:rgba(255,0,0,0.13);}
    }
    .alert-warn {
        background: rgba(196,119,0,0.06);
        border: 1px solid rgba(196,119,0,0.25);
        border-left: 4px solid var(--warn);
        border-radius: var(--radius);
        padding: 9px 13px;
        margin-bottom: 8px;
        font-size: 0.83rem;
        color: #7A4B00;
    }
    .alert-info {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 4px solid var(--teal-light);
        border-radius: var(--radius);
        padding: 9px 13px;
        margin-bottom: 8px;
        font-size: 0.83rem;
        color: var(--text-secondary);
    }

    /* ── PILLS ── */
    .pill {
        display: inline-block;
        background: var(--bg-card);
        border: 1px solid var(--border-strong);
        border-radius: 3px;
        padding: 2px 9px;
        font-size: 0.65rem;
        font-family: var(--mono);
        color: var(--teal-dark);
        margin: 2px 3px;
        letter-spacing: 0.06em;
    }
    .pill-danger {
        background: rgba(255,0,0,0.07);
        border-color: rgba(255,0,0,0.30);
        color: var(--accent);
    }

    /* ── LLM CARD — ENHANCED ── */
    .llm-card-hero {
        background: linear-gradient(135deg, #101F42 0%, #06B6D4 100%);
        border: 2px solid var(--teal-dark);
        border-radius: var(--radius);
        padding: 20px 22px;
        font-size: 0.88rem;
        line-height: 1.85;
        color: #F3F4F6;
        font-family: 'DM Sans', sans-serif;
        box-shadow: 0 4px 20px rgba(6,182,212,0.22);
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }
    .llm-card-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent);
    }
    .llm-card-hero-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(6,182,212,0.18);
    }
    .llm-card-hero-badge {
        background: rgba(245,158,11,0.18);
        border: 1px solid rgba(245,158,11,0.40);
        border-radius: 3px;
        padding: 3px 10px;
        font-family: 'DM Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.18em;
        color: #FCD34D;
        text-transform: uppercase;
    }
    .llm-card-hero-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #F3F4F6;
        text-transform: uppercase;
        flex: 1;
    }
    .llm-row {
        display: flex;
        gap: 10px;
        margin: 8px 0;
        align-items: flex-start;
        background: rgba(6,182,212,0.06);
        border-radius: 4px;
        padding: 7px 10px;
    }
    .llm-num {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        color: #FCD34D;
        min-width: 20px;
        padding-top: 2px;
        flex-shrink: 0;
    }
    .llm-text  {
        flex: 1;
        color: #F3F4F6;
        font-size: 0.86rem;
        line-height: 1.65;
    }
    .llm-para  { margin: 6px 0; color: #F3F4F6; font-size: 0.86rem; }
    .llm-waiting {
        color: rgba(223,241,241,0.45);
        font-family: 'DM Mono', monospace;
        font-size: 0.70rem;
        letter-spacing: 0.1em;
        animation: blink-wait 1.3s ease-in-out infinite;
    }
    @keyframes blink-wait { 0%,100%{opacity:0.3;} 50%{opacity:0.9;} }

    /* Legacy llm-card (fallback) */
    .llm-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-top: 3px solid var(--teal-dark);
        border-radius: var(--radius);
        padding: 14px 18px;
        font-size: 0.84rem;
        line-height: 1.75;
        color: var(--text-secondary);
        font-family: 'DM Sans', sans-serif;
    }

    /* ── IDLE PLACEHOLDER ── */
    .idle-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 260px;
        border: 1.5px dashed var(--border-strong);
        border-radius: var(--radius);
        color: var(--text-dim);
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-align: center;
        gap: 10px;
        background: var(--bg-panel);
    }
    .idle-icon { font-size: 2.2rem; opacity: 0.30; }

    /* ── FRAME INFO ── */
    .frame-info {
        font-family: var(--mono);
        font-size: 0.62rem;
        color: var(--text-dim);
        letter-spacing: 0.10em;
        padding: 5px 0;
        display: flex;
        gap: 18px;
    }
    .frame-info span { color: var(--text-secondary); }

    /* ── STREAMLIT OVERRIDES ── */
    section[data-testid="stSidebar"] { background: var(--bg-panel) !important; }

    .stButton > button {
        background: var(--bg-panel) !important;
        color: var(--teal-dark) !important;
        border: 1.5px solid var(--border-strong) !important;
        border-radius: var(--radius) !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        padding: 7px 18px !important;
        transition: all 0.18s !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        background: var(--bg-card) !important;
        border-color: var(--teal-dark) !important;
        box-shadow: 0 2px 8px rgba(42,110,118,0.15) !important;
    }
    .stButton > button:disabled { opacity: 0.35 !important; cursor: not-allowed !important; }

    div[data-testid="stFileUploader"] {
        background: var(--bg-panel) !important;
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: var(--radius) !important;
    }
    div[data-testid="stMetric"] {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 10px !important;
    }
    iframe { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 0.06em;
        color: var(--teal-dark) !important;
    }
    .stSuccess, .stInfo, .stWarning, .stError { border-radius: var(--radius) !important; }

    ::-webkit-scrollbar { width: 5px; background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--bg-card); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "uploaded_file"  not in st.session_state: st.session_state.uploaded_file  = None
if "pipeline"       not in st.session_state: st.session_state.pipeline       = None
if "video_engine"   not in st.session_state: st.session_state.video_engine   = None
if "run_loop"       not in st.session_state: st.session_state.run_loop       = False
if "latest_result"  not in st.session_state: st.session_state.latest_result  = None
if "input_mode"     not in st.session_state: st.session_state.input_mode     = "Upload Video"
if "live_source"    not in st.session_state: st.session_state.live_source    = "0"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
is_active = st.session_state.run_loop and st.session_state.video_engine is not None
status_dot_html = '<span class="status-dot"></span>' if is_active else '<span class="status-dot-idle"></span>'
if is_active:
    status_label = "SYSTEM ACTIVE - VIDEO" if st.session_state.input_mode == "Upload Video" else "SYSTEM ACTIVE - LIVE FEED"
else:
    status_label = "AWAITING VIDEO INPUT" if st.session_state.input_mode == "Upload Video" else "AWAITING LIVE FEED CONNECTION"

st.markdown(
    f"""
    <div class="adas-header">
        <div>
            <h1 class="adas-title">FOG<span>VISION</span> ADAS</h1>
            <div class="adas-subtitle">
                AI-POWERED DRIVER ASSISTANCE &nbsp;·&nbsp;
                DARK CHANNEL PRIOR · YOLOV8 · RISK ENGINE · LLM INFERENCE
            </div>
        </div>
        <div class="status-badge">
            {status_dot_html}
            {status_label}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SOURCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
input_mode = st.radio(
    "▸  Select Input Source",
    options=["Upload Video", "Live Feed (Sensor + USB Cam/DroidCam)"],
    index=0 if st.session_state.input_mode == "Upload Video" else 1,
    horizontal=True,
)

if input_mode != st.session_state.input_mode:
    st.session_state.input_mode = input_mode
    # Release previous video engine if switching modes
    if st.session_state.video_engine is not None:
        st.session_state.video_engine.release()
        st.session_state.video_engine = None
    st.session_state.run_loop = False
    st.session_state.latest_result = None
    st.rerun()

if st.session_state.input_mode == "Upload Video":
    uploaded_file = st.file_uploader(
        "▸  Upload Video for Analysis",
        type=["mp4", "avi", "mov", "mkv"],
        help="MP4 / AVI / MOV / MKV — the system extracts 1 frame per second for analysis.",
    )

    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_engine  = VideoEngine(source=temp_video_path)
        st.session_state.pipeline      = ADASPipeline()
        st.session_state.latest_result = None
        st.session_state.run_loop      = True   # auto-start on upload
        st.rerun()
else:
    live_c1, live_c2 = st.columns([3, 1])
    with live_c1:
        live_source_input = st.text_input(
            "▸  USB Webcam Index or IP Camera URL (ESP32/DroidCam)",
            value=st.session_state.live_source,
            help="E.g., 0 for default USB webcam, 1 for secondary webcam, or http://192.168.1.100/stream",
        )
    with live_c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        connect_btn = st.button("🔌 CONNECT FEED")

    if connect_btn:
        st.session_state.live_source = live_source_input
        # Parse live source (integer if all digits, else keep string)
        if live_source_input.isdigit():
            actual_source = int(live_source_input)
        else:
            actual_source = live_source_input

        # Re-initialize engine and pipeline
        if st.session_state.video_engine is not None:
            st.session_state.video_engine.release()

        with st.spinner("Connecting to live feed..."):
            try:
                engine = VideoEngine(source=actual_source)
                if engine.is_opened:
                    st.session_state.video_engine = engine
                    st.session_state.pipeline = ADASPipeline()
                    st.session_state.latest_result = None
                    st.session_state.run_loop = True
                    st.success("Connected to feed successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to open live feed source. Please verify it is active and reachable.")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS  (always visible, disabled until a source is connected)
# ─────────────────────────────────────────────────────────────────────────────
feed_active = st.session_state.video_engine is not None
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 10])
with ctrl_c1:
    if st.button("▶ START", disabled=not feed_active):
        st.session_state.run_loop = True
with ctrl_c2:
    if st.button("⏹ STOP", disabled=not feed_active):
        st.session_state.run_loop = False

# ─────────────────────────────────────────────────────────────────────────────
# GPS — browser or fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from streamlit_js_eval import get_geolocation
    location = get_geolocation()
    if location and "coords" in location:
        lat = location["coords"]["latitude"]
        lon = location["coords"]["longitude"]
    else:
        raise ValueError("no coords")
except Exception:
    lat, lon = 28.6139, 77.2090

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE READY FLAG
# ─────────────────────────────────────────────────────────────────────────────
pipeline_ready = (
    st.session_state.video_engine is not None
    and st.session_state.pipeline is not None
)
if st.session_state.input_mode == "Upload Video":
    pipeline_ready = pipeline_ready and st.session_state.uploaded_file is not None

# ─────────────────────────────────────────────────────────────────────────────
# FRAME ACQUISITION & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
video_frame   = None
display_frame = None

if pipeline_ready:
    engine:   VideoEngine  = st.session_state.video_engine
    pipeline: ADASPipeline = st.session_state.pipeline

    if not engine.is_opened:
        st.error("❌ Could not open video source.")
    else:
        video_frame = engine.read()
        if video_frame is None:
            if st.session_state.input_mode == "Upload Video":
                st.session_state.run_loop = False   # end of video
            else:
                st.warning("⚠️ Live feed frame drop / connection timeout.")
        elif st.session_state.run_loop:
            is_video_mode = (st.session_state.input_mode == "Upload Video")
            st.session_state.latest_result = pipeline.process(video_frame.frame, lat, lon, is_video=is_video_mode)

latest_result = st.session_state.latest_result

if latest_result:
    fog_data      = latest_result.get("fog_data",            {}) or {}
    fps           = latest_result.get("fps",                  0)
    detections    = latest_result.get("detections",          []) or []
    road_context  = latest_result.get("road_context",        {}) or {}
    alerts        = latest_result.get("alerts",              []) or []
    llm_response  = latest_result.get("llm_response",        "") or ""
    risk_score    = latest_result.get("risk_score",          0.0)
    risk_level    = latest_result.get("risk_level",     "UNKNOWN")
    risk_comps    = latest_result.get("risk_components",     {}) or {}
    red_glow      = latest_result.get("red_glow",          False)
    dist_m        = latest_result.get("distance_to_nearest", 0.0)
    near_label    = latest_result.get("nearest_label",    "none")
    override      = latest_result.get("hard_override",    False)
    override_r    = latest_result.get("override_reason",     "")
    display_frame = latest_result["frame"]
else:
    fog_data = {}; fps = 0; detections = []; road_context = {}; alerts = []
    llm_response = ""; risk_score = 0.0; risk_level = "UNKNOWN"
    risk_comps = {}; red_glow = False; dist_m = 0.0; near_label = "none"
    override = False; override_r = ""

# ─────────────────────────────────────────────────────────────────────────────
# ⚠ HIGH RISK POPUP BANNER — injected at top when risk is HIGH
# ─────────────────────────────────────────────────────────────────────────────
if risk_level == "HIGH":
    score_pct_banner = int(risk_score * 100)
    fog_dens_banner  = fog_data.get("fog_density", 0.0)
    # Build a concise reason string from top risk component
    top_comp = max(risk_comps, key=risk_comps.get) if risk_comps else "multiple factors"
    st.markdown(
        f"""
        <div class="risk-popup-overlay">
            <div class="risk-popup-left">
                <div class="risk-popup-icon">🚨</div>
                <div>
                    <div class="risk-popup-title">⚠ CRITICAL RISK DETECTED — REDUCE SPEED IMMEDIATELY</div>
                    <div class="risk-popup-sub">
                        PRIMARY FACTOR: {_html.escape(str(top_comp).upper())} &nbsp;·&nbsp;
                        FOG: {fog_dens_banner:.1f}% &nbsp;·&nbsp;
                        VOICE ALERT ACTIVE
                    </div>
                </div>
            </div>
            <div class="risk-popup-score">{score_pct_banner:02d}<span>/100</span></div>
        </div>
        <div style="height:62px"></div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="medium")

# ═══════════════════════════════════════════
# LEFT COLUMN
# ═══════════════════════════════════════════
with left_col:

    # ── VIDEO FEED ──
    st.markdown('<div class="panel-title">LIVE PROCESSED FEED — DEHAZED + ANNOTATED</div>', unsafe_allow_html=True)

    if display_frame is not None:
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        st.image(display_rgb, channels="RGB", use_container_width=True)
        if video_frame is not None:
            eng = st.session_state.video_engine
            st.markdown(
                f'<div class="frame-info">'
                f'FRAME <span>#{video_frame.frame_index}</span> &nbsp;·&nbsp;'
                f'SECOND <span>#{video_frame.second_index}</span> &nbsp;·&nbsp;'
                f'SRC FPS <span>{eng.source_fps:.0f}</span> &nbsp;·&nbsp;'
                f'SKIP <span>{eng.skip_frames + 1}</span> &nbsp;·&nbsp;'
                f'PIPELINE FPS <span>{fps}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        idle_text = "CONNECT A LIVE FEED TO BEGIN" if st.session_state.input_mode != "Upload Video" else "UPLOAD A VIDEO TO BEGIN ANALYSIS"
        st.markdown(
            f'<div class="idle-placeholder">'
            f'<div class="idle-icon">⬡</div>'
            f'<div>{idle_text}</div>'
            f'<div style="font-size:0.6rem;opacity:0.5;letter-spacing:0.12em;margin-top:4px;">PIPELINE STANDBY</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── LIVE SAFETY METRICS ──
    fog_dens   = fog_data.get("fog_density",       0.0)
    visibility = fog_data.get("visibility",        "—")
    rec_speed  = fog_data.get("recommended_speed", 0)

    fog_color  = 'var(--danger)' if fog_dens > 60 else 'var(--warn)' if fog_dens > 30 else 'var(--safe)'
    dist_color = 'var(--danger)' if 0 < dist_m < 15 else 'var(--warn)' if 0 < dist_m < 30 else 'var(--teal-dark)'
    dist_val   = "—" if dist_m == 0 else f"{dist_m:.0f}m"
    rg_color   = 'var(--danger)' if red_glow else 'var(--safe)'
    rg_label   = "🔴 ACTIVE" if red_glow else "✅ CLEAR"

    st.markdown(
        f'<div class="panel-title">LIVE SAFETY METRICS</div>'
        f'<div class="metric-grid">'
        f'<div class="metric-tile"><div class="metric-label">FOG DENSITY</div>'
        f'<div class="metric-value" style="color:{fog_color}">{fog_dens:.1f}%</div></div>'
        f'<div class="metric-tile"><div class="metric-label">VISIBILITY</div>'
        f'<div class="metric-value" style="font-size:1rem;color:var(--teal-dark)">{visibility}</div></div>'
        f'<div class="metric-tile"><div class="metric-label">REC. SPEED</div>'
        f'<div class="metric-value" style="color:var(--accent)">{rec_speed}</div><div class="metric-sub">km/h</div></div>'
        f'<div class="metric-tile"><div class="metric-label">PIPELINE FPS</div>'
        f'<div class="metric-value" style="color:var(--teal-mid)">{fps}</div></div>'
        f'<div class="metric-tile"><div class="metric-label">NEAREST OBJ</div>'
        f'<div class="metric-value" style="color:{dist_color}">{dist_val}</div>'
        f'<div class="metric-sub" style="color:var(--text-muted)">{near_label if near_label != "none" else ""}</div></div>'
        f'<div class="metric-tile"><div class="metric-label">RED GLOW</div>'
        f'<div class="metric-value" style="color:{rg_color};font-size:0.85rem">{rg_label}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── RISK SCORE ENGINE ──
    risk_css    = {"LOW": "risk-LOW", "MEDIUM": "risk-MEDIUM", "HIGH": "risk-HIGH"}.get(risk_level, "risk-LOW")
    score_pct   = int(risk_score * 100)
    score_color = "var(--danger)" if score_pct > 60 else "var(--warn)" if score_pct > 30 else "var(--safe)"

    comp_bars_html = ""
    for k, v in risk_comps.items():
        pct   = int(v * 100)
        color = "var(--danger)" if pct > 60 else "var(--warn)" if pct > 30 else "var(--safe)"
        comp_bars_html += (
            f'<div class="comp-bar-row">'
            f'<span class="comp-bar-label">{_html.escape(str(k))}</span>'
            f'<div class="comp-bar-track"><div class="comp-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<span class="comp-bar-val" style="color:var(--text-muted);">{pct}%</span>'
            f'</div>'
        )

    override_html = ""
    if override and override_r:
        override_html = (
            f'<div style="margin-top:10px;padding:8px 12px;background:var(--danger-bg);'
            f'border:1px solid var(--border-strong);border-radius:5px;font-size:0.76rem;color:var(--danger);">'
            f'⚠ HARD OVERRIDE: {_html.escape(override_r)}</div>'
        )

    ov_color = 'var(--danger)' if override else 'var(--safe)'
    ov_label = "YES ⚠" if override else "NO"

    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-title">RISK SCORE ENGINE</div>'
        f'<div style="display:flex;align-items:center;gap:24px;margin-bottom:12px;">'
        f'<div>'
        f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:2.8rem;font-weight:800;color:{score_color};line-height:1;">'
        f'{score_pct:02d}<span style="font-size:1.1rem;color:{score_color};font-weight:600;">/100</span></div>'
        f'<div style="font-size:0.6rem;font-family:\'DM Mono\',monospace;color:#6A9098;letter-spacing:0.12em;margin-top:3px;">RISK INDEX</div>'
        f'</div>'
        f'<div><span class="risk-badge {risk_css}">{risk_level}</span>'
        f'<div style="font-size:0.6rem;font-family:\'DM Mono\',monospace;color:#6A9098;letter-spacing:0.1em;margin-top:7px;">'
        f'OVERRIDE: <span style="color:{ov_color};font-weight:600;">{ov_label}</span></div></div>'
        f'<div style="flex:1;">{comp_bars_html}</div>'
        f'</div>{override_html}</div>',
        unsafe_allow_html=True,
    )

    # ── SAFETY ALERTS ──
    st.markdown('<div class="panel-title">SAFETY ALERTS</div>', unsafe_allow_html=True)
    if not alerts:
        st.markdown('<div class="alert-info">✅ &nbsp; No active alerts — conditions nominal</div>', unsafe_allow_html=True)
    else:
        alert_html = ""
        for alert in alerts:
            is_crit = any(kw in alert for kw in ["CRITICAL", "DANGER", "🚨"])
            is_warn = any(kw in alert for kw in ["⚠️", "🔴", "🌫️", "📍", "🔺"])
            css = "alert-critical" if is_crit else "alert-warn" if is_warn else "alert-info"
            alert_html += f'<div class="{css}">{_html.escape(alert)}</div>'
        st.markdown(alert_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# RIGHT COLUMN
# ═══════════════════════════════════════════
with right_col:

    # ── AI RECOMMENDATION ENGINE — HERO CARD (top priority) ──
    st.markdown('<div class="panel-title">AI RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)

    if llm_response:
        if isinstance(llm_response, dict):
            inner_html = ""
            row_idx = 1
            for label, key in [
                ("Hazard Alert", "hazard_alert"),
                ("Recommended Speed", "recommended_speed"),
                ("Driving Suggestion", "driving_suggestion"),
                ("Explanation", "short_explanation"),
            ]:
                val = llm_response.get(key)
                if val:
                    inner_html += (
                        f'<div class="llm-row">'
                        f'<span class="llm-num">{row_idx}.</span>'
                        f'<span class="llm-text"><b>{_html.escape(label)}:</b> {_html.escape(str(val))}</span>'
                        f'</div>'
                    )
                    row_idx += 1
        else:
            clean_llm = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
            inner_html = ""
            for line in clean_llm.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^(\d+)\.\s+(.*)", line)
                if match:
                    num  = _html.escape(match.group(1))
                    text = _html.escape(match.group(2))
                    inner_html += (
                        f'<div class="llm-row">'
                        f'<span class="llm-num">{num}.</span>'
                        f'<span class="llm-text">{text}</span>'
                        f'</div>'
                    )
                else:
                    inner_html += f'<p class="llm-para">{_html.escape(line)}</p>'

        # Determine risk level CSS class
        risk_class_map = {"LOW": "risk-LOW", "MEDIUM": "risk-MEDIUM", "HIGH": "risk-HIGH"}
        risk_class = risk_class_map.get(risk_level, "risk-LOW")

        st.markdown(
            f'<div class="llm-card-hero">'
            f'<div class="llm-card-hero-header">'
            f'<span class="llm-card-hero-badge">● LIVE</span>'
            f'<span class="llm-card-hero-title">AI DRIVING ADVISORY</span>'
            f'<span class="risk-badge {risk_class}">{risk_level}</span>'
            f'</div>'
            f'{inner_html}'
            f'</div>',
            unsafe_allow_html=True,
)
    else:
        waiting_msg = "Awaiting LLM inference..." if pipeline_ready else "Upload a video to activate AI inference"

        st.markdown(
            f'<div class="llm-card-hero">'
            f'<div class="llm-card-hero-header">'
            f'<span class="llm-card-hero-badge">○ STANDBY</span>'
            f'<span class="llm-card-hero-title">AI DRIVING ADVISORY</span>'
            f'</div>'
            f'<span class="llm-waiting">{waiting_msg}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── MAP — with red circle zones for blackspots ──
    st.markdown('<div class="panel-title">GEOSPATIAL MAP — BLACKSPOTS & ROAD CONTEXT</div>', unsafe_allow_html=True)

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")

    # Vehicle marker
    folium.Marker(
        [lat, lon],
        popup="<b>VEHICLE LOCATION</b>",
        icon=folium.Icon(color="blue", icon="car", prefix="fa"),
    ).add_to(m)

    # Blackspot markers — red circle zone + pin
    for i, spot in enumerate(road_context.get("blackspots", [])):
        # Use actual coords from spot if available, otherwise offset for demo
        spot_lat = spot.get("lat", lat + 0.002 * (i + 1))
        spot_lon = spot.get("lon", lon + 0.002 * (i + 1))
        severity = spot.get("severity", "medium")

        # Outer danger zone circle (semi-transparent red fill)
        folium.CircleMarker(
            location=[spot_lat, spot_lon],
            radius=38,
            color="#FF0000",
            weight=2,
            fill=True,
            fill_color="#FF0000",
            fill_opacity=0.12,
            tooltip=f"⚠ DANGER ZONE: {spot.get('name', 'Blackspot')}",
        ).add_to(m)

        # Inner solid circle
        folium.CircleMarker(
            location=[spot_lat, spot_lon],
            radius=10,
            color="#CC0000",
            weight=2.5,
            fill=True,
            fill_color="#FF2222",
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b style='color:#CC0000'>⚠ {spot.get('name', 'Blackspot')}</b>"
                f"<br>Severity: <b>{severity}</b>"
                f"<br>Distance: {spot.get('distance', '?')} km",
                max_width=200,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=300, returned_objects=[])

    # ── ROAD CONTEXT ──
    road_name  = road_context.get("road",         "Unknown")
    road_type  = road_context.get("road_type",    "Unknown")
    hazard_lst = road_context.get("hazard_types", []) or []
    black_lst  = road_context.get("blackspots",   []) or []

    # RED MARK for blackspots nearby
    blackspot_indicator = f'<span style="color:#FF4444;font-weight:bold;font-size:1.2rem;margin-left:8px;">●</span>' if black_lst else ''

    hazard_pills = "".join(
        f'<span class="pill{"" if h in ("standard","high_speed") else " pill-danger"}">{_html.escape(h.upper())}</span>'
        for h in hazard_lst
    ) or '<span class="pill">NONE</span>'

    blackspot_rows = ""
    for s in black_lst:
        blackspot_rows += (
            f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;'
            f'padding:5px 0;border-bottom:1px solid var(--border);color:var(--text-secondary);">'
            f'<span>⚠ {_html.escape(str(s.get("name","?")))}</span>'
            f'<span style="color:var(--text-muted)">{_html.escape(str(s.get("distance","?")))} km &nbsp;'
            f'[{_html.escape(str(s.get("severity","?")))}]</span></div>'
        )
    if not blackspot_rows:
        blackspot_rows = '<div style="font-size:0.72rem;color:var(--text-dim);font-family:var(--mono);">No nearby blackspots</div>'

    obj_html = "".join(
        f'<span class="pill">{_html.escape(d.get("label","?").upper())}</span>' for d in detections
    ) or '<span class="pill" style="color:var(--text-dim);">None detected</span>'

    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-title">ROAD CONTEXT</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">'
        f'<div><div style="font-size:0.58rem;font-family:var(--mono);color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase;">Road Name</div>'
        f'<div style="font-size:0.85rem;color:var(--teal-dark);font-weight:500;margin-top:2px;">{_html.escape(road_name)}{blackspot_indicator}</div></div>'
        f'<div><div style="font-size:0.58rem;font-family:var(--mono);color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase;">Road Type</div>'
        f'<div style="font-size:0.85rem;color:var(--teal-dark);font-weight:500;margin-top:2px;">{_html.escape(road_type)}</div></div>'
        f'</div>'
        f'<div style="margin-bottom:10px;"><div style="font-size:0.58rem;font-family:var(--mono);color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">Hazard Types</div>{hazard_pills}</div>'
        f'<div style="margin-bottom:10px;"><div style="font-size:0.58rem;font-family:var(--mono);color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">Blackspots in Range ({len(black_lst)})</div>{blackspot_rows}</div>'
        f'<div><div style="font-size:0.58rem;font-family:var(--mono);color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">Detected Objects</div>{obj_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── DETECTION TABLE ──
    if detections:
        st.markdown('<div class="panel-title" style="margin-top:10px;">DETECTION MANIFEST</div>', unsafe_allow_html=True)
        rows_html = ""
        for d in detections:
            lbl        = _html.escape(d.get("label", "?"))
            conf       = d.get("confidence", 0.0)
            dist       = d.get("distance",   0.0)
            tl         = _html.escape(str(d.get("traffic_light_color") or "—"))
            conf_color = "var(--safe)" if conf > 0.7 else "var(--warn)" if conf > 0.4 else "var(--danger)"
            rows_html += (
                f'<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;'
                f'padding:6px 0;border-bottom:1px solid var(--border);'
                f'font-size:0.72rem;font-family:\'DM Mono\',monospace;">'
                f'<span style="color:var(--teal-dark);font-weight:500;">{lbl.upper()}</span>'
                f'<span style="color:{conf_color};font-weight:600;">{conf:.0%}</span>'
                f'<span style="color:var(--text-muted);">{dist:.1f}m</span>'
                f'<span style="color:var(--text-muted);">{tl}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="panel-card" style="padding:12px 14px;">'
            f'<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;'
            f'padding:0 0 7px 0;border-bottom:1.5px solid var(--border-strong);'
            f'font-size:0.58rem;letter-spacing:0.14em;color:var(--text-dim);font-family:\'DM Mono\',monospace;text-transform:uppercase;">'
            f'<span>Object</span><span>Conf</span><span>Dist</span><span>Signal</span></div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-RERUN LOOP
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_loop and pipeline_ready:
    time.sleep(0.05)
    st.rerun()