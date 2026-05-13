"""
FogVision_ADAS.py
=================
AI-Powered ADAS Road Safety Dashboard — single combined file.

Run with:
    streamlit run FogVision_ADAS.py

All modules are inlined in dependency order:
  1.  voice_alert      — threaded pyttsx3 audio alerts
  2.  fog_density      — Dark Channel Prior fog estimation
  3.  dehaze           — Dark Channel Prior image dehazing
  4.  object_detect    — YOLOv8 detection + traffic-light colour
  5.  red_glow         — brake-light / red-glow detection
  6.  road_context     — reverse geocoding + blackspot lookup
  7.  risk_score       — weighted multi-factor risk engine
  8.  llm              — Ollama LLM driving recommendations
  9.  video_engine     — 1-frame-per-second video sampler
  10. pipeline         — orchestrates modules 1-9
  11. dashboard        — Streamlit UI (futuristic dark theme)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STDLIB / THIRD-PARTY IMPORTS  (shared by all inline modules)
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import folium
import numpy as np
import requests
import streamlit as st
import torch
from geopy.distance import geodesic
from ollama import chat
from streamlit_folium import st_folium
from ultralytics import YOLO
import pyttsx3


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — VOICE AGENT  (priority-aware, LLM-driven TTS)
# ═══════════════════════════════════════════════════════════════════════════════

_last_alert_time: Dict[str, float] = {}
_ALERT_COOLDOWN = 5   # seconds


def _speak_raw(message: str, rate: int = 160, volume: float = 1.0) -> None:
    """Blocking TTS call — always run in a daemon thread."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        # prefer a clear female or male voice if available
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        engine.say(message)
        engine.runAndWait()
    except Exception as exc:
        print(f"[VoiceAgent] TTS error: {exc}")


def speak_alert(alert_key: str, message: str) -> None:
    """Fire a simple voice alert with per-key cooldown. Non-blocking."""
    now = time.time()
    if now - _last_alert_time.get(alert_key, 0) < _ALERT_COOLDOWN:
        return
    _last_alert_time[alert_key] = now
    print(f"[VoiceAgent] ALERT: {message}")
    threading.Thread(target=_speak_raw, args=(message,), daemon=True).start()


# ── LLM Response Parser ───────────────────────────────────────────────────────

def _parse_llm_for_voice(llm_text: str) -> Dict[str, str]:
    """
    Extract structured fields from the LLM response for voice narration.
    Expected sections: Risk Level, Hazard Alert, Recommended Speed,
    Driving Suggestion, Short Explanation.
    Returns a dict with keys: risk_level, hazard, speed, suggestion, explanation.
    Falls back to the full text if parsing fails.
    """
    # strip thinking blocks from models like DeepSeek / Qwen
    clean = re.sub(r"<think>.*?</think>", "", llm_text, flags=re.DOTALL).strip()

    fields: Dict[str, str] = {}
    patterns = {
        "risk_level":  r"(?:1[.\)]?\s*)?[Rr]isk\s+[Ll]evel[:\-]?\s*(.+)",
        "hazard":      r"(?:2[.\)]?\s*)?[Hh]azard\s+[Aa]lert[:\-]?\s*(.+)",
        "speed":       r"(?:3[.\)]?\s*)?[Rr]ecommended\s+[Ss]peed[:\-]?\s*(.+)",
        "suggestion":  r"(?:4[.\)]?\s*)?[Dd]riving\s+[Ss]uggestion[:\-]?\s*(.+)",
        "explanation": r"(?:5[.\)]?\s*)?(?:[Ss]hort\s+)?[Ee]xplanation[:\-]?\s*(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, clean)
        if m:
            fields[key] = m.group(1).strip().rstrip(".")

    if not fields:
        # fallback — use first 200 chars
        fields["fallback"] = clean[:200]

    return fields


# ── VoiceAgent class ──────────────────────────────────────────────────────────

class VoiceAgent:
    """
    Converts LLM driving recommendations into spoken voice output.

    Priority levels:
      HIGH   — spoken immediately, interrupts cooldown (red glow + close obj)
      MEDIUM — spoken if cooldown elapsed
      LOW    — spoken only when nothing else pending
    """

    HIGH_COOLDOWN   = 4    # seconds between HIGH priority utterances
    MEDIUM_COOLDOWN = 10   # seconds between MEDIUM priority utterances
    LOW_COOLDOWN    = 20   # seconds between LOW priority utterances

    def __init__(self) -> None:
        self._last_spoken:  float = 0.0
        self._last_priority: str  = "LOW"
        self._speaking:     bool  = False
        self._lock = threading.Lock()
        # transcript of last 5 utterances for dashboard display
        self.transcript: List[Dict[str, Any]] = []

    # ── internal ──────────────────────────────────────────────────────────────

    def _record(self, text: str, priority: str) -> None:
        entry = {
            "time":     time.strftime("%H:%M:%S"),
            "priority": priority,
            "text":     text,
        }
        self.transcript.append(entry)
        if len(self.transcript) > 5:
            self.transcript.pop(0)

    def _do_speak(self, utterance: str, priority: str) -> None:
        with self._lock:
            self._speaking = True
        rate = {"HIGH": 175, "MEDIUM": 160, "LOW": 150}.get(priority, 160)
        _speak_raw(utterance, rate=rate)
        with self._lock:
            self._speaking = False

    def _build_utterance(self, fields: Dict[str, str], risk_level: str) -> str:
        """Compose a natural-sounding sentence from parsed LLM fields."""
        if "fallback" in fields:
            return fields["fallback"]

        parts: List[str] = []

        rl = fields.get("risk_level", risk_level or "")
        if rl:
            parts.append(f"Risk level: {rl}.")

        hazard = fields.get("hazard", "")
        if hazard and hazard.lower() not in ("none", "no hazard", "clear"):
            parts.append(f"Hazard alert — {hazard}.")

        speed = fields.get("speed", "")
        if speed:
            parts.append(f"Recommended speed: {speed}.")

        suggestion = fields.get("suggestion", "")
        if suggestion:
            parts.append(suggestion + ".")

        explanation = fields.get("explanation", "")
        if explanation:
            parts.append(explanation)

        return " ".join(parts) if parts else "Conditions nominal. Drive safely."

    # ── public API ────────────────────────────────────────────────────────────

    def speak_llm_response(
        self,
        llm_text:   str,
        risk_level: str = "LOW",
        red_glow:   bool = False,
        distance:   float = 999.0,
    ) -> None:
        """
        Called after every LLM inference cycle.
        Decides priority, builds utterance, and fires TTS in a daemon thread.
        """
        now = time.time()

        # determine priority
        if (red_glow and distance < 20) or risk_level == "HIGH":
            priority = "HIGH"
            cooldown = self.HIGH_COOLDOWN
        elif risk_level == "MEDIUM":
            priority = "MEDIUM"
            cooldown = self.MEDIUM_COOLDOWN
        else:
            priority = "LOW"
            cooldown = self.LOW_COOLDOWN

        # downgrade if already speaking or cooldown not elapsed
        elapsed = now - self._last_spoken
        if self._speaking and priority != "HIGH":
            return
        if elapsed < cooldown:
            return

        fields    = _parse_llm_for_voice(llm_text)
        utterance = self._build_utterance(fields, risk_level)

        self._last_spoken   = now
        self._last_priority = priority
        self._record(utterance, priority)

        print(f"[VoiceAgent] {priority} — {utterance}")
        threading.Thread(
            target=self._do_speak,
            args=(utterance, priority),
            daemon=True,
        ).start()

    def speak_immediate(self, text: str) -> None:
        """Bypass cooldown for emergency one-shot utterances."""
        self._last_spoken = time.time()
        self._record(text, "HIGH")
        threading.Thread(
            target=self._do_speak,
            args=(text, "HIGH"),
            daemon=True,
        ).start()

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def last_priority(self) -> str:
        return self._last_priority


# ── singleton ─────────────────────────────────────────────────────────────────
_voice_agent = VoiceAgent()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FOG DENSITY  (Dark Channel Prior, raw frame)
# ═══════════════════════════════════════════════════════════════════════════════

def _fog_dark_channel(img: np.ndarray, window_size: int = 15) -> np.ndarray:
    min_ch = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_ch, kernel)


def _fog_atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    h, w      = img.shape[:2]
    n_px      = h * w
    n_bright  = int(max(n_px * 0.001, 1))
    dark_vec  = dark.reshape(n_px)
    img_vec   = img.reshape(n_px, 3)
    indices   = np.argsort(dark_vec)[::-1][:n_bright]
    return np.mean(img_vec[indices], axis=0)


def _fog_transmission(img: np.ndarray, atm: np.ndarray,
                      window_size: int = 15, omega: float = 0.95) -> np.ndarray:
    norm = np.zeros_like(img, dtype=np.float64)
    for i in range(3):
        norm[:, :, i] = img[:, :, i] / max(atm[i], 1e-6)
    return 1.0 - omega * _fog_dark_channel(norm, window_size)


def estimate_fog_density(image_or_path: Union[str, np.ndarray]) -> Dict[str, Any]:
    """
    Returns:
        dark_channel      : uint8 visual map
        transmission_map  : uint8 visual map
        fog_density       : float 0-100
    """
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        img = cv2.imread(image_or_path)
    if img is None:
        raise ValueError(f"Cannot open image: {image_or_path}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img_f  = img.astype(np.float64) / 255.0
    dark   = _fog_dark_channel(img_f)
    atm    = _fog_atmospheric_light(img_f, dark)
    trans  = _fog_transmission(img_f, atm)

    return {
        "dark_channel":     (dark  * 255).astype(np.uint8),
        "transmission_map": (trans * 255).astype(np.uint8),
        "fog_density":      round(float(np.mean(dark) * 100), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — DEHAZE  (Dark Channel Prior, produces enhanced frame)
# ═══════════════════════════════════════════════════════════════════════════════

class DehazeModel:
    """Dark-channel-prior image dehazing with optional guided-filter refinement."""

    def __init__(self, window_size: int = 15, omega: float = 0.95,
                 t0: float = 0.1, refine: bool = True):
        self.window_size = window_size
        self.omega       = omega
        self.t0          = t0
        self.refine      = refine
        self._kernel     = cv2.getStructuringElement(
            cv2.MORPH_RECT, (window_size, window_size))

    def process(self, frame: np.ndarray) -> np.ndarray:
        img = frame.astype(np.float64) / 255.0

        # Dark channel
        dark = cv2.erode(np.min(img, axis=2), self._kernel)

        # Atmospheric light
        h, w    = dark.shape
        n_px    = h * w
        n_b     = max(int(n_px * 0.001), 1)
        dv      = dark.reshape(n_px)
        iv      = img.reshape(n_px, 3)
        idx     = np.argsort(dv)[-n_b:]
        atm     = np.mean(iv[idx], axis=0)

        # Transmission
        norm = np.empty_like(img)
        for i in range(3):
            norm[:, :, i] = img[:, :, i] / max(atm[i], 1e-6)
        trans = np.clip(1.0 - self.omega * cv2.erode(np.min(norm, axis=2), self._kernel), 0, 1)

        # Optional guided-filter refinement
        if self.refine:
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                trans = cv2.ximgproc.guidedFilter(
                    guide=(gray * 255).astype(np.float32),
                    src=trans.astype(np.float32),
                    radius=60, eps=1e-4)
            except AttributeError:
                pass  # opencv-contrib not installed

        trans = np.maximum(trans, self.t0)

        recovered = np.empty_like(img)
        for i in range(3):
            recovered[:, :, i] = (img[:, :, i] - atm[i]) / trans + atm[i]

        return (np.clip(recovered, 0, 1) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — OBJECT DETECTION  (YOLOv8 + traffic-light colour)
# ═══════════════════════════════════════════════════════════════════════════════

_yolo_model: Optional[YOLO] = None
_HAS_CUDA = bool(torch.cuda.is_available())
_DEVICE   = 0 if _HAS_CUDA else "cpu"


def _get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("models/yolov8n.pt", task="detect")
    return _yolo_model


def _detect_tl_colour(roi: np.ndarray) -> str:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "RED":    cv2.inRange(hsv, np.array([0,120,70]),  np.array([10,255,255]))
                + cv2.inRange(hsv, np.array([170,120,70]),np.array([180,255,255])),
        "GREEN":  cv2.inRange(hsv, np.array([40,40,40]),  np.array([90,255,255])),
        "YELLOW": cv2.inRange(hsv, np.array([15,150,150]),np.array([35,255,255])),
    }
    counts = {k: cv2.countNonZero(v) for k, v in masks.items()}
    best   = max(counts, key=counts.get)
    return best if counts[best] > 20 else "UNKNOWN"


def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Run YOLOv8 on a dehazed frame. Returns (annotated_frame, detections)."""
    model      = _get_yolo()
    results    = model(frame, device=_DEVICE, half=_HAS_CUDA, verbose=False)
    annotated  = frame.copy()
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf    = float(box.conf[0])
        cls_id  = int(box.cls[0])
        label   = model.names[cls_id]
        color   = (0, 255, 0)
        tl_col  = None

        if label == "traffic light":
            roi = frame[y1:y2, x1:x2]
            if roi.size and roi.shape[0] > 15 and roi.shape[1] > 15:
                tl_col = _detect_tl_colour(roi)
                color  = {"RED": (0,0,255), "GREEN": (0,255,0), "YELLOW": (0,255,255)}.get(tl_col, color)

        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated,
                    f"{label}{'' if not tl_col else f': {tl_col}'} {conf:.2f}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        height   = y2 - y1
        distance = round(50.0 / max(height, 1), 2)

        detections.append({
            "label":              label,
            "confidence":         round(conf, 2),
            "bbox":               [x1, y1, x2, y2],
            "traffic_light_color": tl_col,
            "distance":           distance,
        })

    return annotated, detections


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — RED GLOW / BRAKE-LIGHT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_RED_LO1 = np.array([0,   120,  70], dtype=np.uint8)
_RED_HI1 = np.array([10,  255, 255], dtype=np.uint8)
_RED_LO2 = np.array([160, 120,  70], dtype=np.uint8)
_RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)
_GLOW_MIN_AREA  = 300
_GLOW_IOU_THRESH = 0.05
_VEHICLE_LABELS  = {"car", "truck", "bus", "motorcycle", "motorbike"}


def _iou(a: List[int], b: List[int]) -> float:
    xa, ya = max(a[0],b[0]), max(a[1],b[1])
    xb, yb = min(a[2],b[2]), min(a[3],b[3])
    inter  = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_red_glow(frame: np.ndarray,
                    detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    annotated = frame.copy()
    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask      = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LO1, _RED_HI1),
        cv2.inRange(hsv, _RED_LO2, _RED_HI2))
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask      = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glow_boxes: List[List[int]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < _GLOW_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        glow_boxes.append([x, y, x+w, y+h])
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(annotated, "Red Glow", (x, max(y-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    red_glow = False
    if glow_boxes:
        veh_boxes = [d["bbox"] for d in detections
                     if d.get("label","").lower() in _VEHICLE_LABELS]
        if veh_boxes:
            for gb in glow_boxes:
                for vb in veh_boxes:
                    if _iou(gb, vb) > _GLOW_IOU_THRESH:
                        red_glow = True
                        break
                if red_glow:
                    break
        else:
            red_glow = len(glow_boxes) > 0

    banner = (0,0,200) if red_glow else (0,180,0)
    cv2.rectangle(annotated, (0,0), (340,32), banner, -1)
    cv2.putText(annotated,
                "Red Glow Detected: TRUE" if red_glow else "Red Glow Detected: FALSE",
                (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    return {
        "red_glow":   red_glow,
        "glow_boxes": glow_boxes,
        "glow_frame": annotated,
        "glow_count": len(glow_boxes),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — ROAD CONTEXT  (reverse geocoding + blackspot lookup)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_blackspots() -> List[Dict[str, Any]]:
    path = "config/blackspots.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


_BLACKSPOTS: List[Dict[str, Any]] = _load_blackspots()


def get_road_context(lat: float, lon: float) -> Dict[str, Any]:
    current = (lat, lon)
    road    = "Unknown"
    raw     = {}

    try:
        url  = (f"https://nominatim.openstreetmap.org/"
                f"reverse?format=jsonv2&lat={lat}&lon={lon}")
        resp = requests.get(url, headers={"User-Agent": "road-awareness-system"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw  = data
        road = data.get("address", {}).get("road", road)
    except Exception:
        pass

    road_lower = road.lower()
    road_type  = "Highway" if ("highway" in road_lower or "nh" in road_lower) else "Urban Road"

    nearby_blackspots: List[Dict[str, Any]] = []
    hazard_types: set = set()

    for spot in _BLACKSPOTS:
        dist_km = geodesic(current, (spot["latitude"], spot["longitude"])).km
        if dist_km < 2.0:
            nearby_blackspots.append({
                "name":             spot["name"],
                "distance":         round(dist_km, 2),
                "severity":         spot.get("severity",      "unknown"),
                "fog_prone":        bool(spot.get("fog_prone",       False)),
                "rain_prone":       bool(spot.get("rain_prone",      False)),
                "landslide_prone":  bool(spot.get("landslide_prone", False)),
                "weather_risk":     spot.get("weather_risk",  "unknown"),
            })
            for flag in ("fog_prone","rain_prone","landslide_prone","accident_prone"):
                if spot.get(flag):
                    hazard_types.add(flag)
            if spot.get("weather_risk"):
                hazard_types.add(str(spot["weather_risk"]).lower())

    if "curve"      in road_type.lower():  hazard_types.add("curve")
    if "highway"    in road_type.lower():  hazard_types.add("high_speed")
    if not hazard_types:                   hazard_types.add("standard")

    return {
        "latitude":    lat,
        "longitude":   lon,
        "road":        road,
        "road_type":   road_type,
        "blackspots":  nearby_blackspots,
        "hazard_types": sorted(hazard_types),
        "raw_location": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — RISK SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_RISK_LOW_MAX = 0.30
_RISK_MED_MAX = 0.60


def _norm(v: float, lo: float, hi: float) -> float:
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo)))) if hi > lo else 0.0


def _norm_inv(v: float, lo: float, hi: float) -> float:
    return 1.0 - _norm(v, lo, hi)


def compute_risk(
    fog_density:         float,
    distance_to_nearest: float,
    recommended_speed:   float,
    road_context:        Dict[str, Any],
    red_glow:            bool,
    humidity:            float = 0.0,
) -> Dict[str, Any]:
    fog_c  = _norm(fog_density, 0, 100)
    dist_c = _norm_inv(distance_to_nearest, 5, 60) if distance_to_nearest > 0 else 0.0
    spd_c  = _norm_inv(recommended_speed, 20, 80)
    road_c = min(
        (0.4 if road_context.get("blackspots") else 0.0)
        + (0.3 if "curve" in str(road_context.get("road_type","")).lower() else 0.0),
        1.0)
    glow_c = 1.0 if red_glow else 0.0
    hum_c  = _norm(humidity, 0, 100)

    weights = {"fog":0.30,"distance":0.25,"speed":0.20,
               "road":0.10,"red_glow":0.10,"humidity":0.05}
    comps   = {"fog":round(fog_c,3),"distance":round(dist_c,3),
               "speed":round(spd_c,3),"road":round(road_c,3),
               "red_glow":round(glow_c,3),"humidity":round(hum_c,3)}

    raw   = round(sum(weights[k]*comps[k] for k in weights), 4)
    hard  = False
    reason = ""

    road_str = str(road_context.get("road_type","")).lower()

    if red_glow and 0 < distance_to_nearest < 20:
        raw = max(raw, 0.70); hard = True; reason = "Red glow + vehicle within 20 m"
    if 0 < distance_to_nearest < 10:
        raw = max(raw, 0.70); hard = True; reason = "Vehicle within 10 m"
    if "curve" in road_str and 0 < distance_to_nearest < 25:
        raw = max(raw, 0.35)
        if not hard: hard = True; reason = "Curve + vehicle within 25 m"

    final = round(min(raw, 1.0), 4)
    level = "LOW" if final <= _RISK_LOW_MAX else "MEDIUM" if final <= _RISK_MED_MAX else "HIGH"

    return {
        "risk_score":       final,
        "risk_level":       level,
        "component_scores": comps,
        "hard_override":    hard,
        "override_reason":  reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — LLM  (Ollama offline inference)
# ═══════════════════════════════════════════════════════════════════════════════

_LLM_MODEL = "qwen3:1.7b"   # or "deepseek-r1:1.5b"

_LLM_SYSTEM = """
You are an AI-powered Advanced Driver Assistance System (ADAS).

Your role:
- analyze driving conditions
- assess road risk
- provide safe driving recommendations
- generate concise driver alerts

Rules:
- prioritize safety
- keep explanations short
- do not hallucinate
- do not generate unnecessary text
- maintain professional driving-assistant tone

Always provide:
1. Risk Level
2. Hazard Alert
3. Recommended Speed
4. Driving Suggestion
5. Short Explanation
"""


def get_llm_decision(context: Dict[str, Any]) -> str:
    try:
        prompt = f"Analyze the following driving conditions:\n\n{json.dumps(context, indent=2)}\n\nGenerate a driving safety assessment."
        resp   = chat(
            model=_LLM_MODEL,
            messages=[
                {"role": "system",  "content": _LLM_SYSTEM},
                {"role": "user",    "content": prompt},
            ],
        )
        return resp.message.content
    except Exception as e:
        return f"LLM Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — VIDEO ENGINE  (1 frame-per-second sampler)
# ═══════════════════════════════════════════════════════════════════════════════

VideoSource = Union[int, str]


@dataclass
class VideoFrame:
    frame:        "cv2.Mat"
    timestamp:    float
    frame_index:  int = 0
    second_index: int = 0


class VideoEngine:
    """Delivers exactly 1 analysed frame per source-second."""

    _DEFAULT_FPS: float = 30.0

    def __init__(self, source: VideoSource,
                 target_size: Optional[Tuple[int, int]] = (960, 540)) -> None:
        self.source      = source
        self.target_size = target_size
        self._cap        = cv2.VideoCapture(source)
        reported         = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        self._source_fps = reported if reported > 1.0 else self._DEFAULT_FPS
        self._skip_frames       = max(1, round(self._source_fps)) - 1
        self._raw_frame_index   = 0
        self._second_index      = 0
        self._last_deliver_ts   = 0.0
        self._last_delivered: Optional[VideoFrame] = None

    @property
    def is_opened(self) -> bool:
        return bool(self._cap) and self._cap.isOpened()

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def skip_frames(self) -> int:
        return self._skip_frames

    def read(self) -> Optional[VideoFrame]:
        if not self.is_opened:
            return None
        for _ in range(self._skip_frames):
            ok, _ = self._cap.read()
            if not ok:
                self._try_loop(); return None
            self._raw_frame_index += 1
        ok, frame = self._cap.read()
        if not ok:
            self._try_loop(); return None
        self._raw_frame_index += 1

        if isinstance(self.source, int):
            now = time.time()
            if now - self._last_deliver_ts < 1.0:
                return self._last_delivered
            self._last_deliver_ts = now

        if self.target_size:
            frame = cv2.resize(frame, self.target_size)

        vf = VideoFrame(frame=frame, timestamp=time.time(),
                        frame_index=self._raw_frame_index,
                        second_index=self._second_index)
        self._second_index  += 1
        self._last_delivered = vf
        return vf

    def peek_last(self) -> Optional[VideoFrame]:
        return self._last_delivered

    def release(self) -> None:
        if self._cap is not None:
            try:    self._cap.release()
            finally: self._cap = None

    def _try_loop(self) -> None:
        if isinstance(self.source, str):
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._raw_frame_index = 0
                self._second_index    = 0
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — PIPELINE  (orchestrates all modules)
# ═══════════════════════════════════════════════════════════════════════════════

_dehaze_model = DehazeModel()


def _visibility_from_density(d: float) -> str:
    if d >= 80: return "Very Low"
    if d >= 60: return "Low"
    if d >= 40: return "Moderate"
    return "Good"


def _speed_from_density(d: float) -> int:
    if d >= 80: return 30
    if d >= 60: return 40
    if d >= 40: return 50
    if d >= 20: return 60
    return 70


def _rule_engine(context: dict, risk_result: dict, fog_density: float,
                 red_glow: bool, distance_to_nearest: float,
                 detections: list, traffic_signal: str) -> List[str]:
    alerts: List[str] = []

    if red_glow and 0 < distance_to_nearest < 20:
        alerts.append(f"⚠️ CRITICAL — Vehicle Ahead with Brake Light ON "
                      f"({distance_to_nearest:.0f} m). Slow Down Immediately!")
        speak_alert("red_glow_close", "Vehicle ahead. Brake lights on. Slow down immediately.")
    elif 0 < distance_to_nearest < 10:
        alerts.append(f"🚨 DANGER — Object within {distance_to_nearest:.0f} m!")
        speak_alert("too_close", "Danger. Object very close. Brake now.")
    elif 0 < distance_to_nearest < 20:
        alerts.append(f"⚠️ Vehicle Ahead — {distance_to_nearest:.0f} m. Maintain safe distance.")
        speak_alert("vehicle_ahead", "Vehicle ahead. Maintain safe distance.")

    if red_glow and distance_to_nearest >= 20:
        alerts.append("🔴 Brake lights detected ahead. Prepare to slow down.")
        speak_alert("red_glow", "Brake lights ahead. Prepare to slow down.")

    if fog_density >= 70:
        alerts.append(f"🌫️ Dense fog detected ({fog_density:.0f}%). Reduce speed.")
        speak_alert("fog", "Dense fog detected. Reduce speed.")

    if traffic_signal == "RED":
        alerts.append("🔴 Red traffic signal ahead. Stop.")
        speak_alert("red_signal", "Red signal ahead. Slow down.")

    if context.get("blackspot_nearby"):
        alerts.append("📍 Accident-prone area nearby. Drive carefully.")
        speak_alert("blackspot", "Warning. Accident-prone area ahead.")

    if any(d.get("label") in {"car","truck","bus"} for d in detections):
        if not any(kw in a for a in alerts for kw in ["Vehicle Ahead","DANGER","CRITICAL"]):
            alerts.append("🚗 Vehicle detected in frame.")

    if risk_result["hard_override"] and risk_result["risk_level"] == "HIGH":
        reason = risk_result.get("override_reason", "")
        if reason and not any(reason in a for a in alerts):
            alerts.append(f"🔺 Risk override: {reason}")

    return alerts


class ADASPipeline:

    def __init__(self) -> None:
        self.llm_interval_s  = 5.0
        self.gps_interval_s  = 10.0
        self.last_llm_time   = 0.0
        self.last_gps_time   = 0.0
        self.cached_road_context: dict = {}
        self.cached_llm_response: str  = ""
        self.cached_fog_data: dict = {
            "dark_channel": None, "transmission_map": None,
            "fog_density": 0.0, "visibility": "UNKNOWN", "recommended_speed": 0,
        }
        self.cached_full_result: dict = {}

    def process(self, frame: np.ndarray, lat: float, lon: float) -> dict:
        t_start = time.time()
        now     = t_start

        frame = cv2.resize(frame, (640, 480))

        # ── 1. Fog density on RAW frame ───────────────────────────────────────
        fog_data = estimate_fog_density(frame)
        fog_dens = fog_data.get("fog_density", 0.0)
        fog_data["visibility"]        = _visibility_from_density(fog_dens)
        fog_data["recommended_speed"] = _speed_from_density(fog_dens)
        self.cached_fog_data = fog_data
        visibility        = fog_data["visibility"]
        recommended_speed = fog_data["recommended_speed"]

        # ── 2. Dehaze RAW frame ───────────────────────────────────────────────
        dehazed = _dehaze_model.process(frame)

        # ── 3. Object detection on DEHAZED frame ──────────────────────────────
        annotated, detections = process_frame(dehazed)

        # ── 4. Red glow on DEHAZED frame ──────────────────────────────────────
        glow        = detect_red_glow(annotated, detections)
        red_glow    = glow["red_glow"]
        annotated   = glow["glow_frame"]

        # ── 5. Distance ───────────────────────────────────────────────────────
        distances           = [d["distance"] for d in detections if "distance" in d]
        distance_to_nearest = min(distances) if distances else 0.0
        nearest_label       = next((d["label"] for d in detections
                                    if d.get("distance") == distance_to_nearest), "UNKNOWN")

        # ── 6. Traffic signal ─────────────────────────────────────────────────
        traffic_signal = "UNKNOWN"
        for d in detections:
            if d.get("label") == "traffic light":
                traffic_signal = d.get("traffic_light_color") or "UNKNOWN"

        # ── 7. GPS / road context (gated 10 s) ────────────────────────────────
        if now - self.last_gps_time > self.gps_interval_s:
            try:
                self.cached_road_context = get_road_context(lat, lon)
            except Exception as exc:
                print(f"[Pipeline] Road context error: {exc}")
                self.cached_road_context = {"road":"Unknown","road_type":"Unknown","blackspots":[]}
            self.last_gps_time = time.time()
        road_context = self.cached_road_context

        # ── 8. Risk score ─────────────────────────────────────────────────────
        risk_result = compute_risk(
            fog_density=fog_dens, distance_to_nearest=distance_to_nearest,
            recommended_speed=float(recommended_speed), road_context=road_context,
            red_glow=red_glow)
        risk_score = risk_result["risk_score"]
        risk_level = risk_result["risk_level"]

        # ── 9. Sensor-fusion context ──────────────────────────────────────────
        context = {
            "timestamp":           time.strftime("%Y-%m-%d %H:%M:%S"),
            "fog_density":         fog_dens,
            "risk_level":          risk_level,
            "risk_score":          risk_score,
            "visibility":          visibility,
            "recommended_speed":   recommended_speed,
            "road_name":           road_context.get("road",      "Unknown"),
            "road_type":           road_context.get("road_type", "Unknown"),
            "blackspot_nearby":    len(road_context.get("blackspots", [])) > 0,
            "objects":             detections,
            "traffic_signal":      traffic_signal,
            "red_glow":            red_glow,
            "distance_to_nearest": distance_to_nearest,
            "nearest_object":      nearest_label,
            "location":            {"latitude": lat, "longitude": lon},
        }

        # ── 10. Rule engine → alerts ──────────────────────────────────────────
        alerts = _rule_engine(
            context=context, risk_result=risk_result, fog_density=fog_dens,
            red_glow=red_glow, distance_to_nearest=distance_to_nearest,
            detections=detections, traffic_signal=traffic_signal)

        # ── 11. LLM (gated 5 s) + Voice Agent ────────────────────────────────
        if now - self.last_llm_time > self.llm_interval_s:
            self.cached_llm_response = get_llm_decision(context)
            self.last_llm_time       = now
            # ── 11b. Hand off LLM output to Voice Agent ───────────────────────
            if self.cached_llm_response and not self.cached_llm_response.startswith("LLM Error"):
                _voice_agent.speak_llm_response(
                    llm_text   = self.cached_llm_response,
                    risk_level = risk_level,
                    red_glow   = red_glow,
                    distance   = distance_to_nearest,
                )

        fps = round(1.0 / max(time.time() - t_start, 1e-6), 2)

        result = {
            "frame":                annotated,
            "fog_data":             fog_data,
            "detections":           detections,
            "road_context":         road_context,
            "alerts":               alerts,
            "llm_response":         self.cached_llm_response,
            "fps":                  fps,
            "context":              context,
            "red_glow":             red_glow,
            "glow_boxes":           glow["glow_boxes"],
            "distance_to_nearest":  distance_to_nearest,
            "nearest_label":        nearest_label,
            "per_object_distances": [d.get("distance", 0.0) for d in detections],
            "risk_score":           risk_score,
            "risk_level":           risk_level,
            "risk_components":      risk_result["component_scores"],
            "hard_override":        risk_result["hard_override"],
            "override_reason":      risk_result["override_reason"],
            # Voice Agent
            "voice_speaking":       _voice_agent.is_speaking,
            "voice_priority":       _voice_agent.last_priority,
            "voice_transcript":     list(_voice_agent.transcript),
        }
        self.cached_full_result = result
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 11 — STREAMLIT DASHBOARD  (futuristic dark-theme UI)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FogVision ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Nunito+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════ CAR DASHBOARD — AUTOMOTIVE HMI THEME ═══════════════════════ */
:root {
    --hmi-bg:         #0b0e13;
    --hmi-surface:    #111620;
    --hmi-card:       #161d2c;
    --hmi-card2:      #1c2538;
    --hmi-border:     rgba(255,255,255,0.07);
    --hmi-border2:    rgba(255,255,255,0.12);
    --amber:          #f5a623;
    --amber-dim:      #b87010;
    --teal:           #00d4c8;
    --safe:           #1de98b;
    --warn:           #ffd454;
    --danger:         #ff4757;
    --font-display:   'Rajdhani', sans-serif;
    --font-body:      'Nunito Sans', sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
    --r-card:         16px;
    --r-inner:        10px;
    --r-pill:         999px;
}

html, body, [class*="css"], .stApp {
    background-color: var(--hmi-bg) !important;
    color: #c8d4e8 !important;
    font-family: var(--font-body) !important;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse 110% 80% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%);
    pointer-events: none;
    z-index: 9998;
}

#MainMenu, footer, header { visibility: hidden; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {
    font-family: var(--font-display) !important;
    letter-spacing: 0.04em;
    color: var(--amber) !important;
}
::-webkit-scrollbar { width:5px; background: var(--hmi-bg); }
::-webkit-scrollbar-thumb { background:rgba(245,166,35,0.25); border-radius:99px; }

/* Header */
.hmi-header {
    background: linear-gradient(180deg, #1a2035 0%, #111620 100%);
    border: 1px solid var(--hmi-border2);
    border-radius: var(--r-card);
    padding: 14px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(255,255,255,0.08) inset, 0 8px 32px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hmi-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber), rgba(245,166,35,0.3), transparent);
}
.hmi-logo-wrap { display:flex; align-items:center; gap:14px; }
.hmi-logo-icon {
    width:42px; height:42px;
    background: radial-gradient(circle, #f5a623 0%, #9a5c00 100%);
    border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:1.2rem; box-shadow:0 0 16px rgba(245,166,35,0.4); flex-shrink:0;
}
.hmi-title {
    font-family: var(--font-display);
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: 0.06em; color: #fff; line-height: 1;
}
.hmi-subtitle {
    font-family: var(--font-mono);
    font-size: 0.6rem; letter-spacing: 0.16em;
    color: var(--amber-dim); margin-top: 3px; text-transform: uppercase;
}
.hmi-status-cluster { display:flex; align-items:center; gap:20px; }
.hmi-status-item { display:flex; flex-direction:column; align-items:flex-end; }
.hmi-status-label {
    font-family: var(--font-mono); font-size:0.55rem;
    letter-spacing:0.14em; color:#5a6a84; text-transform:uppercase;
}
.hmi-status-val {
    font-family: var(--font-display);
    font-size: 1.1rem; font-weight: 600;
    color: var(--teal); letter-spacing: 0.05em; line-height: 1.1;
}
.hmi-live-badge {
    display:flex; align-items:center; gap:6px;
    background:rgba(29,233,139,0.1); border:1px solid rgba(29,233,139,0.3);
    border-radius:var(--r-pill); padding:4px 12px;
    font-family:var(--font-mono); font-size:0.62rem;
    letter-spacing:0.12em; color:var(--safe);
}
.hmi-live-dot {
    width:7px; height:7px; background:var(--safe);
    border-radius:50%; box-shadow:0 0 8px var(--safe);
    animation:pulse-live 1.2s ease-in-out infinite;
}
@keyframes pulse-live { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }

/* Panel card */
.panel-card {
    background: var(--hmi-card);
    border: 1px solid var(--hmi-border);
    border-radius: var(--r-card);
    padding: 16px 18px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 4px 20px rgba(0,0,0,0.4);
}
.panel-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 30%, rgba(255,255,255,0.05) 70%, transparent 100%);
}

/* Section titles */
.panel-title {
    font-family: var(--font-display);
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.18em; color: var(--amber-dim);
    text-transform: uppercase; margin-bottom: 10px;
    padding-bottom: 6px; border-bottom: 1px solid var(--hmi-border);
    display: flex; align-items: center; gap: 8px;
}
.panel-title::before {
    content: '';
    width: 3px; height: 12px;
    background: var(--amber); border-radius: 2px;
    box-shadow: 0 0 8px var(--amber); flex-shrink: 0;
}

/* Metric tiles */
.metric-tile {
    background: var(--hmi-card2);
    border: 1px solid var(--hmi-border);
    border-radius: var(--r-inner);
    padding: 14px 16px; text-align: center;
    position: relative; overflow: hidden;
}
.metric-tile::after {
    content: '';
    position: absolute; bottom:0; left:0; right:0; height:2px;
    border-radius: 0 0 var(--r-inner) var(--r-inner);
    background: var(--amber); opacity: 0.6;
}
.metric-label {
    font-family: var(--font-mono); font-size: 0.58rem;
    letter-spacing: 0.16em; color: #5a6a84;
    text-transform: uppercase; margin-bottom: 8px; display: block;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 1.45rem; font-weight: 700;
    color: var(--amber); line-height: 1; letter-spacing: 0.02em;
}
.metric-sub {
    font-size: 0.62rem; color: #5a6a84;
    margin-top: 4px; font-family: var(--font-mono); letter-spacing: 0.06em;
}
.metric-tile-safe  .metric-value { color: var(--safe); }
.metric-tile-safe::after         { background: var(--safe); }
.metric-tile-warn  .metric-value { color: var(--warn); }
.metric-tile-warn::after         { background: var(--warn); }
.metric-tile-danger .metric-value{ color: var(--danger); }
.metric-tile-danger::after       { background: var(--danger); }
.metric-tile-teal  .metric-value { color: var(--teal); }
.metric-tile-teal::after         { background: var(--teal); }

/* Risk badge */
.risk-badge {
    font-family: var(--font-display);
    font-size: 1.8rem; font-weight: 700;
    letter-spacing: 0.12em; padding: 8px 24px;
    border-radius: var(--r-pill); display: inline-block;
}
.risk-LOW    { color:var(--safe);   background:rgba(29,233,139,0.12); border:2px solid var(--safe);   box-shadow:0 0 16px rgba(29,233,139,0.2); }
.risk-MEDIUM { color:var(--warn);   background:rgba(255,212,84,0.12); border:2px solid var(--warn);   box-shadow:0 0 16px rgba(255,212,84,0.2); }
.risk-HIGH   { color:var(--danger); background:rgba(255,71,87,0.15);  border:2px solid var(--danger); box-shadow:0 0 20px rgba(255,71,87,0.3);
               animation: risk-pulse 0.7s ease-in-out infinite; }
@keyframes risk-pulse { 0%,100%{box-shadow:0 0 20px rgba(255,71,87,0.3)} 50%{box-shadow:0 0 36px rgba(255,71,87,0.6)} }

/* Component bars */
.comp-bar-row { display:flex; align-items:center; gap:10px; margin:6px 0; }
.comp-bar-label { width:68px; color:#5a6a84; font-family:var(--font-mono); font-size:0.6rem; text-transform:capitalize; letter-spacing:0.06em; }
.comp-bar-track { flex:1; background:rgba(255,255,255,0.04); border-radius:var(--r-pill); height:5px; overflow:hidden; border:1px solid rgba(255,255,255,0.06); }
.comp-bar-fill  { height:100%; border-radius:var(--r-pill); transition:width 0.6s ease; }
.comp-bar-val   { width:32px; text-align:right; font-family:var(--font-mono); font-size:0.62rem; color:#5a6a84; }

/* Alerts */
.alert-critical {
    background:rgba(255,71,87,0.1); border:1px solid rgba(255,71,87,0.4);
    border-left:3px solid var(--danger); border-radius:var(--r-inner);
    padding:10px 16px; margin-bottom:8px; font-weight:600;
    font-size:0.82rem; color:#ffb3bb;
    animation:alert-flash 0.8s ease-in-out infinite;
}
@keyframes alert-flash { 0%,100%{background:rgba(255,71,87,0.08)} 50%{background:rgba(255,71,87,0.18)} }
.alert-warn {
    background:rgba(255,212,84,0.08); border:1px solid rgba(255,212,84,0.25);
    border-left:3px solid var(--warn); border-radius:var(--r-inner);
    padding:10px 16px; margin-bottom:8px; font-size:0.82rem; color:#ffe999;
}
.alert-info {
    background:rgba(0,212,200,0.06); border:1px solid rgba(0,212,200,0.18);
    border-left:3px solid var(--teal); border-radius:var(--r-inner);
    padding:10px 16px; margin-bottom:8px; font-size:0.82rem; color:#7ee8e4;
}

/* Pills */
.pill {
    display:inline-block; background:rgba(0,212,200,0.08);
    border:1px solid rgba(0,212,200,0.22); border-radius:var(--r-pill);
    padding:3px 12px; font-size:0.66rem; font-family:var(--font-mono);
    color:var(--teal); margin:2px 3px; letter-spacing:0.06em;
}
.pill-danger { background:rgba(255,71,87,0.1); border-color:rgba(255,71,87,0.3); color:#ff8f9c; }

/* LLM card */
.llm-card {
    background: linear-gradient(145deg, #111a2a, #161d2c);
    border: 1px solid rgba(245,166,35,0.18);
    border-top: 2px solid rgba(245,166,35,0.5);
    border-radius: var(--r-inner);
    padding: 14px 18px; font-size: 0.83rem;
    line-height: 1.75; color: #b8c8de;
}
.llm-waiting {
    color: #3a4a5c; font-family: var(--font-mono);
    font-size: 0.7rem; letter-spacing: 0.1em;
    animation: blink-wait 1.2s ease-in-out infinite;
}
@keyframes blink-wait { 0%,100%{opacity:0.3} 50%{opacity:0.8} }

/* Frame info */
.frame-info {
    font-family: var(--font-mono); font-size: 0.62rem;
    color: #3a4a5c; letter-spacing: 0.1em;
    padding: 5px 0; display: flex; gap: 18px;
}
.frame-info span { color: #5a6a84; }

/* Buttons */
.stButton > button {
    background: linear-gradient(145deg, #1e2d44, #162038) !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,166,35,0.35) !important;
    border-radius: var(--r-pill) !important;
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important; letter-spacing: 0.12em !important;
    font-weight: 600 !important; padding: 8px 24px !important;
    transition: all 0.25s !important; box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(145deg, #283d58, #1e2e48) !important;
    box-shadow: 0 0 16px rgba(245,166,35,0.25), 0 2px 8px rgba(0,0,0,0.4) !important;
    border-color: rgba(245,166,35,0.6) !important;
}
div[data-testid="stFileUploader"] {
    background: var(--hmi-card) !important;
    border: 1px dashed rgba(245,166,35,0.2) !important;
    border-radius: var(--r-card) !important;
}
iframe { border-radius:var(--r-inner) !important; border:1px solid var(--hmi-border) !important; }

/* Voice Agent */
.voice-agent-card {
    background: linear-gradient(145deg, #101820, #161d2c);
    border: 1px solid rgba(29,233,139,0.18);
    border-top: 2px solid rgba(29,233,139,0.4);
    border-radius: var(--r-inner);
    padding: 14px 18px; margin-top: 10px;
}
.voice-status-row { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
.voice-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.voice-dot-active {
    background: var(--safe); box-shadow: 0 0 12px var(--safe);
    animation: pulse-live 0.8s ease-in-out infinite;
}
.voice-dot-idle { background: #2a3a50; }
.voice-priority-badge {
    font-family: var(--font-display); font-size: 0.62rem;
    letter-spacing: 0.14em; padding: 3px 12px;
    border-radius: var(--r-pill); font-weight: 700; text-transform: uppercase;
}
.vp-HIGH   { background:rgba(255,71,87,0.15);  border:1px solid #ff4757; color:#ff8f9c; }
.vp-MEDIUM { background:rgba(255,212,84,0.1);  border:1px solid #ffd454; color:#ffe999; }
.vp-LOW    { background:rgba(0,212,200,0.08);  border:1px solid rgba(0,212,200,0.3); color:var(--teal); }
.voice-transcript-row {
    display:flex; gap:10px; font-size:0.7rem; padding:6px 0;
    border-bottom:1px solid rgba(255,255,255,0.04); align-items:flex-start;
}
.vt-time  { font-family:var(--font-mono); color:#3a4a5c; min-width:54px; font-size:0.62rem; }
.vt-badge { font-family:var(--font-mono); font-size:0.56rem; min-width:48px; padding:2px 6px; border-radius:var(--r-pill); text-align:center; margin-top:1px; }
.vt-text  { color:#8090a8; line-height:1.55; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hmi-header">
    <div class="hmi-logo-wrap">
        <div class="hmi-logo-icon">🚗</div>
        <div>
            <div class="hmi-title">FogVision ADAS</div>
            <div class="hmi-subtitle">FOG · OBJECT DETECTION · RISK ENGINE · LLM · VOICE AGENT</div>
        </div>
    </div>
    <div class="hmi-status-cluster">
        <div class="hmi-status-item">
            <span class="hmi-status-label">System</span>
            <span class="hmi-status-val">ONLINE</span>
        </div>
        <div class="hmi-live-badge">
            <div class="hmi-live-dot"></div>
            LIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Video Upload ──────────────────────────────────────────────────────────────
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader(
    "▸  Upload Video for Analysis",
    type=["mp4", "avi", "mov", "mkv"],
    help="MP4 / AVI / MOV / MKV — system extracts 1 frame per second for analysis.",
)

if uploaded_file and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    with open("temp_uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_engine  = VideoEngine(source="temp_uploaded_video.mp4")
    st.session_state.pipeline      = ADASPipeline()
    st.session_state.latest_result = None
    st.rerun()

if not st.session_state.uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#3a4a5c;font-family:'Orbitron',sans-serif;
                font-size:0.8rem;letter-spacing:0.15em;border:1px dashed rgba(0,200,255,0.1);
                border-radius:4px;margin-top:20px;">
        ◈ &nbsp; UPLOAD A VIDEO FILE TO INITIALISE THE ADAS PIPELINE &nbsp; ◈
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session-state singletons ──────────────────────────────────────────────────
if "pipeline"      not in st.session_state:
    st.session_state.pipeline      = ADASPipeline()
if "run_loop"      not in st.session_state:
    st.session_state.run_loop      = True
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "video_engine"  not in st.session_state:
    st.session_state.video_engine  = VideoEngine(source="temp_uploaded_video.mp4")

# ── GPS ───────────────────────────────────────────────────────────────────────
try:
    from streamlit_js_eval import get_geolocation
    _loc = get_geolocation()
    lat  = _loc["coords"]["latitude"]  if (_loc and "coords" in _loc) else 28.6139
    lon  = _loc["coords"]["longitude"] if (_loc and "coords" in _loc) else 77.2090
except Exception:
    lat, lon = 28.6139, 77.2090

# ── Engine / Pipeline refs ────────────────────────────────────────────────────
engine:   VideoEngine  = st.session_state.video_engine
pipeline: ADASPipeline = st.session_state.pipeline

# ── Controls ──────────────────────────────────────────────────────────────────
cc1, cc2, _ = st.columns([1, 1, 10])
with cc1:
    if st.button("▶ START"): st.session_state.run_loop = True
with cc2:
    if st.button("⏹ STOP"):  st.session_state.run_loop = False

if not engine.is_opened:
    st.error("❌ Could not open video source.")
    st.stop()

# ── Frame acquisition ─────────────────────────────────────────────────────────
video_frame = engine.read()
if video_frame is None:
    time.sleep(0.05)
    st.rerun()

frame = video_frame.frame
if st.session_state.run_loop:
    st.session_state.latest_result = pipeline.process(frame, lat, lon)

latest_result = st.session_state.latest_result

# ── Unpack result ─────────────────────────────────────────────────────────────
if latest_result:
    fog_data         = latest_result.get("fog_data",             {}) or {}
    fps              = latest_result.get("fps",                   0)
    detections       = latest_result.get("detections",           []) or []
    road_context     = latest_result.get("road_context",         {}) or {}
    alerts           = latest_result.get("alerts",               []) or []
    llm_response     = latest_result.get("llm_response",         "") or ""
    risk_score       = latest_result.get("risk_score",           0.0)
    risk_level       = latest_result.get("risk_level",           "UNKNOWN")
    risk_comps       = latest_result.get("risk_components",      {}) or {}
    red_glow         = latest_result.get("red_glow",             False)
    dist_m           = latest_result.get("distance_to_nearest",  0.0)
    near_label       = latest_result.get("nearest_label",        "none")
    override         = latest_result.get("hard_override",        False)
    override_r       = latest_result.get("override_reason",      "")
    voice_speaking   = latest_result.get("voice_speaking",       False)
    voice_priority   = latest_result.get("voice_priority",       "LOW")
    voice_transcript = latest_result.get("voice_transcript",     []) or []
    display_frame    = latest_result["frame"]
else:
    fog_data = {}; fps = 0; detections = []; road_context = {}; alerts = []
    llm_response = ""; risk_score = 0.0; risk_level = "UNKNOWN"
    risk_comps = {}; red_glow = False; dist_m = 0.0; near_label = "none"
    override = False; override_r = ""; display_frame = frame
    voice_speaking = False; voice_priority = "LOW"; voice_transcript = []

display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([3, 2], gap="medium")

# ─────────────────────────────────────
# LEFT COLUMN
# ─────────────────────────────────────
with left_col:

    # Video feed
    st.markdown('<div class="panel-title">◈ LIVE PROCESSED FEED — DEHAZED + ANNOTATED</div>',
                unsafe_allow_html=True)
    st.image(display_rgb, channels="RGB", use_container_width=True)
    st.markdown(
        f"""<div class="frame-info">
            FRAME <span>#{video_frame.frame_index}</span> &nbsp;·&nbsp;
            SECOND <span>#{video_frame.second_index}</span> &nbsp;·&nbsp;
            SRC FPS <span>{engine.source_fps:.0f}</span> &nbsp;·&nbsp;
            SKIP <span>{engine.skip_frames+1}</span> &nbsp;·&nbsp;
            PIPELINE FPS <span>{fps}</span>
        </div>""",
        unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Safety metrics
    fog_dens  = fog_data.get("fog_density",       0.0)
    visib     = fog_data.get("visibility",        "—")
    rec_spd   = fog_data.get("recommended_speed",  0)

    fog_color  = "#ff4757" if fog_dens > 60 else "#ffd454" if fog_dens > 30 else "#1de98b"
    dist_color = "#ff4757" if 0 < dist_m < 15 else "#ffd454" if 0 < dist_m < 30 else "#00c8ff"
    glow_color = "#ff4757" if red_glow else "#1de98b"

    fog_tile_cls  = "metric-tile-danger" if fog_dens > 60 else "metric-tile-warn" if fog_dens > 30 else "metric-tile-safe"
    dist_tile_cls = "metric-tile-danger" if 0 < dist_m < 15 else "metric-tile-warn" if 0 < dist_m < 30 else "metric-tile-teal"
    glow_tile_cls = "metric-tile-danger" if red_glow else "metric-tile-safe"
    spd_tile_cls  = "metric-tile-warn" if rec_spd < 40 else ""

    st.markdown(f"""
    <div class="panel-title">LIVE SAFETY METRICS</div>
    <div class="metric-grid">
        <div class="metric-tile {fog_tile_cls}">
            <span class="metric-label">FOG DENSITY</span>
            <div class="metric-value">{fog_dens:.1f}%</div>
        </div>
        <div class="metric-tile metric-tile-teal">
            <span class="metric-label">VISIBILITY</span>
            <div class="metric-value" style="font-size:1rem;">{visib}</div>
        </div>
        <div class="metric-tile {spd_tile_cls}">
            <span class="metric-label">REC. SPEED</span>
            <div class="metric-value">{rec_spd}</div>
            <div class="metric-sub">km/h</div>
        </div>
        <div class="metric-tile">
            <span class="metric-label">PIPELINE FPS</span>
            <div class="metric-value">{fps}</div>
        </div>
        <div class="metric-tile {dist_tile_cls}">
            <span class="metric-label">NEAREST OBJ</span>
            <div class="metric-value">{"—" if dist_m == 0 else f"{dist_m:.0f}m"}</div>
            <div class="metric-sub">{near_label if near_label != 'none' else '—'}</div>
        </div>
        <div class="metric-tile {glow_tile_cls}">
            <span class="metric-label">RED GLOW</span>
            <div class="metric-value" style="font-size:0.95rem;">{"🔴 ACTIVE" if red_glow else "✅ CLEAR"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk score engine
    risk_css   = {"LOW":"risk-LOW","MEDIUM":"risk-MEDIUM","HIGH":"risk-HIGH"}.get(risk_level,"risk-LOW")
    score_pct  = int(risk_score * 100)
    scol       = "#ff4757" if score_pct > 60 else "#ffd454" if score_pct > 30 else "#1de98b"

    comp_html = ""
    for k, v in risk_comps.items():
        p = int(v * 100)
        c = "#ff4757" if p > 60 else "#ffd454" if p > 30 else "#1de98b"
        comp_html += f"""
        <div class="comp-bar-row">
            <span class="comp-bar-label">{k}</span>
            <div class="comp-bar-track">
                <div class="comp-bar-fill" style="width:{p}%;background:{c};"></div>
            </div>
            <span class="comp-bar-val">{p}%</span>
        </div>"""

    override_html = ""
    if override and override_r:
        override_html = f"""<div style="margin-top:10px;padding:8px 12px;
            background:rgba(255,51,85,0.1);border:1px solid rgba(255,51,85,0.4);
            border-radius:3px;font-size:0.77rem;color:#ff8080;">
            ⚠ HARD OVERRIDE: {override_r}</div>"""

    st.markdown(f"""
    <div class="panel-card" style="margin-top:4px;">
        <div class="panel-title">◈ RISK SCORE ENGINE</div>
        <div style="display:flex;align-items:center;gap:24px;margin-bottom:12px;">
            <div>
                <div style="font-family:'Orbitron',sans-serif;font-size:2.4rem;font-weight:900;
                            color:{scol};text-shadow:0 0 16px {scol};">
                    {score_pct:02d}<span style="font-size:1rem;color:{scol};">/100</span>
                </div>
                <div style="font-size:0.65rem;font-family:'Share Tech Mono',monospace;color:#4a7fa8;
                            letter-spacing:0.1em;margin-top:2px;">RISK INDEX</div>
            </div>
            <div>
                <span class="risk-badge {risk_css}">{risk_level}</span>
                <div style="font-size:0.65rem;font-family:'Share Tech Mono',monospace;color:#4a7fa8;
                            letter-spacing:0.1em;margin-top:6px;">
                    OVERRIDE: <span style="color:{'#ff4757' if override else '#1de98b'}">
                    {"YES ⚠" if override else "NO"}</span>
                </div>
            </div>
            <div style="flex:1;">{comp_html}</div>
        </div>
        {override_html}
    </div>
    """, unsafe_allow_html=True)

    # Safety alerts
    st.markdown('<div class="panel-title">◈ SAFETY ALERTS</div>', unsafe_allow_html=True)
    if not alerts:
        st.markdown('<div class="alert-info">✅ &nbsp; NO ACTIVE ALERTS — CONDITIONS NOMINAL</div>',
                    unsafe_allow_html=True)
    else:
        ahtml = ""
        for alert in alerts:
            is_crit = any(kw in alert for kw in ["CRITICAL","DANGER","🚨"])
            is_warn = any(kw in alert for kw in ["⚠️","🔴","🌫️","📍","🔺"])
            css = "alert-critical" if is_crit else "alert-warn" if is_warn else "alert-info"
            ahtml += f'<div class="{css}">{alert}</div>'
        st.markdown(ahtml, unsafe_allow_html=True)


# ─────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────
with right_col:

    # Map
    st.markdown('<div class="panel-title">◈ GEOSPATIAL MAP — BLACKSPOTS & ROAD CONTEXT</div>',
                unsafe_allow_html=True)
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB dark_matter")
    folium.Marker([lat, lon], popup="<b>VEHICLE LOCATION</b>",
                  icon=folium.Icon(color="blue", icon="car", prefix="fa")).add_to(m)
    for spot in road_context.get("blackspots", []):
        folium.Marker([lat+0.002, lon+0.002],
                      popup=f"<b>{spot.get('name','Blackspot')}</b><br>Severity: {spot.get('severity','?')}",
                      icon=folium.Icon(color="red", icon="exclamation-sign")).add_to(m)
    st_folium(m, width=None, height=300, returned_objects=[])

    # Road context
    road_name  = road_context.get("road",         "Unknown")
    road_type  = road_context.get("road_type",    "Unknown")
    hazard_lst = road_context.get("hazard_types", []) or []
    black_lst  = road_context.get("blackspots",   []) or []

    hazard_pills = "".join(
        f'<span class="pill{"" if h in ("standard","high_speed") else " pill-danger"}">{h.upper()}</span>'
        for h in hazard_lst) or '<span class="pill">NONE</span>'

    bs_rows = "".join(
        f"""<div style="display:flex;justify-content:space-between;font-size:0.72rem;
            padding:4px 0;border-bottom:1px solid rgba(0,200,255,0.06);color:#7aaac8;">
            <span>⚠ {s.get('name','?')}</span>
            <span style="color:#4a7fa8">{s.get('distance','?')} km [{s.get('severity','?')}]</span>
        </div>""" for s in black_lst
    ) or '<div style="font-size:0.72rem;color:#3a4a5c;font-family:var(--font-mono);">NO NEARBY BLACKSPOTS</div>'

    obj_html = "".join(
        f'<span class="pill">{d.get("label","?").upper()}</span>' for d in detections
    ) or '<span class="pill" style="color:#3a4a5c;">NONE DETECTED</span>'

    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-title">◈ ROAD CONTEXT</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
            <div>
                <div style="font-size:0.6rem;font-family:'Share Tech Mono',monospace;
                            color:#3a4a5c;letter-spacing:0.12em;">ROAD NAME</div>
                <div style="font-size:0.82rem;color:var(--teal);margin-top:2px;">{road_name}</div>
            </div>
            <div>
                <div style="font-size:0.6rem;font-family:'Share Tech Mono',monospace;
                            color:#3a4a5c;letter-spacing:0.12em;">ROAD TYPE</div>
                <div style="font-size:0.82rem;color:var(--teal);margin-top:2px;">{road_type}</div>
            </div>
        </div>
        <div style="margin-bottom:8px;">
            <div style="font-size:0.6rem;font-family:'Share Tech Mono',monospace;
                        color:#3a4a5c;letter-spacing:0.12em;margin-bottom:4px;">HAZARD TYPES</div>
            {hazard_pills}
        </div>
        <div style="margin-bottom:8px;">
            <div style="font-size:0.6rem;font-family:'Share Tech Mono',monospace;
                        color:#3a4a5c;letter-spacing:0.12em;margin-bottom:4px;">
                BLACKSPOTS IN RANGE ({len(black_lst)})
            </div>
            {bs_rows}
        </div>
        <div>
            <div style="font-size:0.6rem;font-family:'Share Tech Mono',monospace;
                        color:#3a4a5c;letter-spacing:0.12em;margin-bottom:4px;">DETECTED OBJECTS</div>
            {obj_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI recommendation
    st.markdown('<div class="panel-title">◈ AI RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)
    if llm_response:
        clean = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
        fmt   = ""
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and "." in line[:3]:
                parts = line.split(".", 1)
                fmt += f"""<div style="display:flex;gap:8px;margin:5px 0;">
                    <span style="font-family:'Orbitron',sans-serif;font-size:0.65rem;
                                 color:var(--accent);min-width:16px;margin-top:2px;">{parts[0]}.</span>
                    <span>{parts[1].strip() if len(parts)>1 else ''}</span>
                </div>"""
            else:
                fmt += f"<p style='margin:4px 0;'>{line}</p>"
        st.markdown(f'<div class="llm-card">{fmt}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="llm-card"><span class="llm-waiting">⬛ AWAITING LLM INFERENCE...</span></div>',
                    unsafe_allow_html=True)

    # ── Voice Agent Panel ──────────────────────────────────────────────────────
    st.markdown('<div class="panel-title" style="margin-top:12px;">◈ VOICE AGENT — LLM NARRATION</div>',
                unsafe_allow_html=True)

    dot_cls  = "voice-dot-active" if voice_speaking else "voice-dot-idle"
    dot_lbl  = "SPEAKING" if voice_speaking else "STANDBY"
    vp_cls   = f"vp-{voice_priority}"
    vp_badge = f'<span class="voice-priority-badge {vp_cls}">{voice_priority}</span>'

    # build transcript rows
    transcript_html = ""
    if voice_transcript:
        for entry in reversed(voice_transcript):
            p   = entry.get("priority", "LOW")
            tc  = {"HIGH": "vp-HIGH", "MEDIUM": "vp-MEDIUM", "LOW": "vp-LOW"}.get(p, "vp-LOW")
            transcript_html += f"""
            <div class="voice-transcript-row">
                <span class="vt-time">{entry.get('time','')}</span>
                <span class="vt-badge {tc}">{p}</span>
                <span class="vt-text">{entry.get('text','')}</span>
            </div>"""
    else:
        transcript_html = '<div style="font-size:0.72rem;color:#3a4a5c;font-family:var(--font-mono);">NO UTTERANCES YET</div>'

    st.markdown(f"""
    <div class="voice-agent-card">
        <div class="voice-status-row">
            <div class="voice-dot {dot_cls}"></div>
            <span style="font-family:'Orbitron',sans-serif;font-size:0.68rem;
                         color:{'var(--safe)' if voice_speaking else '#5a6a84'};
                         letter-spacing:0.12em;">{dot_lbl}</span>
            <div style="flex:1;"></div>
            <span style="font-size:0.6rem;font-family:var(--font-mono);color:#3a4a5c;
                         letter-spacing:0.1em;margin-right:6px;">LAST PRIORITY</span>
            {vp_badge}
        </div>
        <div style="font-size:0.6rem;font-family:var(--font-mono);color:#3a4a5c;
                    letter-spacing:0.12em;margin-bottom:6px;">UTTERANCE LOG (LAST 5)</div>
        {transcript_html}
    </div>
    """, unsafe_allow_html=True)

    # Detection manifest
    if detections:
        st.markdown('<div class="panel-title" style="margin-top:10px;">◈ DETECTION MANIFEST</div>',
                    unsafe_allow_html=True)
        det_rows = ""
        for d in detections:
            lbl  = d.get("label", "?")
            conf = d.get("confidence", 0.0)
            dist = d.get("distance",   0.0)
            tl   = d.get("traffic_light_color") or "—"
            cc   = "#1de98b" if conf > 0.7 else "#ffd454" if conf > 0.4 else "#ff8f9c"
            det_rows += f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;
                        padding:5px 0;border-bottom:1px solid rgba(0,200,255,0.06);
                        font-size:0.72rem;font-family:'Share Tech Mono',monospace;">
                <span style="color:#80b8d8;">{lbl.upper()}</span>
                <span style="color:{cc};">{conf:.0%}</span>
                <span style="color:#4a7fa8;">{dist:.1f}m</span>
                <span style="color:#4a7fa8;">{tl}</span>
            </div>"""
        st.markdown(f"""
        <div class="panel-card" style="padding:12px 14px;">
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;
                        padding:0 0 6px 0;border-bottom:1px solid rgba(0,200,255,0.15);
                        font-size:0.6rem;letter-spacing:0.12em;color:#3a4a5c;
                        font-family:'Share Tech Mono',monospace;">
                <span>OBJECT</span><span>CONF</span><span>DIST</span><span>SIGNAL</span>
            </div>
            {det_rows}
        </div>
        """, unsafe_allow_html=True)

# ── Auto-rerun loop ───────────────────────────────────────────────────────────
if st.session_state.run_loop:
    time.sleep(0.05)
    st.rerun()