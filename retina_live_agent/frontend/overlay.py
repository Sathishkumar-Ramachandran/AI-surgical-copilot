"""
Overlay — draws clinical alerts and bounding boxes onto OCT frames using OpenCV.

The overlay is severity-aware:
  - CRITICAL  → red box + flashing text
  - WARNING   → orange box + bold text
  - INFO/STABLE → green box + regular text
"""

import cv2
import numpy as np
import time

# ── Colour palette (BGR) ───────────────────────────────────────────────────────
COLOURS = {
    "CRITICAL": (0,   0,   220),   # red
    "WARNING":  (0,   140, 255),   # orange
    "INFO":     (50,  200,  50),   # green
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


def draw_alert_overlay(
    frame: np.ndarray,
    result: dict,
    bounding_box: tuple | None = None,
) -> np.ndarray:
    """
    Render analysis results onto a copy of *frame*.

    Parameters
    ----------
    frame        : BGR numpy array
    result       : pipeline output dict
    bounding_box : optional (x, y, w, h) for abnormality region

    Returns
    -------
    annotated BGR numpy array (same shape as input)
    """
    canvas   = frame.copy()
    severity = result.get("severity", "INFO")
    colour   = COLOURS.get(severity, COLOURS["INFO"])

    h, w = canvas.shape[:2]

    # ── Bounding box ───────────────────────────────────────────────────────────
    if bounding_box:
        x, y, bw, bh = bounding_box
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), colour, THICKNESS)
    elif severity != "INFO":
        # Default: highlight central macular region (rough estimate)
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.25), int(h * 0.20)
        cv2.rectangle(
            canvas,
            (cx - rw, cy - rh),
            (cx + rw, cy + rh),
            colour,
            THICKNESS,
        )
        # Diagonal cross-hairs for dramatic effect
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), colour, 1)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), colour, 1)

    # ── Alert banner ───────────────────────────────────────────────────────────
    alert_text  = result.get("alert", "")[:80]          # truncate long strings
    label_text  = f"  {result.get('label', '')}  {result.get('confidence', 0):.0%}"
    risk_text   = f"Risk: {result.get('risk_level', '').upper()}"

    # Semi-transparent black banner at top
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    # Flashing effect for CRITICAL (every 500 ms)
    flash = (severity == "CRITICAL") and (int(time.time() * 2) % 2 == 0)
    text_colour = colour if not flash else (255, 255, 255)

    cv2.putText(canvas, alert_text,  (10, 22), FONT, FONT_SCALE,       text_colour,  THICKNESS)
    cv2.putText(canvas, label_text,  (10, 45), FONT, FONT_SCALE - 0.1, (200, 200, 200), 1)

    # Bottom-right risk badge
    (tw, th), _ = cv2.getTextSize(risk_text, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(canvas, (w - tw - 20, h - 35), (w, h), colour, -1)
    cv2.putText(canvas, risk_text, (w - tw - 10, h - 10), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)

    return canvas


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR numpy array to JPEG bytes for streaming."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
