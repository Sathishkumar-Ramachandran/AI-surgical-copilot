"""
OCT Detector — loads trained CNN model and runs per-frame predictions.
Expected model: a Keras/TensorFlow SavedModel stored in models/oct_cnn_model/
"""

import numpy as np
import cv2
from pathlib import Path

# Try to import TensorFlow; fall back to a stub for environments without GPU/TF
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_LABELS = ["NORMAL", "DRUSEN", "DME", "CNV"]
IMG_SIZE = (224, 224)           # must match training input shape
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "oct_cnn_model"


# ── Model loader ───────────────────────────────────────────────────────────────
class OCTDetector:
    """Wraps the trained CNN model with preprocessing + prediction helpers."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if not TF_AVAILABLE:
            print("[OCTDetector] TensorFlow not available — running in stub mode.")
            return

        if not self.model_path.exists():
            print(
                f"[OCTDetector] Model not found at {self.model_path}. "
                "Running in stub mode (random predictions)."
            )
            return

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            print(f"[OCTDetector] Model loaded from {self.model_path}")
        except Exception as exc:
            print(f"[OCTDetector] Failed to load model: {exc}. Stub mode active.")

    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize + normalise a raw BGR/RGB frame for the CNN.
        Returns a batch tensor of shape (1, H, W, 3).
        """
        resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else resized
        normalised = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalised, axis=0)

    # ------------------------------------------------------------------
    def predict(self, frame: np.ndarray) -> dict:
        """
        Run inference on a single frame.

        Returns
        -------
        {
            "label":      "DME",
            "confidence": 0.91,
            "all_scores": {"NORMAL": 0.03, "DRUSEN": 0.04, "DME": 0.91, "CNV": 0.02}
        }
        """
        if self.model is None:
            return self._stub_prediction()

        tensor = self.preprocess(frame)
        raw = self.model.predict(tensor, verbose=0)[0]          # shape (4,)
        scores = {cls: float(raw[i]) for i, cls in enumerate(CLASS_LABELS)}
        best_idx = int(np.argmax(raw))
        return {
            "label":      CLASS_LABELS[best_idx],
            "confidence": float(raw[best_idx]),
            "all_scores": scores,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _stub_prediction() -> dict:
        """Return a plausible random prediction when the model is unavailable."""
        probs = np.random.dirichlet(np.ones(4))
        best = int(np.argmax(probs))
        return {
            "label":      CLASS_LABELS[best],
            "confidence": float(probs[best]),
            "all_scores": {cls: float(probs[i]) for i, cls in enumerate(CLASS_LABELS)},
        }
