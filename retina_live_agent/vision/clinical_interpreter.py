"""
Clinical Interpreter — maps CNN predictions to surgeon-readable alerts
with severity metadata for downstream routing.
"""

from dataclasses import dataclass

# ── Risk levels (used by AlertAgent) ──────────────────────────────────────────
RISK_LOW    = "low"
RISK_MEDIUM = "medium"
RISK_HIGH   = "high"

# ── Clinical interpretation table ─────────────────────────────────────────────
# Each entry: (clinical_text, risk_level, recommended_action)
_INTERPRETATION_TABLE: dict[str, tuple[str, str, str]] = {
    "NORMAL": (
        "Retina appears stable. No abnormalities detected.",
        RISK_LOW,
        "Continue monitoring. No immediate intervention required.",
    ),
    "DRUSEN": (
        "Subretinal deposits detected. Early degenerative changes present.",
        RISK_LOW,
        "Document findings. Schedule follow-up imaging.",
    ),
    "DME": (
        "Fluid accumulation detected in retinal layers. "
        "Diabetic macular edema suspected.",
        RISK_MEDIUM,
        "Consider anti-VEGF therapy. Avoid mechanical pressure near macular region.",
    ),
    "CNV": (
        "Abnormal vascular structures detected beneath the retina. "
        "Choroidal neovascularisation suspected.",
        RISK_HIGH,
        "Immediate evaluation required. Proceed with caution - high bleeding risk.",
    ),
}


@dataclass
class ClinicalResult:
    label: str
    confidence: float
    clinical_text: str
    risk_level: str
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "label":              self.label,
            "confidence":         self.confidence,
            "clinical_text":      self.clinical_text,
            "risk_level":         self.risk_level,
            "recommended_action": self.recommended_action,
        }


class ClinicalInterpreter:
    """Converts a raw OCT prediction into a structured clinical interpretation."""

    def interpret(self, prediction: dict) -> ClinicalResult:
        """
        Parameters
        ----------
        prediction : dict
            Output of OCTDetector.predict()
            {"label": str, "confidence": float, "all_scores": dict}

        Returns
        -------
        ClinicalResult
        """
        label      = prediction.get("label", "NORMAL")
        confidence = prediction.get("confidence", 0.0)

        # Graceful fallback for unexpected labels
        text, risk, action = _INTERPRETATION_TABLE.get(
            label,
            (
                f"Unrecognised pattern: {label}. Further review needed.",
                RISK_MEDIUM,
                "Consult specialist immediately.",
            ),
        )

        return ClinicalResult(
            label=label,
            confidence=confidence,
            clinical_text=text,
            risk_level=risk,
            recommended_action=action,
        )
