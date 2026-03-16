"""
RiskAlertTool — MCP tool that generates a formatted surgeon alert
from a risk level and clinical description.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RiskAlertTool")

RISK_TEMPLATES = {
    "low": (
        "ℹ️  STABLE | {clinical_text} "
        "Confidence: {confidence:.0%}. No immediate action required."
    ),
    "medium": (
        "⚠️  ALERT | {clinical_text} "
        "Confidence: {confidence:.0%}. Review recommended."
    ),
    "high": (
        "🚨 CRITICAL | {clinical_text} "
        "Confidence: {confidence:.0%}. Immediate surgical attention required!"
    ),
}


@mcp.tool()
def generate_alert(risk_level: str, clinical_text: str, confidence: float) -> dict:
    """
    Produce a surgeon-facing alert message.

    Parameters
    ----------
    risk_level   : str    — low | medium | high
    clinical_text: str    — clinical description from ClinicalInterpreter
    confidence   : float  — model confidence (0.0 – 1.0)

    Returns
    -------
    {
        "alert":      str,
        "risk_level": str,
        "severity":   str   — INFO | WARNING | CRITICAL
    }
    """
    template = RISK_TEMPLATES.get(risk_level, RISK_TEMPLATES["medium"])
    alert    = template.format(clinical_text=clinical_text, confidence=confidence)

    severity_map = {"low": "INFO", "medium": "WARNING", "high": "CRITICAL"}

    return {
        "alert":      alert,
        "risk_level": risk_level,
        "severity":   severity_map.get(risk_level, "WARNING"),
    }
