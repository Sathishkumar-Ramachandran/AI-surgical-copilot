"""
AlertAgent — ADK agent that generates surgeon-facing alert messages
and decides escalation based on risk level.
"""

import logging
from google.adk.agents import BaseAgent
from google.adk.events import Event

logger = logging.getLogger(__name__)

RISK_CONFIG = {
    "low":    {"severity": "INFO",     "prefix": "INFO  STABLE"},
    "medium": {"severity": "WARNING",  "prefix": "ALERT"},
    "high":   {"severity": "CRITICAL", "prefix": "CRITICAL"},
}


class AlertAgent(BaseAgent):
    """Generates escalation alerts for the surgical team."""

    def build_alert(self, clinical_dict: dict) -> dict:
        risk   = clinical_dict.get("risk_level", "low")
        config = RISK_CONFIG.get(risk, RISK_CONFIG["medium"])
        prefix = config["prefix"]

        alert_msg = (
            f"{prefix} | {clinical_dict.get('clinical_text', '')} "
            f"(Confidence: {clinical_dict.get('confidence', 0):.0%})"
        )

        if risk == "high":
            logger.critical(alert_msg)
        elif risk == "medium":
            logger.warning(alert_msg)
        else:
            logger.info(alert_msg)

        return {
            "alert":              alert_msg,
            "risk_level":         risk,
            "severity":           config["severity"],
            "label":              clinical_dict.get("label", ""),
            "confidence":         clinical_dict.get("confidence", 0.0),
            "recommended_action": clinical_dict.get("recommended_action", ""),
        }

    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=None)
