"""
CoordinatorAgent — ADK orchestrator that wires the full pipeline and
exposes run_frame() / query() as direct async methods.
"""

import numpy as np
from google.adk.agents import BaseAgent
from google.adk.events import Event
from pydantic import PrivateAttr

from agents.vision_analysis_agent       import VisionAnalysisAgent
from agents.clinical_interpreter_agent  import ClinicalInterpreterAgent
from agents.alert_agent                 import AlertAgent
from agents.surgical_assistant_agent    import SurgicalAssistantAgent

RISK_CONFIG = {
    "low":    {"severity": "INFO",     "prefix": "INFO  STABLE"},
    "medium": {"severity": "WARNING",  "prefix": "ALERT"},
    "high":   {"severity": "CRITICAL", "prefix": "CRITICAL"},
}


class CoordinatorAgent(BaseAgent):
    """Top-level orchestrator for the RetinaLive pipeline."""

    # Declare sub-agents as pydantic fields so ADK can introspect them
    sub_agents: list = []

    _vision:      VisionAnalysisAgent      = PrivateAttr()
    _interpreter: ClinicalInterpreterAgent = PrivateAttr()
    _alerter:     AlertAgent               = PrivateAttr()
    _assistant:   SurgicalAssistantAgent   = PrivateAttr()
    _latest:      dict                     = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        self._vision      = VisionAnalysisAgent(name="VisionAnalysisAgent",
                                                 description="CNN inference")
        self._interpreter = ClinicalInterpreterAgent(name="ClinicalInterpreterAgent",
                                                      description="Clinical mapping")
        self._alerter     = AlertAgent(name="AlertAgent",
                                        description="Alert generation")
        self._assistant   = SurgicalAssistantAgent(name="SurgicalAssistantAgent",
                                                    description="Gemini voice interface")

    # ── Core pipeline ──────────────────────────────────────────────────────────
    async def run_frame(self, frame: np.ndarray) -> dict:
        """
        Process one OCT frame through the full agent pipeline.

        Returns
        -------
        {
            "alert", "risk_level", "severity", "label",
            "confidence", "clinical_text", "recommended_action", "all_scores"
        }
        """
        # Step 1 — Vision
        prediction = self._vision.predict(frame)

        # Step 2 — Clinical interpretation
        clinical      = self._interpreter.interpret(prediction)
        clinical_dict = clinical.to_dict()

        # Step 3 — Alert
        alert_data = self._alerter.build_alert(clinical_dict)

        result = {
            **alert_data,
            "clinical_text": clinical_dict["clinical_text"],
            "all_scores":    prediction.get("all_scores", {}),
        }

        self._latest = result
        self._assistant.update_analysis_context(result)
        return result

    # ── Query ──────────────────────────────────────────────────────────────────
    async def query(self, question: str) -> str:
        return await self._assistant.ask_text(question)

    @property
    def latest_result(self) -> dict:
        return self._latest

    # ── ADK stub ───────────────────────────────────────────────────────────────
    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=None)
