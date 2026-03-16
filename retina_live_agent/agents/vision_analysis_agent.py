"""
VisionAnalysisAgent — ADK agent that receives raw OCT frames and returns
a structured prediction via the OCTDetector.
"""

import numpy as np
from google.adk.agents import BaseAgent
from google.adk.events import Event
from pydantic import PrivateAttr

from vision.oct_detector import OCTDetector


class VisionAnalysisAgent(BaseAgent):
    """Consumes OCT frames and emits prediction events."""

    # Private non-pydantic state
    _detector: OCTDetector = PrivateAttr()

    def model_post_init(self, __context):
        self._detector = OCTDetector()

    def predict(self, frame: np.ndarray) -> dict:
        return self._detector.predict(frame)

    async def _run_async_impl(self, ctx):
        """ADK event loop — not used in direct pipeline mode."""
        yield Event(author=self.name, content=None)
