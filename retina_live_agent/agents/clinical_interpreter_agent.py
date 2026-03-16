"""
ClinicalInterpreterAgent — ADK agent that maps raw predictions to clinical alerts.
"""

from google.adk.agents import BaseAgent
from google.adk.events import Event
from pydantic import PrivateAttr

from vision.clinical_interpreter import ClinicalInterpreter, ClinicalResult


class ClinicalInterpreterAgent(BaseAgent):
    """Translates CNN predictions into structured clinical alerts."""

    _interpreter: ClinicalInterpreter = PrivateAttr()

    def model_post_init(self, __context):
        self._interpreter = ClinicalInterpreter()

    def interpret(self, prediction: dict) -> ClinicalResult:
        return self._interpreter.interpret(prediction)

    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=None)
