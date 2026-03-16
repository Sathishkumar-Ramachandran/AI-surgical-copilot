"""
ClinicalInterpretationTool — MCP tool that converts a model prediction
into a structured clinical interpretation.
"""

from mcp.server.fastmcp import FastMCP
from vision.clinical_interpreter import ClinicalInterpreter

mcp = FastMCP("ClinicalInterpretationTool")
_interpreter = ClinicalInterpreter()


@mcp.tool()
def interpret_prediction(label: str, confidence: float) -> dict:
    """
    Map a CNN prediction label to a clinical explanation and risk level.

    Parameters
    ----------
    label      : str    — one of NORMAL, DRUSEN, DME, CNV
    confidence : float  — model confidence score (0.0 – 1.0)

    Returns
    -------
    {
        "label":              str,
        "confidence":         float,
        "clinical_text":      str,
        "risk_level":         str,   # low | medium | high
        "recommended_action": str
    }
    """
    prediction = {"label": label, "confidence": confidence}
    result = _interpreter.interpret(prediction)
    return result.to_dict()
