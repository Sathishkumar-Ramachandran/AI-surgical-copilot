"""
OCTAnalysisTool — MCP tool that accepts a base64-encoded image frame
and returns a prediction from the OCT detector.
"""

import base64
import numpy as np
import cv2
from mcp.server.fastmcp import FastMCP
from vision.oct_detector import OCTDetector

mcp = FastMCP("OCTAnalysisTool")
_detector = OCTDetector()


@mcp.tool()
def analyze_oct_frame(image_b64: str) -> dict:
    """
    Analyse a single OCT frame.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded JPEG/PNG image of the OCT frame.

    Returns
    -------
    {
        "label":      str,    # NORMAL | DRUSEN | DME | CNV
        "confidence": float,
        "all_scores": dict
    }
    """
    img_bytes = base64.b64decode(image_b64)
    arr       = np.frombuffer(img_bytes, dtype=np.uint8)
    frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to decode image. Ensure base64 encodes a valid JPEG/PNG."}

    return _detector.predict(frame)
