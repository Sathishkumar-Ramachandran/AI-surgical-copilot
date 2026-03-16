"""
MCP Server — registers all RetinaLive tools and starts the MCP server.

Run with:
    python mcp/mcp_server.py

The server exposes three tools over stdio (default) or SSE transport:
    - analyze_oct_frame
    - interpret_prediction
    - generate_alert
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

# Import tool registrations (each module registers its tool on import)
from mcp.tools.oct_analysis_tool         import mcp as oct_mcp          # noqa: F401
from mcp.tools.clinical_interpretation_tool import mcp as clinical_mcp  # noqa: F401
from mcp.tools.risk_alert_tool           import mcp as alert_mcp         # noqa: F401

# ── Unified server ─────────────────────────────────────────────────────────────
server = FastMCP("RetinaLive-MCP")


# Re-register all tools on the unified server
@server.tool()
def analyze_oct_frame(image_b64: str) -> dict:
    from mcp.tools.oct_analysis_tool import analyze_oct_frame as _fn
    return _fn(image_b64)


@server.tool()
def interpret_prediction(label: str, confidence: float) -> dict:
    from mcp.tools.clinical_interpretation_tool import interpret_prediction as _fn
    return _fn(label, confidence)


@server.tool()
def generate_alert(risk_level: str, clinical_text: str, confidence: float) -> dict:
    from mcp.tools.risk_alert_tool import generate_alert as _fn
    return _fn(risk_level, clinical_text, confidence)


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    print(f"[MCP Server] Starting RetinaLive MCP server (transport={transport})")
    server.run(transport=transport)
