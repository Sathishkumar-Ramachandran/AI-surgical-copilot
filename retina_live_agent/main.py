"""
RetinaLive — entry point.

Usage
-----
# Start the Streamlit dashboard (recommended for demo)
python main.py dashboard

# Start the FastAPI REST API
python main.py api

# Run the pipeline once (headless, 30 s)
python main.py headless

# Start the MCP server
python main.py mcp
"""

import sys
import time
import asyncio
from pathlib import Path


def run_dashboard():
    import subprocess
    dashboard = Path(__file__).parent / "frontend" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard)],
        check=True,
    )


def run_api():
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def run_headless(duration: int = 30):
    from backend.pipeline import RetinaLivePipeline

    def on_result(r):
        print(
            f"[{r.get('frame_idx', '?'):>4}] "
            f"{r.get('label', '?'):<8} "
            f"{r.get('confidence', 0):.0%} "
            f"| {r.get('risk_level', '?').upper():<8} "
            f"| {r.get('clinical_text', '')[:60]}"
        )

    pipeline = RetinaLivePipeline(on_result=on_result)
    pipeline.start()
    print(f"[Main] Headless pipeline running for {duration}s…")
    time.sleep(duration)
    pipeline.stop()


def run_mcp():
    """Start the RetinaLive MCP server (stdio transport)."""
    from mcp.mcp_server import server
    server.run(transport="stdio")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dashboard"

    if cmd == "dashboard":
        run_dashboard()
    elif cmd == "api":
        run_api()
    elif cmd == "headless":
        dur = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        run_headless(dur)
    elif cmd == "mcp":
        run_mcp()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: dashboard | api | headless | mcp")
        sys.exit(1)
