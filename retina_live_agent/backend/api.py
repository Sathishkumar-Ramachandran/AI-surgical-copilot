"""
FastAPI backend — exposes the pipeline over HTTP for external clients.

Endpoints
---------
POST /analyze          — analyse a single uploaded frame
GET  /latest           — return the latest pipeline result
POST /query            — send a text question to SurgicalAssistantAgent
POST /stream/start     — start streaming from video path
POST /stream/stop      — stop streaming
GET  /health           — health check

Run with:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.pipeline import RetinaLivePipeline

app = FastAPI(title="RetinaLive API", version="1.0.0")

# Single global pipeline instance (started on demand)
_pipeline: Optional[RetinaLivePipeline] = None
_results_log: list[dict] = []


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_pipeline() -> RetinaLivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetinaLivePipeline(on_result=lambda r: _results_log.append(r))
    return _pipeline


# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class StreamRequest(BaseModel):
    video_path: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "RetinaLive"}


@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    """Upload a single OCT image and get an instant analysis."""
    contents = await file.read()
    arr   = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipeline = _get_pipeline()
    import asyncio
    result = await pipeline.coordinator.run_frame(frame)
    return JSONResponse(content=result)


@app.get("/latest")
def get_latest():
    pipeline = _get_pipeline()
    result   = pipeline.latest_result
    if not result:
        return JSONResponse(content={"message": "No analysis yet."})
    return JSONResponse(content=result)


@app.post("/query")
async def query_agent(req: QueryRequest):
    pipeline = _get_pipeline()
    answer   = await pipeline.query(req.question)
    return {"question": req.question, "response": answer}


@app.post("/stream/start")
def start_stream(req: StreamRequest):
    global _pipeline
    _pipeline = RetinaLivePipeline(
        on_result=lambda r: _results_log.append(r),
        video_path=req.video_path,
    )
    _pipeline.start()
    return {"status": "streaming started"}


@app.post("/stream/stop")
def stop_stream():
    global _pipeline
    if _pipeline:
        _pipeline.stop()
        _pipeline = None
    return {"status": "streaming stopped"}


@app.get("/logs")
def get_logs(limit: int = 20):
    return _results_log[-limit:]
