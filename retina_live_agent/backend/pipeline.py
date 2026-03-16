"""
Real-Time Pipeline — ties OCTStream → CoordinatorAgent together.

This module drives the end-to-end flow:

    OCT frame (OCTStream)
        ↓
    VisionAnalysisAgent      (OCTDetector.predict)
        ↓
    ClinicalInterpreterAgent (ClinicalInterpreter.interpret)
        ↓
    AlertAgent               (format + escalate)
        ↓
    CoordinatorAgent         (aggregate + context-push to SurgicalAssistantAgent)
        ↓
    Return:  {"alert": ..., "risk_level": ..., ...}
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from streaming.oct_stream      import OCTStream
from agents.coordinator_agent  import CoordinatorAgent

# ── Logging setup ─────────────────────────────────────────────────────────────
_log_file = Path(__file__).resolve().parents[1] / "retinalive.log"
_fmt       = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                datefmt="%H:%M:%S")

_file_handler   = logging.FileHandler(_log_file, mode="w", encoding="utf-8")
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)

log = logging.getLogger("retinalive")
log.setLevel(logging.INFO)
log.addHandler(_file_handler)
log.addHandler(_stream_handler)
log.propagate = False


class RetinaLivePipeline:
    """
    High-level façade used by the Streamlit dashboard and the API server.

    Usage
    -----
    pipeline = RetinaLivePipeline(on_result=my_callback)
    pipeline.start()
    ...
    pipeline.stop()
    result = await pipeline.query("What did you detect?")
    """

    def __init__(
        self,
        on_result: Optional[Callable[[dict], None]] = None,
        video_path: Optional[str] = None,
    ):
        self.on_result   = on_result
        self.coordinator = CoordinatorAgent(name="CoordinatorAgent", description="Orchestrator")
        self._loop       = asyncio.new_event_loop()
        self._thread: Optional[threading.Thread] = None

        stream_kwargs: dict = {}
        if video_path:
            from pathlib import Path
            stream_kwargs["video_path"] = Path(video_path)

        self.stream = OCTStream(
            on_frame=self._handle_frame,
            **stream_kwargs,
        )

    # ── Start / stop ───────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the async event loop in a background thread, then start streaming."""
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        self.stream.start()
        log.info("=" * 60)
        log.info("RetinaLive pipeline STARTED")
        log.info("=" * 60)

    def stop(self) -> None:
        self.stream.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Pipeline STOPPED")

    # ── Frame handler ──────────────────────────────────────────────────────────
    def _handle_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Called from OCTStream thread — dispatches async work to event loop."""
        future = asyncio.run_coroutine_threadsafe(
            self._process_frame(frame, frame_idx),
            self._loop,
        )
        future.add_done_callback(lambda f: None)

    async def _process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        result = await self.coordinator.run_frame(frame)
        result["frame_idx"] = frame_idx

        # ── Per-frame log ────────────────────────────────────────────────────
        risk  = result.get("risk_level", "?").upper()
        label = result.get("label", "?")
        conf  = result.get("confidence", 0)
        alert = result.get("alert", "")
        lvl   = logging.CRITICAL if risk == "HIGH" else \
                logging.WARNING  if risk == "MEDIUM" else \
                logging.INFO
        log.log(lvl,
            "Frame %04d | %-8s | %4.0f%% | Risk: %-8s | %s",
            frame_idx, label, conf * 100, risk, alert[:60]
        )

        if self.on_result:
            self.on_result(result)
        return result

    # ── Query ──────────────────────────────────────────────────────────────────
    async def query(self, question: str) -> str:
        return await self.coordinator.query(question)

    def query_sync(self, question: str) -> str:
        """Blocking version for use from synchronous contexts (e.g. Streamlit)."""
        future = asyncio.run_coroutine_threadsafe(self.query(question), self._loop)
        return future.result(timeout=30)

    @property
    def latest_result(self) -> dict:
        return self.coordinator.latest_result

    # ── Internal ───────────────────────────────────────────────────────────────
    def _start_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
