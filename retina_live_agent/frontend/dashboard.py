"""
RetinaLive — Streamlit Demo Dashboard

Threading model (key design)
-----------------------------
Streamlit re-executes this entire module on EVERY rerun.
Module-level variables are therefore re-created each time — unusable for
cross-thread state.

Solution: store the shared mutable objects (frame_queue, result_store) in
st.session_state, which DOES persist across reruns.  Background threads
receive direct object references captured at Start time — they never touch
st.session_state themselves (which would crash).

  session_state["frame_queue"]  → queue.Queue  — background writes, main reads
  session_state["result_store"] → dict          — background writes, main reads
  session_state["pipeline"]     → pipeline ref  — main thread only
  session_state["running"]      → bool          — main thread only
  _lock (module-level)          → threading.Lock — guards result_store writes
"""

import sys
import os
import time
import queue
import asyncio
import threading
from pathlib import Path

# ── Load .env before anything else ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st
import numpy as np
import cv2

from backend.pipeline    import RetinaLivePipeline
from frontend.overlay    import draw_alert_overlay
from audio_recorder_streamlit import audio_recorder

# ── Module-level lock only (primitives are safe to keep at module level) ───────
_lock = threading.Lock()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaLive — AI Surgical Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state — persists across reruns ────────────────────────────────────
def _init_state():
    defaults = {
        "pipeline":                None,
        "running":                 False,
        # Shared mutable containers — background threads hold direct refs to these
        "frame_queue":             queue.Queue(maxsize=5),
        "result_store":            {"latest": {}, "log": [], "latest_raw_frame": None},
        "chat_history":            [],
        "audio_answer_wav":        None,
        # Training Mode state
        "training_explanation_wav": None,
        "training_frame_rgb":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Callback factory — captures session_state objects by direct reference ─────
def _make_callbacks(frame_q: queue.Queue, result_store: dict, pipeline_ref: list):
    """
    Returns (on_result, on_frame) closures that hold direct object references.
    Background threads call these — NO st.session_state access inside.

    pipeline_ref is a single-element list so the on_frame closure can read
    the pipeline object after it's been assigned (mutable container trick).
    """
    def on_result(result: dict) -> None:
        with _lock:
            result_store["latest"] = result
            result_store["log"].append(result)
            if len(result_store["log"]) > 50:
                result_store["log"].pop(0)

    def on_frame(frame: np.ndarray, frame_idx: int) -> None:
        # 1. Forward to pipeline AI chain (non-blocking async dispatch)
        p = pipeline_ref[0]
        if p is not None:
            p._handle_frame(frame, frame_idx)

        # 2. Store raw frame for Training Mode capture
        with _lock:
            result_store["latest_raw_frame"] = frame.copy()
            result = dict(result_store["latest"])

        # 3. Overlay latest result and push to display queue
        annotated = draw_alert_overlay(frame, result) if result else frame
        try:
            frame_q.put_nowait((annotated, frame_idx))
        except queue.Full:
            pass

    return on_result, on_frame


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RetinaLive")
    st.caption("AI Surgical Assistant for OCT Monitoring")

    _api_ok = bool(os.environ.get("GOOGLE_API_KEY"))
    st.caption("✅ API key loaded" if _api_ok else "❌ GOOGLE_API_KEY missing in .env")

    uploaded = st.file_uploader("Upload OCT Video (.mp4)", type=["mp4", "avi", "mov"])
    video_path_input = None
    if uploaded:
        save_path = _PROJECT_ROOT / "data" / uploaded.name
        save_path.parent.mkdir(exist_ok=True)
        save_path.write_bytes(uploaded.read())
        video_path_input = str(save_path)
        st.success(f"Video ready: {save_path.name}")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", type="primary",
                               disabled=st.session_state.running)
    with col2:
        stop_btn = st.button("⏹ Stop", disabled=not st.session_state.running)

    st.divider()
    st.caption("RetinaLive v1.0 — Gemini Live 2.5")


# ── Start ──────────────────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    # Fresh containers stored in session_state so they survive reruns
    frame_q      = queue.Queue(maxsize=5)
    result_store = {"latest": {}, "log": [], "latest_raw_frame": None}
    st.session_state.frame_queue  = frame_q
    st.session_state.result_store = result_store

    pipeline_ref = [None]   # mutable container; set after pipeline is created
    on_result, on_frame = _make_callbacks(frame_q, result_store, pipeline_ref)

    pipeline = RetinaLivePipeline(
        on_result=on_result,
        video_path=video_path_input,
    )
    pipeline_ref[0] = pipeline  # now the closure can see it
    # Replace stream's default on_frame with our combined AI+display handler
    pipeline.stream.on_frame = on_frame
    pipeline.start()

    st.session_state.pipeline = pipeline
    st.session_state.running  = True
    st.rerun()


# ── Stop ───────────────────────────────────────────────────────────────────────
if stop_btn and st.session_state.running:
    if st.session_state.pipeline:
        st.session_state.pipeline.stop()
    st.session_state.pipeline = None
    st.session_state.running  = False
    st.rerun()


# ── Read shared state (main thread snapshot) ───────────────────────────────────
with _lock:
    _snap_result    = dict(st.session_state.result_store.get("latest", {}))
    _snap_log       = list(st.session_state.result_store.get("log", []))
    _snap_raw_frame = st.session_state.result_store.get("latest_raw_frame")


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("## 👁️ RetinaLive — Real-Time AI Surgical Assistant")

tab_surgical, tab_training = st.tabs(["🔬 Surgical Mode", "🎓 Training Mode"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SURGICAL MODE (all existing features)
# ══════════════════════════════════════════════════════════════════════════════
with tab_surgical:

    col_feed, col_alerts = st.columns([2, 1])

    # ── Left: Live OCT Feed ────────────────────────────────────────────────────
    with col_feed:
        st.subheader("Live OCT Feed")
        frame_placeholder = st.empty()

        if st.session_state.running:
            try:
                frame, _ = st.session_state.frame_queue.get(timeout=0.4)
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Live OCT Stream",
                    width="stretch",
                )
            except queue.Empty:
                frame_placeholder.info("Waiting for frames…")
        else:
            frame_placeholder.info("Upload a video and click ▶ Start.")

    # ── Right: Alert Panel ─────────────────────────────────────────────────────
    with col_alerts:
        st.subheader("Surgical Alert Panel")

        if _snap_result:
            sev  = _snap_result.get("severity", "INFO")
            icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🟢"}.get(sev, "⚪")

            st.metric(
                label=f"{icon} Finding",
                value=_snap_result.get("label", "—"),
                delta=f"{_snap_result.get('confidence', 0):.0%} confidence",
            )
            if sev == "CRITICAL":
                st.error(_snap_result.get("alert", ""))
            elif sev == "WARNING":
                st.warning(_snap_result.get("alert", ""))
            else:
                st.success(_snap_result.get("alert", ""))

            with st.expander("Recommended Action"):
                st.write(_snap_result.get("recommended_action", "—"))

            with st.expander("Score Breakdown"):
                for cls, score in _snap_result.get("all_scores", {}).items():
                    st.progress(score, text=f"{cls}: {score:.2%}")
        else:
            st.info("No analysis results yet.")


    # ── Voice Q&A — Gemini Live API ────────────────────────────────────────────
    st.divider()
    st.subheader("🎙️ Surgeon Voice Interface — Gemini Live 2.5")

    pipeline_ready = (
        st.session_state.running
        and st.session_state.pipeline is not None
        and bool(_snap_result)
    )

    if not pipeline_ready:
        st.info(
            "Voice Q&A unlocks once the pipeline is running and has analysed a frame. "
            "Gemini Live will only answer questions grounded in the active OCT scan."
        )
    else:
        if not _api_ok:
            st.warning("GOOGLE_API_KEY is missing in .env — voice responses unavailable.")
        else:
            st.caption(
                "Press **record**, ask your question about the OCT scan, then press **stop**. "
                "Gemini Live answers **only** from the current OCT analysis."
            )

            with st.expander("Context Gemini is using", expanded=False):
                assistant = st.session_state.pipeline.coordinator._assistant
                st.code(assistant._build_context_block(), language=None)

            wav_bytes = audio_recorder(
                text="",
                recording_color="#e87070",
                neutral_color="#6aa36f",
                icon_size="2x",
                key="mic_recorder",
            )

            if wav_bytes and len(wav_bytes) > 44:
                with st.spinner("Sending to Gemini Live (gemini-2.5-flash-native-audio)…"):
                    from agents.surgical_assistant_agent import SurgicalAssistantAgent
                    assistant = st.session_state.pipeline.coordinator._assistant
                    try:
                        pcm_response = asyncio.run(assistant.ask_audio_live(wav_bytes))
                        if pcm_response:
                            st.session_state.audio_answer_wav = \
                                SurgicalAssistantAgent.pcm24_to_wav(pcm_response)
                        else:
                            st.error("Gemini returned an empty response. Check API key.")
                    except Exception as exc:
                        st.error(f"Gemini Live error: {exc}")

            if st.session_state.audio_answer_wav:
                st.markdown("**Gemini Live response:**")
                st.audio(st.session_state.audio_answer_wav, format="audio/wav")
                if st.button("Clear response"):
                    st.session_state.audio_answer_wav = None
                    st.rerun()


    # ── Text Q&A fallback ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Text Q&A (typed fallback)")
    st.caption("Grounded exclusively in the current OCT scan.")

    with st.form("qa_form", clear_on_submit=True):
        question  = st.text_input("Your question",
                                   placeholder="e.g. What is the current risk level?")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        if not pipeline_ready:
            st.warning("Start the pipeline with an uploaded video first.")
        elif not _api_ok:
            st.warning("GOOGLE_API_KEY is missing in .env.")
        else:
            with st.spinner("Consulting Gemini…"):
                answer = st.session_state.pipeline.query_sync(question)
            st.session_state.chat_history.append(("surgeon", question))
            st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history[-10:]:
        with st.chat_message(role if role == "assistant" else "user"):
            st.write(msg)


    # ── Analysis Log ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Analysis Log")
    if _snap_log:
        for entry in reversed(_snap_log[-10:]):
            fi   = entry.get("frame_idx", "?")
            lbl  = entry.get("label", "—")
            conf = entry.get("confidence", 0)
            risk = entry.get("risk_level", "—")
            st.text(f"Frame {fi:>5} | {lbl:<8} | {conf:.0%} | Risk: {risk}")
    else:
        st.caption("No frames analysed yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab_training:
    st.subheader("🎓 OCT Training Mode")
    st.caption(
        "Capture any frame from the live stream and have Gemini explain the "
        "OCT finding aloud — covering anatomy, pathology, and clinical significance."
    )

    training_ready = (
        st.session_state.running
        and st.session_state.pipeline is not None
        and _snap_raw_frame is not None
    )

    if not training_ready:
        st.info("Start the pipeline with an uploaded video to enable Training Mode.")
    else:
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            capture_btn = st.button(
                "📸 Capture & Explain",
                type="primary",
                help="Grabs the current frame and generates an audio explanation",
            )

        if capture_btn:
            if not _api_ok:
                st.warning("GOOGLE_API_KEY is missing in .env.")
            else:
                # Snapshot the raw frame right now
                with _lock:
                    raw = st.session_state.result_store.get("latest_raw_frame")

                if raw is not None:
                    # Convert BGR→RGB and store for display
                    st.session_state.training_frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    # Reset previous explanation
                    st.session_state.training_explanation_wav = None

                    with st.spinner("Gemini is preparing your educational explanation…"):
                        from agents.surgical_assistant_agent import SurgicalAssistantAgent
                        assistant = st.session_state.pipeline.coordinator._assistant
                        try:
                            pcm = asyncio.run(assistant.explain_frame_audio())
                            if pcm:
                                st.session_state.training_explanation_wav = \
                                    SurgicalAssistantAgent.pcm24_to_wav(pcm)
                            else:
                                st.error("Gemini returned an empty explanation. Check API key.")
                        except Exception as exc:
                            st.error(f"Gemini Live error: {exc}")

        # ── Display captured frame + audio explanation ─────────────────────────
        if st.session_state.training_frame_rgb is not None:
            st.divider()
            col_img, col_audio = st.columns([1, 1])

            with col_img:
                st.markdown("**Captured OCT Frame**")
                # Show the finding label as a caption if available
                label = _snap_result.get("label", "")
                conf  = _snap_result.get("confidence", 0)
                caption = f"{label} — {conf:.0%} confidence" if label else "OCT Frame"
                st.image(
                    st.session_state.training_frame_rgb,
                    caption=caption,
                    use_container_width=True,
                )

            with col_audio:
                st.markdown("**Audio Explanation**")
                if st.session_state.training_explanation_wav:
                    st.audio(
                        st.session_state.training_explanation_wav,
                        format="audio/wav",
                        autoplay=True,
                    )
                    with st.expander("OCT Analysis used for explanation"):
                        assistant = st.session_state.pipeline.coordinator._assistant
                        st.code(assistant._build_context_block(), language=None)

                    if st.button("Clear explanation"):
                        st.session_state.training_explanation_wav = None
                        st.session_state.training_frame_rgb = None
                        st.rerun()
                else:
                    st.info("Click **Capture & Explain** to generate an audio explanation.")


# ── Auto-refresh while running ─────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.3)
    st.rerun()
