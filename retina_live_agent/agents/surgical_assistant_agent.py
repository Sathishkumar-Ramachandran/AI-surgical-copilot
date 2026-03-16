"""
SurgicalAssistantAgent — Gemini Live API voice interface for the surgeon.

Audio flow:
    Mic (WAV bytes from audio-recorder-streamlit)
        → decode + resample to PCM 16 kHz mono  (wave + numpy, Python 3.13 safe)
        → Gemini Live API  (gemini-2.0-flash-live-001, AUDIO modality)
        → PCM 24 kHz response
        → wrap in WAV header  → st.audio() playback

Grounding contract:
    Every session injects the current OCT analysis as the ONLY permitted
    knowledge source.  Gemini is explicitly forbidden to use any other
    medical knowledge.

Requires:  GOOGLE_API_KEY environment variable.
"""

import asyncio
import io
import os
import queue
import wave
from pathlib import Path
from typing import Optional

# Load .env so the backend process sees GOOGLE_API_KEY without needing the UI
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google.adk.agents import BaseAgent
from google.adk.events import Event
from pydantic import PrivateAttr

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

MODEL_ID    = "gemini-2.5-flash-native-audio-latest"
SAMPLE_RATE_IN  = 16_000   # Hz — Gemini Live input requirement
SAMPLE_RATE_OUT = 24_000   # Hz — Gemini Live audio output rate

# ── Strict grounding system prompt (surgical mode) ────────────────────────────
SYSTEM_PROMPT = """
You are RetinaLive, an intraoperative AI assistant for real-time OCT monitoring.

STRICT GROUNDING RULE — THIS IS NON-NEGOTIABLE:
You may ONLY answer using information present in the "Current OCT Analysis Context"
block that will be provided to you. You must NOT use general medical knowledge,
training data, or any information outside that context block.

If the context block says "No OCT analysis available yet", respond ONLY with:
"No OCT scan data is available. Please upload a video and start the pipeline."

If the surgeon asks anything that cannot be answered from the context block alone,
respond ONLY with:
"I can only answer based on the current OCT scan. That detail is not in the
pipeline output."

When context IS available, follow these rules:
- Prefix CRITICAL findings (risk_level = high) with "ALERT:".
- Keep responses under 3 sentences unless asked for more detail.
- Always cite the specific label, confidence, and risk level from the context.
- Never speculate, infer, or extrapolate beyond what the context states.
"""


class SurgicalAssistantAgent(BaseAgent):
    """Gemini Live–powered voice interface for the operating surgeon."""

    _analysis_ctx:  list   = PrivateAttr(default_factory=list)
    _audio_queue:   object = PrivateAttr(default_factory=queue.Queue)
    _session:       object = PrivateAttr(default=None)
    _client:        object = PrivateAttr(default=None)
    _context_window: int   = PrivateAttr(default=5)

    def model_post_init(self, __context):
        if GENAI_AVAILABLE:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if api_key:
                self._client = genai.Client(api_key=api_key)

    # ── OCT context management ─────────────────────────────────────────────────
    def update_analysis_context(self, result: dict) -> None:
        """Push the latest OCT pipeline result into the rolling context window."""
        self._analysis_ctx.append(result)
        if len(self._analysis_ctx) > self._context_window:
            self._analysis_ctx.pop(0)

    def _build_context_block(self) -> str:
        if not self._analysis_ctx:
            return "No OCT analysis available yet."
        latest = self._analysis_ctx[-1]
        return (
            f"Finding:            {latest.get('label', 'N/A')}\n"
            f"Confidence:         {latest.get('confidence', 0):.0%}\n"
            f"Clinical summary:   {latest.get('clinical_text', 'N/A')}\n"
            f"Risk level:         {latest.get('risk_level', 'N/A')}\n"
            f"Recommended action: {latest.get('recommended_action', 'N/A')}"
        )

    def _build_grounded_turn(self) -> str:
        """
        The FIRST message sent to Gemini in every Live session.
        Binds Gemini to the current OCT context as its sole knowledge source.
        """
        ctx = self._build_context_block()
        return (
            "Current OCT Analysis Context "
            "(the ONLY source you are permitted to use):\n"
            f"{ctx}\n\n"
            "Acknowledge you have received the context and are ready for "
            "the surgeon's question. Reply with a single sentence only."
        )

    def has_context(self) -> bool:
        return bool(self._analysis_ctx)

    # ── Audio helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _wav_to_pcm16k(wav_bytes: bytes) -> bytes:
        """
        Convert WAV (any rate / channels) → PCM 16 kHz mono S16LE.
        Uses wave (stdlib) + numpy — compatible with Python 3.13+.
        """
        import numpy as np

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            src_rate  = wf.getframerate()
            src_width = wf.getsampwidth()   # bytes per sample
            src_chans = wf.getnchannels()
            raw       = wf.readframes(wf.getnframes())

        # Decode bytes → int16 numpy array
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(src_width, np.int16)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Mix stereo → mono
        if src_chans == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample to 16 kHz via linear interpolation
        if src_rate != SAMPLE_RATE_IN:
            n_orig   = len(audio)
            n_target = int(n_orig * SAMPLE_RATE_IN / src_rate)
            audio    = np.interp(
                np.linspace(0, n_orig - 1, n_target),
                np.arange(n_orig),
                audio,
            )

        # Convert back to S16LE bytes
        audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def pcm24_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE_OUT) -> bytes:
        """Wrap Gemini's raw PCM output in a WAV container for st.audio()."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # ── Gemini Live — audio Q&A (single turn per call) ────────────────────────
    async def ask_audio_live(self, wav_bytes: bytes) -> bytes:
        """
        Accept WAV bytes from the microphone (audio-recorder-streamlit).
        Opens a fresh Gemini Live session per question:
          1. Sends the strict OCT context block as the grounding turn.
          2. Sends the surgeon's audio (PCM 16 kHz).
          3. Collects and returns the audio response (PCM 24 kHz S16LE).

        Returns empty bytes if client unavailable.
        """
        if not GENAI_AVAILABLE or not self._client:
            return b""

        pcm_input = self._wav_to_pcm16k(wav_bytes)

        # Embed OCT context directly into system instruction — avoids a
        # separate text grounding turn whose audio ack would be silently dropped.
        grounded_system = (
            SYSTEM_PROMPT
            + "\n\nCurrent OCT Analysis Context "
            "(the ONLY source you are permitted to use):\n"
            + self._build_context_block()
        )

        config = genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=grounded_system)]
            ),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

        audio_chunks: list[bytes] = []

        async with self._client.aio.live.connect(model=MODEL_ID, config=config) as session:
            # Single turn — send surgeon's audio, collect audio response
            await session.send(
                input=genai_types.Blob(
                    data=pcm_input,
                    mime_type=f"audio/pcm;rate={SAMPLE_RATE_IN}",
                ),
                end_of_turn=True,
            )
            async for msg in session.receive():
                if msg.server_content:
                    sc = msg.server_content
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                audio_chunks.append(part.inline_data.data)
                    if sc.turn_complete:
                        break

        return b"".join(audio_chunks)

    def ask_audio_live_sync(self, wav_bytes: bytes) -> bytes:
        """
        Synchronous wrapper for ask_audio_live().
        Safe to call from the Streamlit main thread (creates its own event loop).
        """
        return asyncio.run(self.ask_audio_live(wav_bytes))

    # ── Training Mode — educational audio explanation ─────────────────────────
    async def explain_frame_audio(self) -> bytes:
        """
        Training Mode: generate an educational audio explanation of the current
        OCT frame using Gemini Live.  Sends a text trigger to the session
        (no microphone input needed) and collects the spoken audio response.

        Returns empty bytes if client unavailable or no OCT context loaded.
        """
        if not GENAI_AVAILABLE or not self._client:
            return b""
        if not self._analysis_ctx:
            return b""

        training_system = (
            "You are RetinaLive Training Mode — an educational AI assistant for "
            "ophthalmology trainees and medical students learning to interpret "
            "OCT (Optical Coherence Tomography) scans.\n\n"
            "Structure your explanation in three parts:\n"
            "1. FINDING: Name the condition and describe its OCT appearance "
            "(layer changes, fluid, deposits, structural features).\n"
            "2. PATHOLOGY: Explain the underlying disease process.\n"
            "3. CLINICAL SIGNIFICANCE: What this means for the patient and "
            "what action is typically indicated.\n\n"
            "Speak naturally as an experienced clinician educator. Keep the "
            "explanation to approximately 30–40 seconds (~90 words). Define "
            "technical terms briefly.\n\n"
            "OCT Analysis Result to explain:\n"
            + self._build_context_block()
        )

        config = genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=training_system)]
            ),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Aoede"   # distinct voice from surgical mode
                    )
                )
            ),
        )

        audio_chunks: list[bytes] = []

        async with self._client.aio.live.connect(model=MODEL_ID, config=config) as session:
            await session.send(
                input=(
                    "Please give an educational explanation of this OCT finding. "
                    "Cover what it looks like on the scan, the underlying pathology, "
                    "and its clinical significance."
                ),
                end_of_turn=True,
            )
            async for msg in session.receive():
                if msg.server_content:
                    sc = msg.server_content
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                audio_chunks.append(part.inline_data.data)
                    if sc.turn_complete:
                        break

        return b"".join(audio_chunks)

    def explain_frame_audio_sync(self) -> bytes:
        """Synchronous wrapper for explain_frame_audio()."""
        return asyncio.run(self.explain_frame_audio())

    # ── Text Q&A (fallback / no-API path) ────────────────────────────────────
    async def ask_text(self, question: str) -> str:
        """
        Text query — uses gemini-2.0-flash with the same strict grounding prompt.
        Falls back to hardcoded stub if no API key.
        """
        if not GENAI_AVAILABLE or not self._client:
            return self._stub_response(question)

        ctx_block   = self._build_context_block()
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Current OCT Analysis Context "
            "(the ONLY source you are permitted to use):\n"
            f"{ctx_block}\n\n"
            f"Surgeon: {question}"
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        return response.text.strip()

    # ── ADK stub ───────────────────────────────────────────────────────────────
    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=None)

    # ── Hardcoded stub (no API key) ────────────────────────────────────────────
    @staticmethod
    def _stub_response(question: str) -> str:
        responses = {
            "analyze":   "Fluid accumulation detected near the macular region. Confidence: high.",
            "highlight": "The abnormality is located in the central retinal layer.",
            "risk":      "Current risk level is medium. Avoid mechanical pressure near the macular region.",
            "status":    "Retina scan in progress. Latest finding: DME pattern detected.",
            "normal":    "Retina appears stable. No abnormalities detected.",
            "drusen":    "Subretinal deposits detected. Early degenerative changes present.",
        }
        q = question.lower()
        for key, reply in responses.items():
            if key in q:
                return reply
        return "Analysing the OCT feed. Please hold while I process the latest frames."
