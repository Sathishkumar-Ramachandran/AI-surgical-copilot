# RetinaLive — Deployment Details

## Live Application

| | |
|---|---|
| **URL** | https://retinalive-1002439980089.asia-south1.run.app |
| **Platform** | Google Cloud Run (asia-south1) |
| **Project** | gen-lang-client-0256042453 |

---

## Google Cloud Services & APIs Used

### 1. Gemini Live API — Real-Time Voice Interface
**Code:** [`retina_live_agent/agents/surgical_assistant_agent.py`](https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot/blob/main/retina_live_agent/agents/surgical_assistant_agent.py)

The `SurgicalAssistantAgent` opens a live bidirectional WebSocket session with the Gemini Live API (`gemini-2.5-flash-native-audio-latest`) via the `google-genai` SDK. It sends surgeon audio (PCM 16 kHz) and receives spoken responses (PCM 24 kHz) in real time — grounded strictly to the current OCT scan context.

The `Training Mode` method `explain_frame_audio()` in the same file uses the same Live API to generate educational spoken explanations of OCT findings for trainees.

```python
async with self._client.aio.live.connect(model=MODEL_ID, config=config) as session:
    await session.send(input=genai_types.Blob(data=pcm_input,
                       mime_type=f"audio/pcm;rate={SAMPLE_RATE_IN}"), end_of_turn=True)
    async for msg in session.receive():
        ...
```

### 2. Gemini API — Text Q&A Fallback
**Code:** [`retina_live_agent/agents/surgical_assistant_agent.py`](https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot/blob/main/retina_live_agent/agents/surgical_assistant_agent.py)

`ask_text()` calls `gemini-2.5-flash` via `client.models.generate_content()` for typed questions when audio is unavailable.

### 3. Google Agent Development Kit (ADK)
**Code:** [`retina_live_agent/agents/`](https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot/blob/main/retina_live_agent/agents/)

Five ADK `BaseAgent` subclasses form the reasoning pipeline:

| Agent | Role |
|---|---|
| `VisionAnalysisAgent` | Runs CNN inference on each OCT frame |
| `ClinicalInterpreterAgent` | Maps model output to clinical findings |
| `AlertAgent` | Generates severity-graded surgeon alerts |
| `SurgicalAssistantAgent` | Gemini Live voice interface |
| `CoordinatorAgent` | Orchestrates the full pipeline |

### 4. MCP (Model Context Protocol) Server
**Code:** [`retina_live_agent/mcp/`](https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot/blob/main/retina_live_agent/mcp/)

Three MCP tools (`analyze_oct_frame`, `interpret_prediction`, `generate_alert`) expose the OCT pipeline to any MCP-compatible LLM agent.

### 5. Google Cloud Run
The Streamlit dashboard is containerised and deployed on Cloud Run:

```bash
gcloud run deploy retinalive \
  --source . \
  --region asia-south1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --set-secrets "GOOGLE_API_KEY=retinalive-api-key:latest"
```

### 6. Google Cloud Secret Manager
The `GOOGLE_API_KEY` is stored as a versioned secret (`retinalive-api-key`) and injected into the Cloud Run container at runtime — never baked into the image.

```bash
gcloud secrets create retinalive-api-key --data-file=-
```

### 7. Google Artifact Registry & Cloud Build
Source-based deployment triggers Cloud Build automatically, which builds the Docker image and pushes it to Artifact Registry (`cloud-run-source-deploy`, asia-south1).

---

## Container Configuration

| Setting | Value |
|---|---|
| Base image | `python:3.11-slim` |
| OpenCV | `opencv-python-headless` (no X11) |
| Port | `8080` |
| Memory | `2 GiB` |
| CPU | `2 vCPU` |
| Request timeout | `300 s` |
| Authentication | Public (unauthenticated) |

**Dockerfile:** [`Dockerfile`](https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot/blob/main/Dockerfile)

---

## Repository

**GitHub:** https://github.com/Sathishkumar-Ramachandran/AI-surgical-copilot

```
AI-surgical-copilot/
├── Dockerfile
├── retina_live_agent/
│   ├── agents/
│   │   └── surgical_assistant_agent.py   ← Gemini Live API calls
│   ├── backend/pipeline.py               ← ADK orchestration
│   ├── frontend/dashboard.py             ← Streamlit UI
│   ├── mcp/                              ← MCP server + tools
│   ├── vision/oct_detector.py            ← CNN inference
│   └── requirements.txt
└── retinal-optical-coherence-tomography-oct-cnn.ipynb
```

---

## Environment Variables

| Variable | Source | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Cloud Secret Manager | Authenticates Gemini API and ADK calls |
