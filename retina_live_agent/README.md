# RetinaLive — Real-Time AI Surgical Assistant for OCT Monitoring

A hackathon project demonstrating a multimodal AI system that assists retinal surgeons
by analysing live OCT feeds, detecting abnormalities, and responding to voice queries.

## Architecture

```
retina_live_agent/
├── agents/
│   ├── vision_analysis_agent.py        # ADK agent — runs CNN inference
│   ├── clinical_interpreter_agent.py   # ADK agent — maps predictions to alerts
│   ├── alert_agent.py                  # ADK agent — generates surgeon alerts
│   ├── surgical_assistant_agent.py     # Gemini Live API voice interface
│   └── coordinator_agent.py            # ADK orchestrator
├── mcp/
│   ├── mcp_server.py                   # Unified MCP server
│   └── tools/
│       ├── oct_analysis_tool.py        # MCP: analyze_oct_frame
│       ├── clinical_interpretation_tool.py  # MCP: interpret_prediction
│       └── risk_alert_tool.py          # MCP: generate_alert
├── vision/
│   ├── oct_detector.py                 # CNN model loader + inference
│   └── clinical_interpreter.py         # Label → clinical text mapping
├── streaming/
│   └── oct_stream.py                   # OpenCV video streamer (300 ms frames)
├── backend/
│   ├── pipeline.py                     # End-to-end pipeline orchestration
│   └── api.py                          # FastAPI REST interface
├── frontend/
│   ├── overlay.py                      # OpenCV alert overlays
│   └── dashboard.py                    # Streamlit demo dashboard
├── models/
│   ├── oct_cnn_model/                  # Exported TF SavedModel (after training)
│   └── export_model.py                 # Helper to export trained model
├── data/
│   └── oct_demo_video.mp4              # PLACEHOLDER — upload your OCT video here
├── requirements.txt
├── .env.example
└── main.py
```

## Pipeline Flow

```
OCT Frame (OCTStream @ 300 ms)
    ↓
VisionAnalysisAgent   → OCTDetector.predict()  → {"label": "DME", "confidence": 0.91}
    ↓
ClinicalInterpreterAgent → ClinicalInterpreter.interpret() → risk + clinical text
    ↓
AlertAgent            → formatted surgeon alert  → "⚠️ ALERT | Fluid accumulation…"
    ↓
CoordinatorAgent      → aggregates + pushes context to SurgicalAssistantAgent
    ↓
SurgicalAssistantAgent (Gemini Live) → voice / text response to surgeon
```

## OCT Classes & Clinical Interpretations

| Label  | Clinical Meaning | Risk |
|--------|-----------------|------|
| NORMAL | Retina appears stable | Low |
| DRUSEN | Subretinal deposits detected | Low |
| DME    | Fluid accumulation detected in retinal layers | Medium |
| CNV    | Abnormal vascular structures detected beneath the retina | High |

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo>
cd retina_live_agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

### 4. Train / export CNN model

Open and run the notebook:
```
retinal-optical-coherence-tomography-oct-cnn.ipynb
```

This will save the model to `models/oct_cnn_model/`.

Dataset: [Kaggle Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

### 5. Add OCT video (optional)

Place your OCT video at:
```
data/oct_demo_video.mp4
```

If no video is present, the system streams synthetic frames automatically.

---

## Running the System

### Option A — Streamlit Dashboard (recommended for demo)

```bash
python main.py dashboard
# or directly:
streamlit run frontend/dashboard.py
```

Open browser at: http://localhost:8501

### Option B — FastAPI REST API

```bash
python main.py api
# or:
uvicorn backend.api:app --reload
```

API docs at: http://localhost:8000/docs

### Option C — Headless pipeline (30 s)

```bash
python main.py headless 30
```

### Option D — MCP Server

```bash
python main.py mcp
```

---

## Demo Workflow

1. Open Streamlit dashboard
2. Enter your Google API Key in the sidebar
3. Upload an OCT video (or skip to use synthetic frames)
4. Click **▶ Start**
5. Watch live OCT frames with AI annotations
6. Observe surgical alerts in the right panel
7. Type a question in the Voice Interface panel, e.g.:
   - *"Analyze the retina"*
   - *"What is the risk level?"*
   - *"Highlight the abnormal region"*
8. Gemini responds with a context-aware answer

---

## MCP Tools (for Gemini agent tool use)

| Tool | Input | Output |
|------|-------|--------|
| `analyze_oct_frame` | base64 image | `{label, confidence, all_scores}` |
| `interpret_prediction` | label, confidence | `{clinical_text, risk_level, recommended_action}` |
| `generate_alert` | risk_level, clinical_text, confidence | `{alert, severity}` |

Start the MCP server to expose these tools to any MCP-compatible LLM.

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Agent orchestration | Google ADK |
| Voice + multimodal LLM | Gemini 2.0 Flash Live |
| Tool integration | MCP (FastMCP) |
| Vision model | TensorFlow CNN |
| Video streaming | OpenCV |
| REST API | FastAPI + Uvicorn |
| Demo UI | Streamlit |
