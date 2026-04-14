<div align="center">

# 🩺 HealthAI — Agentic Health Recommendation System

**An AI-powered, multi-agent health platform for symptom triage, early disease prediction, personalised nutrition, mental wellness, and family doctor recommendations — built entirely on free-tier infrastructure.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Core-7F77DD?style=flat)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!--[![Deploy: Free Tier](https://img.shields.io/badge/Deploy-100%25%20Free%20Tier-1D9E75?style=flat)](docs/deployment.md)-->

[Live Demo](https://healthai-demo.vercel.app) · [Architecture Diagram](docs/healthai_clean_architecture_free_tier.svg) · [Report a Bug](../../issues)
---

> **Disclaimer:** HealthAI is an AI-assisted wellness tool for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

</div>

---

## Why HealthAI?

Most health apps are either too simple (basic symptom checkers) or too complex (full EHR systems). HealthAI sits in the gap — it brings production-grade AI engineering patterns to personal and family health management, with a real agentic architecture that engineers can learn from and non-technical users can actually use.

This project demonstrates:
- **Multi-agent orchestration** with LangGraph and a policy enforcement layer
- **RAG-powered medical knowledge** using ChromaDB and sentence-transformers
- **Early disease prediction** with trained ML models (XGBoost, scikit-learn) + SHAP explainability
- **MCP-style tool integration** for appointments, nutrition databases, and wearable data
- **Production safety patterns** — escalation triggers, audit logs, medical disclaimers, and hard agent boundaries
- **Zero-cost deployment** — fully operational on Vercel + Render + Hugging Face Spaces free tiers

---

## Feature Overview

| Module | What it does |
|---|---|
| **Symptom Triage** | NLP-based severity scoring (LOW / MODERATE / HIGH / EMERGENCY) with "when should I worry?" timeline |
| **Disease Predictor** | XGBoost models for diabetes, hypertension, and cardiac risk — with SHAP-based plain-English explanations |
| **Nutrition Agent** | Detects RDA/RDI gaps from diet input; recommends foods to reach optimal levels by condition |
| **Wellness Agent** | Home remedies by severity tier, yoga sequences, and condition-aware exercise plans with contraindication flags |
| **Mental Health Module** | PHQ-9 / GAD-7 screening, CBT-structured journaling, mood trend tracking, mandatory crisis escalation |
| **Family Profiles** | Multi-user health graph — shared risk flags, family history analysis, individual dashboards |
| **Doctor Recommender** | Specialty matching + proximity + simulated insurance-aware filtering |
| **Policy Agent** | Every request passes through a safety classifier before reaching any health agent — blocks, escalates, or allows |

---

## Architecture
![HealthAI Architecture](docs/healthai_clean_architecture_free_tier.svg)
```
User (Web / PWA / Voice)
        │
        ▼
┌──────────────────────────────────────┐
│  FastAPI Gateway                     │
│  Policy Agent Middleware             │  ← Block / Escalate / Allow
│  Auth · Rate Limit · Audit Log       │
└────────────────┬─────────────────────┘
                 │
        ▼
┌──────────────────────────────────────┐
│  LangGraph Orchestrator              │
│  ┌──────────┐  ┌──────────────────┐  │
│  │ Symptom  │  │   Nutrition      │  │
│  │ agent    │  │   agent          │  │
│  ├──────────┤  ├──────────────────┤  │
│  │ Wellness │  │   Mental health  │  │
│  │ agent    │  │   agent          │  │
│  ├──────────┤  ├──────────────────┤  │
│  │ ML pred. │  │   Doctor recomm. │  │
│  └──────────┘  └──────────────────┘  │
│  RAG Pipeline  │  LLM Router         │
│  (LlamaIndex)  │  (Gemini/Groq/Ollama)│
└────────────────┬─────────────────────┘
                 │
        ▼
┌──────────────────────────────────────┐
│  Data Layer                          │
│  SQLite (dev) / PostgreSQL (prod)    │
│  ChromaDB  ·  Model Store (joblib)   │
└──────────────────────────────────────┘
```

Full architecture diagram and decision boundary docs → [`docs/architecture.md`](docs/architecture.md)

---

## Tech Stack

| Layer | Technology | Why free |
|---|---|---|
| LLM (primary) | Google Gemini 1.5 Flash | 1M tokens/day free |
| LLM (fallback) | Groq + LLaMA 3.1 8B | Free inference tier |
| LLM (offline) | Ollama + Mistral 7B | Runs locally |
| Embeddings | `sentence-transformers` | Runs locally, no API cost |
| Vector store | ChromaDB | Local persistent store |
| Orchestration | LangGraph | Open source |
| Backend | FastAPI + Python 3.11 | Open source |
| Frontend | React 18 + Vite | Open source |
| ML models | scikit-learn · XGBoost | Open source |
| Database (dev) | SQLite | Zero setup |
| Database (prod) | PostgreSQL on Railway | 500MB free tier |
| Frontend deploy | Vercel | Free tier |
| Backend deploy | Render | Free tier (spins down) |
| ML demo | Hugging Face Spaces | Free Gradio apps |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) (optional, for offline LLM)

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/healthai.git
cd healthai

# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM — get free key at aistudio.google.com
GEMINI_API_KEY=your_key_here

# Groq fallback — free at console.groq.com
GROQ_API_KEY=your_key_here

# App
SECRET_KEY=your_random_secret
ENVIRONMENT=development
DATABASE_URL=sqlite:///./healthai.db
```

### 3. Initialise database and knowledge base

```bash
python scripts/init_db.py
python scripts/build_knowledge_base.py   # builds ChromaDB index (~2 min)
python scripts/train_ml_models.py        # trains risk prediction models (~3 min)
```

### 4. Run locally

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open `http://localhost:5173`

---

## Project Structure

```
healthai/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── gateway/
│   │   ├── policy_agent.py      # Safety classifier middleware
│   │   └── router.py            # API route definitions
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph graph definition
│   │   ├── symptom_agent.py
│   │   ├── nutrition_agent.py
│   │   ├── wellness_agent.py
│   │   ├── mental_health_agent.py
│   │   ├── ml_predictor_agent.py
│   │   └── doctor_recommender.py
│   ├── rag/
│   │   ├── pipeline.py          # LlamaIndex RAG setup
│   │   └── knowledge_base/      # Medical KB documents
│   ├── llm/
│   │   └── router.py            # Gemini → Groq → Ollama fallback
│   ├── models/
│   │   ├── schemas.py           # Pydantic input/output contracts
│   │   └── db_models.py         # SQLAlchemy ORM models
│   └── ml/
│       ├── train.py             # Model training scripts
│       └── predict.py           # Inference + SHAP explanations
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── api/
│   └── vite.config.ts
├── scripts/
│   ├── init_db.py
│   ├── build_knowledge_base.py
│   └── train_ml_models.py
├── tests/
│   ├── test_policy_agent.py
│   ├── test_agents.py
│   └── test_ml_models.py
├── docs/
│   ├── architecture.md
│   ├── agent_boundaries.md
│   └── deployment.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## Deployment 

### Frontend → Vercel
```bash
cd frontend
npm run build
# Connect GitHub repo to Vercel — auto-deploys on push
```

### Backend → Render
- Create a new Web Service on [render.com](https://render.com)
- Connect your GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add environment variables from `.env`

### ML Demo → Hugging Face Spaces
```bash
cd ml_demo  # Gradio app for the prediction models
# Push to a new HF Space — runs for free
```

Full deployment guide → [`docs/deployment.md`](docs/deployment.md)

---

## ML Models

Trained on publicly available datasets:

| Model | Dataset | Accuracy | AUC |
|---|---|---|---|
| Diabetes risk | Pima Indians Diabetes (UCI) | 78% | 0.84 |
| Hypertension risk | Synthetic + NHANES features | 81% | 0.87 |
| Cardiac risk | Heart Disease UCI | 83% | 0.89 |

All predictions include SHAP-based explanations — users see *why* the model scored them, not just the score.

---

## Safety & Ethics

- Every request passes through the **policy agent** before reaching any health agent
- Crisis keywords (suicidal ideation, self-harm, emergency symptoms) trigger **immediate escalation** with resource links — the AI never tries to handle these itself
- All ML predictions include a **confidence score** and a mandatory disclaimer
- No agent stores data beyond the active session by default
- No agent can write to another agent's data domain
- Full audit log of every policy decision

---

## Roadmap

- [ ] Wearable data sync (Google Fit / Apple Health export)
- [ ] Multilingual support (Hindi, Tamil, Spanish)
- [ ] Appointment booking via Google Calendar API
- [ ] Medication interaction checker (DrugBank open data)
- [ ] Weekly family health digest (proactive notifications)
- [ ] FHIR R4 export for health records portability

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [`LICENSE`](LICENSE)

---

<div align="center">
Built with care by an AI graduate who believes good health tools should be accessible to everyone.
</div>
