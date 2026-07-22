# Traveling Foodie Agent — Public Edition

An **agentic AI travel concierge**: give it a city, days, budget, cuisines, allergies and party
size, and it returns a complete, reasoned **2-day food-focused itinerary** — restaurants +
attractions + routing + a running budget, with every hard constraint (allergy, budget, opening
hours) enforced in code, not just in a prompt.

Runs entirely on **free tiers — $0/month, no credit card**: Vercel · Render · GitHub Actions ·
Upstash Vector · Groq / Gemini / OpenRouter.

> Public, open-source implementation of a three-tier agentic architecture (Tier 0 RAG copilot →
> Tier 1 scripted agent → Tier 2 multi-agent with a bounded critic loop). All venue data is
> public and pre-staged; the agent answers only from its knowledge base.

**Status: M5 — complete.** All three tiers are live behind a Next.js chat UI with a live
agent-trace timeline, itinerary cards, a running budget bar, and a Leaflet/OpenStreetMap map. The
whole thing runs with no API keys (deterministic mock model + local lexical retriever); add the free
keys to light up the live LLM chain and Upstash retrieval.

### Using it

- **Ask the guide** (Tier 0): a RAG copilot that answers questions about the Calgary dataset and
  cites what it used, refusing anything outside it.
- **Plan a trip** (Tier 2, default): preferences → the parallel multi-agent pipeline with the
  bounded Critic loop, streamed live into the trace timeline, then a day-by-day itinerary + map.
- **Plan (simple)** (Tier 1): the same result from the sequential pipeline.

---

## Architecture

```
Browser ──> Vercel (Next.js chat UI, static) ──HTTPS/SSE──> Render (FastAPI orchestrator)
   trace timeline · itinerary cards                          ├── Tier 0  RAG copilot          /chat
   budget bar · Leaflet/OSM map                              ├── Tier 1  scripted agent       /itinerary
                                                             ├── Tier 2  multi-agent + critic /itinerary (SSE)
                                                             ├── Tools   SQLite catalog · distance matrix · budget
                                                             └── LLM     Groq → Gemini → OpenRouter (free)
                                          Data: pre-staged CSV → SQLite + Upstash Vector
```

| Layer | Service | Free-tier limit that matters |
|---|---|---|
| Frontend | Vercel Hobby | 100 GB bandwidth; we ship static only → near-zero function usage |
| Backend | Render free web service | 750 h/mo (one 24/7 service); sleeps after 15 min idle, ~60 s cold start |
| Vector DB | Upstash Vector | 10K vectors; no idle suspension |
| LLM | Groq → Gemini → OpenRouter | ~1K / ~1.5K / ~50 requests per day |
| CI/CD | GitHub Actions | free on public repos |

---

## Repository layout

```
.
├── backend/                  # FastAPI orchestrator (deployed to Render)
│   ├── src/
│   │   ├── main.py           #   /health /readiness /dataset/meta /chat /itinerary
│   │   ├── orchestrator.py   #   Tier 1 (sequential) + Tier 2 (parallel + critic loop)
│   │   ├── agents/           #   planner · executors · critic · formatter · mock
│   │   ├── rag/              #   Tier 0 copilot: retriever (Upstash/local) + grounding guard
│   │   ├── tools/            #   catalog (SQLite) · distance matrix · budget
│   │   ├── guards.py         #   SLOT_IDS · allergen · budget · venue-exists
│   │   ├── config.py         #   env-driven settings & provider chain
│   │   └── llm_client.py     #   OpenAI-compatible client + fallback + embeddings
│   ├── data/raw/*.csv        #   pre-staged Calgary dataset (fictional venues, real geography)
│   ├── scripts/              #   seed.py · embed_push.py · demo.py · ask.py
│   └── tests/                #   offline tests — no API keys needed
├── frontend/                 # Next.js chat UI (deployed to Vercel, static)
│   ├── app/page.tsx          #   tier switcher, cold-start + disclaimer, orchestration
│   ├── components/           #   PreferenceForm · TraceTimeline · ItineraryView · MapView · CopilotChat
│   └── lib/api.ts            #   backend client: SSE stream reader + cold-start retry
├── scripts/smoke_test.py     # provider gate (runs in Actions, reads Secrets)
├── .github/workflows/        # ci.yml · smoke-test.yml · keepalive.yml · embed-push.yml
├── render.yaml               # Render blueprint (free plan)
└── .env.example
```

---

## Setup

### 1. Collect free API keys (all optional — the app runs without them)

| Key | Where | Notes |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com | primary chat provider (starts `gsk_`) |
| `GEMINI_API_KEY` | aistudio.google.com | chat fallback **and** embeddings |
| `UPSTASH_VECTOR_REST_URL` / `_TOKEN` | console.upstash.com → Vector | create index: **1536 dims, COSINE** |
| `OPENROUTER_API_KEY` *(optional)* | openrouter.ai | last-resort fallback |

> 1536/COSINE pairs with `gemini-embedding-001` (which supports 768 / 1536 / 3072). If the smoke
> test reports a different dimension, it tells you exactly what to change — recreate the index or set
> `EMBEDDING_DIMENSIONS` to match.

### 2. Add GitHub repository Secrets

`Settings → Secrets and variables → Actions → New repository secret`:

```
GROQ_API_KEY
GEMINI_API_KEY
UPSTASH_VECTOR_REST_URL
UPSTASH_VECTOR_REST_TOKEN
OPENROUTER_API_KEY        (optional)
```

Optionally add repository **Variables** to override model IDs without editing code:
`GROQ_MODEL`, `GEMINI_MODEL`, `OPENROUTER_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`,
and `RENDER_HEALTH_URL` (used by the keep-alive workflow).

### 3. Deploy the backend to Render

1. New → **Web Service** → connect the repo.
2. Root Directory `backend` · Runtime **Python 3** · Plan **Free** ← *confirm this; the create
   page sometimes preselects Starter.*
3. Build: `pip install -r requirements.txt && python -m scripts.seed`
   Start: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
   Health check path: `/health`
4. Add the environment variables (keys above) plus:
   `ALLOWED_ORIGIN=https://traveling-foodie-agent-public-edition.vercel.app,http://localhost:3000`

*(Or use `render.yaml` via New → Blueprint, then fill the secret values in the dashboard.)*

### 4. Deploy the frontend to Vercel

1. Import the repo.
2. **Root Directory: `frontend`** · framework preset Next.js.
3. Environment variable: `NEXT_PUBLIC_API_BASE=https://<your-service>.onrender.com`
4. Redeploy so the variable is baked into the build.

---

## Local development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scripts.seed             # build the dataset
uvicorn src.main:app --reload --port 8000
pytest -q                          # offline tests, no keys required
python -m scripts.demo --tier 2    # Tier 2 itinerary in the terminal (mock, no keys)
python -m scripts.ask              # Tier 0 copilot, the three standard questions

# Frontend (second terminal)
cd frontend
npm install
cp .env.example .env.local         # NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev                        # http://localhost:3000
```

---

## Design rules (carried through every milestone)

1. **Knowledge-base only.** Venues come from the pre-staged dataset; a code-level guard rejects
   any venue an LLM invents.
2. **Arithmetic in Python, never in the LLM.** The budget tracker is deterministic.
3. **Bounded critic loop.** Max 2 revisions; `issue.slot` is validated against a closed
   `SLOT_IDS` enum — an off-vocabulary slot re-asks the critic instead of re-planning everything.
4. **No vendor SDKs.** `httpx` only, so the backend stays small enough for a 512 MB instance and
   providers stay swappable.
5. **Secrets never in code.** `.env` is gitignored; production values live in Render/Vercel/GitHub.
6. **Offline path is real, not a stub.** A deterministic mock model + a local lexical retriever run
   the entire app with no keys, so CI is deterministic and a cold demo always works.

---

## Roadmap

| Milestone | Scope |
|---|---|
| **M0** ✅ | Scaffold, deploy path, provider fallback chain, CI, smoke test |
| **M1** ✅ | Calgary dataset (+ planted edge cases), `seed.py`, catalog/distance/budget tools |
| **M2** ✅ | Tier 1 — Planner → Restaurant → Budget → Formatter |
| **M3** ✅ | Tier 0 — RAG copilot over Upstash Vector, venue-exists guard |
| **M4** ✅ | Tier 2 — parallel executors, Attraction + Route agents, critic loop |
| **M5** ✅ | Chat UI, trace timeline, itinerary cards, Leaflet map, public launch |

---

## Public launch checklist

The app works deployed with **no keys** (mock model + local retriever). To go fully live:

1. **Frontend (Vercel):** root directory `frontend`, set `NEXT_PUBLIC_API_BASE` to your Render URL,
   redeploy. The build is fully static — no serverless functions, so it stays deep inside Hobby.
2. **Backend (Render):** confirm the free plan; set `ALLOWED_ORIGIN` to your Vercel domain (CORS).
3. **Keep it warm:** set the repo Variable `RENDER_HEALTH_URL` so the keep-alive workflow pings
   `/health` and visitors rarely hit a ~60 s cold start. The UI shows a "waking backend" state
   regardless.
4. **Live LLMs + Upstash (optional):** add `GROQ_API_KEY` / `GEMINI_API_KEY` and the Upstash
   secrets on Render, run **Actions → Embed corpus to Upstash** once, and `/readiness` flips to
   `rag_retriever: upstash`. Until then it serves grounded answers from the local lexical retriever.

Cost audit at every step: **$0/month**, no credit card in the runtime path.

## License

MIT. Venue data is fictional (see `backend/data/README.md`); the schema is OpenStreetMap-compatible
for a future real-data import.
