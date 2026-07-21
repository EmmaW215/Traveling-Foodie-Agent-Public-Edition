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

**Status: M0 — scaffold.** The deployment path, provider fallback chain and API contract are in
place. Tiers land in M2–M4.

---

## Architecture

```
Browser ──> Vercel (Next.js, static) ──HTTPS/SSE──> Render (FastAPI orchestrator)
                                                     ├── Tier 0  RAG copilot        (M3)
                                                     ├── Tier 1  scripted agent     (M2)
                                                     ├── Tier 2  multi-agent + critic (M4)
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
│   │   ├── main.py           #   /health /readiness /dataset/meta /echo  + tier stubs
│   │   ├── config.py         #   env-driven settings & provider chain
│   │   └── llm_client.py     #   OpenAI-compatible client + fallback + embeddings
│   ├── tests/                #   offline tests — no API keys needed
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/                 # Next.js app (deployed to Vercel)
│   ├── app/page.tsx          #   M0 status panel + LLM round-trip test
│   └── lib/api.ts            #   backend client with cold-start retry
├── scripts/smoke_test.py     # M0 provider gate (runs in Actions, reads Secrets)
├── .github/workflows/        # ci.yml · smoke-test.yml · keepalive.yml
├── render.yaml               # Render blueprint (free plan)
└── .env.example
```

---

## M0 setup — do this once

### 1. Collect free API keys

| Key | Where | Notes |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com | primary chat provider |
| `GEMINI_API_KEY` | aistudio.google.com | chat fallback **and** embeddings |
| `UPSTASH_VECTOR_REST_URL` / `_TOKEN` | console.upstash.com → Vector | create index: **768 dims, COSINE** |
| `OPENROUTER_API_KEY` *(optional)* | openrouter.ai | last-resort fallback |

> The 768/COSINE choice pairs with `text-embedding-004`. If the smoke test reports a different
> dimension, it tells you exactly what to change — recreate the index or set
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
3. Build: `pip install -r requirements.txt`
   Start: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
   Health check path: `/health`
4. Add the same environment variables (keys above) plus:
   `ALLOWED_ORIGIN=https://traveling-foodie-agent-public-edition.vercel.app,http://localhost:3000`

*(Or use `render.yaml` via New → Blueprint, then fill the secret values in the dashboard.)*

### 4. Deploy the frontend to Vercel

1. Import the repo (already linked to project `traveling-foodie-agent-public-edition`).
2. **Root Directory: `frontend`** · framework preset Next.js.
3. Environment variable: `NEXT_PUBLIC_API_BASE=https://<your-service>.onrender.com`
4. Redeploy so the variable is baked into the build.

### 5. Run the M0 gate

`Actions → M0 Smoke Test (providers) → Run workflow`

---

## M0 exit criteria

- [ ] `GET https://<render-service>.onrender.com/health` returns `200 {"status":"ok"}`
- [ ] Vercel URL loads and its status card shows **connected**
- [ ] `/readiness` lists at least one LLM provider; embeddings + vector DB show *configured*
- [ ] Smoke-test workflow is green; working model IDs recorded in repo Variables
- [ ] CI (backend tests + frontend build) passes on `main`
- [ ] The "LLM round-trip test" button on the live site returns a reply and names the provider

All six checked → **M1** (Calgary dataset, `seed.py`, catalog/distance/budget tools).

---

## Local development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example .env          # fill in keys
uvicorn src.main:app --reload --port 8000
pytest -q                        # offline tests, no keys required

# Frontend (second terminal)
cd frontend
npm install
cp .env.example .env.local       # NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev                      # http://localhost:3000
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

---

## Roadmap

| Milestone | Scope |
|---|---|
| **M0** ✅ | Scaffold, deploy path, provider fallback chain, CI, smoke test |
| **M1** | Calgary dataset (+ planted edge cases), `seed.py`, catalog/distance/budget tools |
| **M2** | Tier 1 — Planner → Restaurant → Budget → Formatter |
| **M3** | Tier 0 — RAG copilot over Upstash Vector, venue-exists guard |
| **M4** | Tier 2 — parallel executors, Attraction + Route agents, critic loop |
| **M5** | Chat UI, trace timeline, itinerary cards, Leaflet map, public launch |

## License

MIT. Venue data derived from OpenStreetMap (© OpenStreetMap contributors, ODbL) and public
open-data sources.
