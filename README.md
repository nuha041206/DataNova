# DataNova

**Synthetic Data Studio** — a full-stack web app by Nuha Khan.

DataNova is a **local-first** application: no account, no sign-up, no cloud. Everything — uploads, generated datasets, evaluations, reports and project history — stays on your computer.

DataNova helps you:

- **Expand an existing dataset** (Mode A): upload a CSV, get a dataset health check, automatic target/task detection, then generate additional synthetic rows with [SDV](https://docs.sdv.dev/sdv/) (GaussianCopula, CTGAN or TVAE).
- **Generate from requirements** (Mode B): describe the dataset you want in plain English; DataNova parses it into a structured spec (and asks about anything ambiguous) before generating.
- **Evaluate honestly**: statistical comparison (KS / TVD similarities, association matrices, target comparison), an **ML utility experiment** (train on real → test on real vs. train on synthetic → test on real), and rule-based explanations — every number is computed from your actual data.
- **Export**: synthetic CSV plus a branded 10-section PDF evaluation report generated from the exact results stored in the project database.

> **Honesty policy**: DataNova never fabricates results. If something cannot be measured (e.g. there is no real dataset to compare against in Mode B), the UI says so instead of showing a fake score. The "DataNova Score" is a transparent, weighted teaching metric — **not** an industry standard.

---

## Tech Stack

| Layer    | Technology                                                                 |
| -------- | -------------------------------------------------------------------------- |
| Frontend | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, shadcn/ui-style components, Recharts |
| Backend  | Python 3.11+ / FastAPI, Uvicorn, pandas, scikit-learn, SDV (synthetic data), ReportLab + matplotlib (reports), SQLAlchemy + SQLite |

The frontend and backend are **fully separated**: the Next.js app talks to FastAPI only over HTTP (default `http://127.0.0.1:8000`), so each can run, restart and be developed independently. There is **no Streamlit** anywhere in this project.

```
datanova/
├── backend/                  # FastAPI application (runs on :8000)
│   ├── app/
│   │   ├── main.py           # FastAPI entry point + CORS
│   │   ├── routes.py         # All /api endpoints
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── database.py       # SQLite via SQLAlchemy (projects table)
│   │   ├── config.py         # Paths & limits (50 MB upload, 200k rows…)
│   │   └── services/         # One module per concern:
│   │       ├── dataset_analyzer.py     # Health check of uploaded CSV
│   │       ├── target_detector.py      # Target/task detection (explainable)
│   │       ├── sdv_generator.py        # SDV synthesizers (isolated here)
│   │       ├── evaluation_engine.py    # KS/TVD, associations, DataNova Score
│   │       ├── ml_utility_engine.py    # Real→real vs synth→real experiment
│   │       ├── model_recommender.py    # Baseline model suggestion
│   │       ├── explainability_engine.py# Plain-language findings
│   │       ├── requirement_parser.py   # NL → structured spec (Mode B)
│   │       ├── spec_synthesizer.py     # Spec → synthetic dataset (Mode B)
│   │       └── report_generator.py     # Branded PDF report (ReportLab + matplotlib)
│   ├── scripts/
│   │   ├── make_sample_data.py  # Creates sample_data/*.csv for testing
│   │   ├── smoke_test.py        # Offline engine tests (no server needed)
│   │   ├── e2e_api_test.py      # 19-step API test (needs server on :8000)
│   │   └── verify_pdf_text.py   # Cross-checks PDF content vs stored SQLite values
│   ├── data/                    # Created at runtime: uploads, reports, SQLite DB
│   └── requirements.txt
└── frontend/                 # Next.js app (runs on :3000)
    ├── app/                  # One folder per route (dashboard, create, projects…)
    ├── components/
    │   ├── ui/               # Reusable primitives (button, card, dialog, toast…)
    │   ├── layout/           # Sidebar (logo + nav), Header (status pill)
    │   ├── workflow/         # Step indicator, evaluation view, health cards…
    │   └── charts/           # Recharts comparison charts
    ├── public/logo.png       # DataNova logo (also used by the PDF generator)
    └── lib/                  # API client, types, helpers
```

## How the App Is Organized

**Application shell** (sidebar navigation):

1. **Dashboard** (`/`) — live stats, the two generation modes, recent projects.
2. **New Project** (`/create`) — pick a mode: *Expand an Existing Dataset* or *Generate From Requirements*.
3. **Project History** (`/projects`) — every workflow saved **on this computer**, with results, CSV/PDF downloads, rename and delete.
4. **Methodology & About** (`/about`) — visual workflow, methodology, ML-utility experiments, score components, limitations and tech stack.

**Workflow** (guided steps shown at the top of each page):

```
Configure → Generate → Preview → Evaluate → Export
```

There is no Beginner/Advanced mode — every user sees the same complete interface. Technical options (e.g. synthesizer override) sit in collapsed sections so the main flow stays simple.

**Local-first storage.** No user account exists or is required. Projects created on this computer are stored locally (SQLite + files under `datanova/backend/data/`) and are available from this browser/application. Removing that data folder removes the history.

## Running the Project

### 1. Backend (FastAPI) — port 8000

```bash
cd datanova/backend

# (optional) create an isolated environment
python -m venv .venv
.venv\Scripts\activate        # Windows  (use .venv/bin/activate on macOS/Linux)

pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
```

Check it is up: open <http://127.0.0.1:8000/api/health> — you should see JSON like `{"status": "ok", …}`. Interactive API docs: <http://127.0.0.1:8000/docs>.

### 2. Frontend (Next.js) — port 3000

In a **second terminal**:

```bash
cd datanova/frontend

npm install
npm run dev
```

Open <http://localhost:3000>. The dashboard shows a banner and the header shows a status pill if the backend is not reachable.

> The frontend expects the API at `http://127.0.0.1:8000`. To point it elsewhere, set `NEXT_PUBLIC_API_BASE` **before** building/starting (e.g. `NEXT_PUBLIC_API_BASE=http://127.0.0.1:9000 npm run dev`).

### Production-style run (optional)

```bash
# terminal 1 — backend
cd datanova/backend && python -m uvicorn app.main:app --port 8000

# terminal 2 — frontend
cd datanova/frontend && npm run build && npm start
```

## Trying It Out

```bash
cd datanova/backend
python scripts/make_sample_data.py     # creates sample_data/*.csv
```

Then in the app: **New Project → Expand an Existing Dataset** and upload one of:

- `telco_churn.csv` — classification, imbalanced target (`Churn`)
- `loan_approval.csv` — includes an ID column and missing values
- `house_prices.csv` — regression target (`price`)

Or pick **Tell DataNova What You Want** and type a requirement, e.g.:
"Generate 500 rows of customer data for a loan approval classification task with a target column approved and features age, income, credit_score; about 30% approved."

## The PDF Report

Generate it from any saved project (**Generate PDF report** button, or `POST /api/projects/{id}/report`). The report is built **only** from the canonical project record stored in SQLite — the same payloads the web UI renders (analysis, detection, evaluation, ML utility, explanation). Nothing is recalculated, so UI and PDF always agree. It contains 13 sections: project information, dataset health, generation configuration, synthetic summary, statistical comparison (means/medians/std/quantiles and category proportions), distribution charts, correlation heatmaps, target comparison, ML utility test, model information, DataNova Score with weights, interpretation derived from the stored numbers, and limitations — with logo header, page numbers and DataNova branding throughout.

## Tests

```bash
# engine tests — no server required (run from datanova/backend)
python -X utf8 scripts/smoke_test.py

# full API test — start the server first (port 8000), then:
python -X utf8 scripts/e2e_api_test.py

# generate a report for a saved project, then verify its contents match the DB:
python -X utf8 scripts/verify_pdf_text.py <project_id>
```

All scripts print `PASS`/`FAIL` per check and a final summary.

## Known Limitations (by design)

- Single-user, local-first: SQLite storage, in-memory session store for in-flight generations (restart the backend between upload and generate and you must re-upload).
- CTGAN/TVAE on large datasets can take minutes; GaussianCopula is the fast default for small data.
- The DataNova Score is a weighted teaching metric (30/30/25/15 over column similarity, correlation structure, target similarity, ML-informed transparency) — it is **not** an industry standard and the UI says so wherever it appears.
- Synthetic data is not automatically anonymous or unbiased; always review the evaluation before using generated data.
