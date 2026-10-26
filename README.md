# DataNova — Synthetic Data Studio

## Overview

DataNova is a full-stack application for generating and evaluating synthetic datasets. Synthetic data consists of artificially generated records designed to reflect relevant patterns and statistical properties of source data without reproducing the original records directly.

The application supports two generation workflows: users can upload a CSV file for data-driven synthesis, or describe a desired dataset in plain English. DataNova then generates a dataset that can be previewed, evaluated, saved as a project, and exported as a CSV file. It can also produce a PDF report summarizing the evaluation.

## How DataNova Works

DataNova provides two approaches to synthetic data generation:

- **Data-driven generation:** When a source CSV is provided, the application interprets its schema and uses synthetic-data models from the Synthetic Data Vault (SDV) ecosystem. Supported models include GaussianCopula, CTGAN, and TVAE. These models aim to reproduce important statistical patterns and relationships in the source dataset.
- **Description-driven generation:** When a dataset is specified in plain English, the application converts the description into a structured specification and generates records using rule-based semantic generators.

After generation, DataNova evaluates the resulting dataset using statistical comparisons and data-quality checks. It also provides machine-learning utility experiments that compare model performance when trained on real data versus synthetic data. Results can be reviewed in the application and included in a PDF evaluation report.

## Technology

- **Frontend:** Next.js, React, TypeScript, and Tailwind CSS
- **Backend:** Python and FastAPI
- **Synthetic data generation:** Synthetic Data Vault (SDV), including GaussianCopula, CTGAN, and TVAE
- **Data processing and evaluation:** pandas, NumPy, and scikit-learn
- **Persistence:** SQLite and SQLAlchemy
- **Reports:** ReportLab and matplotlib

## Local Setup (Windows)

### Prerequisites

- Python 3.11 or later
- Node.js 18 or later
- npm
- Git

### 1. Clone the repository

```powershell
git clone https://github.com/nuha041206/Datanova-2.0.git
cd Datanova-2.0
```

### 2. Install backend dependencies

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Install frontend dependencies

Open a separate terminal:

```powershell
cd frontend
npm install
```

### 4. Start the backend

From the `backend` directory, with the virtual environment activated:

```powershell
python -X utf8 -m uvicorn app.main:app --reload --port 8000
```

The backend will be available at `http://127.0.0.1:8000`.

### 5. Start the frontend

In a separate terminal, from the `frontend` directory:

```powershell
npm run dev
```

Open `http://localhost:3000` in your browser.

## Author and License

Built by **Nuha Khan**.

This project uses open-source libraries, including FastAPI, Next.js, SDV, scikit-learn, pandas, and ReportLab, under their respective licenses.
