# DataNova — Synthetic Data Studio

## Overview

DataNova is a full-stack application for generating, evaluating, and experimenting with synthetic tabular data. Synthetic data is artificially generated data designed to reflect meaningful patterns in an original dataset while replacing its individual records with generated ones.

**DataNova’s key differentiator is CSV-based generation:** users can upload their own dataset, inspect its data quality, generate additional synthetic records, and evaluate how closely the generated data resembles the source. Users can also generate a dataset from a plain-English description.

The application combines statistical synthesis, rule-based semantic generation, data-quality analysis, similarity evaluation, machine-learning utility testing, and configurable messy-data generation in one workflow.

## How DataNova Works

### 1. Upload a CSV or describe a dataset

DataNova supports two workflows:

- **CSV upload — the primary use case:** Upload an existing CSV file. DataNova analyzes its columns and data types, checks dataset health, and uses the source data to generate additional synthetic records.
- **Text-based generation:** Describe the required dataset in plain English. DataNova converts the description into a structured specification and generates records using rule-based semantic generators.

### 2. Generate synthetic records

For uploaded datasets, DataNova uses models from the Synthetic Data Vault (SDV) ecosystem:

- **GaussianCopula** — a statistical approach for modeling distributions and dependencies between columns.
- **CTGAN** — a conditional generative adversarial network designed for tabular data.
- **TVAE** — a tabular variational autoencoder.

The application selects or recommends a suitable approach based on dataset characteristics. Generated records are validated, and semantic post-processing helps preserve realistic formats for values such as names, emails, phone numbers, and dates.

For text-based generation, DataNova uses rule-based parsing and semantic templates rather than a large language model.

### 3. Assess similarity to the source data

DataNova evaluates how closely the synthetic dataset reflects the uploaded data. Its statistical evaluation includes **Kolmogorov–Smirnov (KS) and Total Variation Distance (TVD)-based similarity measures**, alongside comparisons of column associations and target distributions where applicable.

The resulting similarity scores and visual comparisons help users identify where synthetic data preserves the source’s statistical patterns and where differences remain. Similarity is an evaluation of statistical resemblance; it does not, by itself, guarantee privacy, correctness, or suitability for every use case.

### 4. Measure machine-learning utility

Statistical resemblance does not necessarily mean that synthetic data will perform well in a machine-learning workflow. DataNova therefore provides **ML utility experiments** that compare model performance under two scenarios:

- Train a model on real data and test it on real data.
- Train a model on synthetic data and test it on real data.

The comparison helps users understand whether patterns learned from the synthetic dataset remain useful for a downstream classification or regression task. DataNova can recommend a model based on the task and dataset characteristics and provides explanations of the evaluation results.

### 5. Generate and evaluate messy data

DataNova can deliberately introduce configurable data-quality problems into a dataset. Supported examples include missing values, duplicate records, extra whitespace, and outliers, with corruption levels controlled by the user.

This capability is useful for testing data-cleaning pipelines, validating preprocessing workflows, and examining how data-quality issues affect analysis or machine-learning performance.

### 6. Review and export results

Users can preview generated data, save and revisit project workflows, download synthetic or messy datasets as CSV files, and export a PDF evaluation report. Reports summarize dataset health, generation details, statistical comparisons, and ML utility results.

## What Type of System Is DataNova?

DataNova is a **hybrid synthetic-data application**, rather than a single standalone AI model. It combines:

- Statistical synthesis using GaussianCopula, CTGAN, and TVAE for uploaded CSV datasets.
- Rule-based parsing and semantic value generation for plain-English dataset specifications.
- Statistical evaluation and machine-learning experiments to assess the generated data.

CTGAN and TVAE are trained on the uploaded data during a generation request. DataNova does not rely on a pre-trained large language model or an external AI API for its generation workflows.

## Technology Stack

- **Frontend:** Next.js, React, TypeScript, and Tailwind CSS
- **Backend:** Python and FastAPI
- **Synthetic data generation:** Synthetic Data Vault (SDV), GaussianCopula, CTGAN, and TVAE
- **Data processing and evaluation:** pandas, NumPy, and scikit-learn
- **Persistence:** SQLite and SQLAlchemy
- **Reports and visualizations:** ReportLab and matplotlib

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

## Author

Built by **Nuha Khan**.
