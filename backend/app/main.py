"""
DataNova FastAPI application entry point.

Run from the backend folder with:
    uvicorn app.main:app --reload --port 8000

CORS defaults to the local Next.js dev server (localhost:3000). In
deployment, set the CORS_ORIGINS environment variable to a comma-separated
list of the real frontend origins (e.g. "https://my-app.vercel.app") so the
hosted frontend can call this API.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="DataNova API",
    description="Synthetic Data Generation & Quality Evaluation System",
    version="1.0.0",
)

# Local dev origins, always allowed.
_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Extra origins for deployment, e.g. CORS_ORIGINS="https://my-app.vercel.app"
_extra_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    """Simple pointer to the API docs."""
    return {
        "app": "DataNova API",
        "docs": "/docs",
        "health": "/api/health",
    }
