"""
DataNova FastAPI application entry point.

Run from the backend folder with:
    uvicorn app.main:app --reload --port 8000

CORS is wide open because the Next.js dev server runs on a different port;
this is a local application, not a public deployment.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="DataNova API",
    description="Synthetic Data Generation & Quality Evaluation System",
    version="1.0.0",
)

# Allow the Next.js frontend (any local origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
