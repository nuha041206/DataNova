# DataNova backend — Docker image for cloud deployment (e.g. Hugging Face Spaces).
#
# The build context is the backend/ folder. It installs Python dependencies
# from requirements.txt and starts the FastAPI app with uvicorn.
#
# Environment variables (all optional):
#   PORT                 Port to listen on (default 7860 — required by HF Spaces).
#   CORS_ORIGINS         Comma-separated frontend origins, e.g. "https://x.vercel.app".
#   DATANOVA_DATA_DIR    Where to store uploads/DB/reports (default /data if it
#                        exists, else <app>/data). Point this at a mounted volume.

# Python 3.11 is the version SDV is tested against and is available as a slim
# base image, keeping the image small.
FROM python:3.11-slim

# Never write .pyc files, keep logs unbuffered (so uvicorn logs show up
# immediately in the hosting provider's log viewer).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Matplotlib must use a non-interactive backend (the server has no display).
    MPLBACKEND=Agg

WORKDIR /app

# System libraries needed by numpy/scipy/scikit-learn/torch wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt ./
# Install CPU-only torch first so the image stays small (no CUDA libraries);
# sdv -> ctgan -> torch will then reuse the installed CPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code (app/ package + assets used by PDF reports).
COPY app ./app

# Default port for Hugging Face Spaces; the host can override with PORT.
ENV PORT=7860

# Use /data when the platform provides a mounted volume (HF Spaces does),
# otherwise fall back to the in-container /app/data.
ENV DATANOVA_DATA_DIR=/data/datanova
RUN mkdir -p /data/datanova /app/data

EXPOSE 7860

# Run uvicorn on all interfaces on $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
