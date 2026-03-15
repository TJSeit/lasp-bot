FROM python:3.11-slim

# Non-interactive, no .pyc files, unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy application source
COPY main.py rag.py ./

EXPOSE 8000

# Azure Container Apps routes traffic to port 8000 by default.
# One Uvicorn worker is sufficient for serverless; scale via replicas.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
