FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r backend/requirements.txt

COPY . .

ENV PYTHONPATH=/app \
    COGNIVERSE_AI_PROVIDER=vertex \
    COGNIVERSE_AUTO_ADVANCE=false

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
