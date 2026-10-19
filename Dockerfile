
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies for GIS libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libexpat1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Copy source code
COPY api.py agent.py rag.py schemas.py ./
COPY tools/ ./tools/

EXPOSE 8000

CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
