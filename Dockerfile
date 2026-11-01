# rec-engine serving image (FastAPI + FAISS).
# Build artifacts first (see README): python scripts/train_all.py && python scripts/build_index.py
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY serve ./serve
COPY artifacts ./artifacts
COPY data/raw/ml-1m/movies.dat data/raw/ml-1m/movies.dat
COPY data/raw/ml-1m/id_maps.json data/raw/ml-1m/id_maps.json

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
