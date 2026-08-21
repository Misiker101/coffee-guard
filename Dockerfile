FROM python:3.11-slim

WORKDIR /code

# System deps needed by Pillow/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 7860

# Hugging Face Spaces expects the app to listen on port 7860 by default.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
