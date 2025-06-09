FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.pkl

# Command to download model and start the application
CMD python download_model.py && \
    uvicorn main:app --host 0.0.0.0 --port 8000
