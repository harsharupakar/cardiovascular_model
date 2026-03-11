# Multi-stage Dockerfile for FastAPI + Streamlit
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Scripts to run both: normally handled via docker-compose or a shell script
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
