# Use Python 3.11 slim as the base
FROM python:3.11-slim

# Install system dependencies and the Docker CLI
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including autonomous_engine.py and main.py)
COPY . .

# Expose the port used by Railway
EXPOSE 8000

# Health check to ensure the API is responsive
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD python3 -c "import http.client; import os; port = os.getenv('PORT', '8000'); conn = http.client.HTTPConnection('localhost', int(port)); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)"

# Start the server using the dynamic PORT provided by Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
