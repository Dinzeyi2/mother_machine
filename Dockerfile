FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY mothermachine_sdk.py .

# Expose port
EXPOSE 8000

# Health check
# Simpler health check using curl (already in many slim images) or internal python
# Change this line in your Dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD python3 -c "import http.client; import os; port = os.getenv('PORT', '8000'); conn = http.client.HTTPConnection('localhost', int(port)); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)"

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
