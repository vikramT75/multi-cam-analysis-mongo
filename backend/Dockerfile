FROM python:3.11-slim

WORKDIR /app

# Install backend-specific packages.
# We do this directly instead of using requirements.txt to avoid installing heavy
# CV/ML packages (like torch, ultralytics, opencv) which are only needed by edge nodes.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets aio-pika numpy motor

# Copy backend files
COPY backend /app/backend

# The server expects a data directory for the SQLite database
RUN mkdir -p /app/data

# Expose the FastAPI port
EXPOSE 8000

# Start server
CMD ["python", "backend/server.py"]
