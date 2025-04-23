# Dockerfile
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- ADD SYSTEM DEPENDENCIES ---
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
# --- END ADD ---

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose ports
EXPOSE 5001
EXPOSE 8000
EXPOSE 8001
EXPOSE 8501

# Default command
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]