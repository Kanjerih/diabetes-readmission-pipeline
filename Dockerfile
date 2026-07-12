# Lightweight Python runtime base image
FROM python:3.11-slim

# Set an isolated working directory inside the container
WORKDIR /app

# Install basic compiler tools, then clean up cache to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to utilize Docker's layer caching
COPY requirements.txt .

# Install dependencies cleanly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Spin up web server to host our script endpoints
CMD ["python", "-m", "streamlit", "run", "src/frontend.py", "--server.port", "10000", "--server.address", "0.0.0.0"]