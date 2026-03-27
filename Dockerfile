# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files & enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip first (avoid old pip issues)
RUN pip install --upgrade pip

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Expose FastAPI port
EXPOSE 4440

# Run FastAPI with auto-reload (dev mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4440", "--reload"]