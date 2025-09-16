#!/bin/bash
# Startup script for FastAPI application
echo "Starting FastAPI application with uvicorn..."
echo "PORT: $PORT"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version)"
uvicorn app:app --host 0.0.0.0 --port $PORT --log-level info
