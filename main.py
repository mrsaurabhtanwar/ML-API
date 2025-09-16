#!/usr/bin/env python3
"""
Main entry point for the FastAPI application
This script ensures uvicorn is used instead of gunicorn
"""
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
