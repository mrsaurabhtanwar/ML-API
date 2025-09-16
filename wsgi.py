"""
WSGI wrapper for FastAPI application
This provides a fallback WSGI interface for deployment platforms that don't support ASGI
"""
from app import app

# For platforms that require WSGI, we can use this wrapper
# However, the main deployment should use ASGI with uvicorn workers
application = app
