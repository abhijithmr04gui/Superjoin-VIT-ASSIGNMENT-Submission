"""Run with: python -m app.main  (or: uvicorn app.api.main:app --reload)"""
import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
