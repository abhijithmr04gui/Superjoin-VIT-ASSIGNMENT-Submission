"""
Central configuration. All tunables live here and are sourced from
environment variables (see .env.example). Nothing in the pipeline
should read os.environ directly outside this module.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Rate limit management for the Gemini API (see app/pipeline/llm_client.py).
    # Keep gemini_max_requests_per_minute comfortably under your plan's actual
    # RPM limit - pacing calls under the limit avoids 429s far more cheaply
    # than retrying after they happen.
    gemini_max_requests_per_minute: int = 12
    gemini_max_retries: int = 5
    gemini_retry_base_delay_seconds: float = 2.0
    # Once a per-day quota 429 is seen, further calls fail fast for this long
    # instead of continuing to hit an exhausted quota chunk-by-chunk.
    gemini_daily_quota_cooldown_seconds: int = 3600

    # How many chunks to extract facts from concurrently. Bounded so it stays
    # well under gemini_max_requests_per_minute even at full concurrency; the
    # shared rate limiter is still the actual gate.
    extraction_max_concurrency: int = 4

    database_url: str = "sqlite:///./data/factlayer.db"
    data_dir: str = "./data"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    candidate_top_k: int = 3

    @property
    def uploads_dir(self) -> Path:
        p = Path(self.data_dir) / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vectors_dir(self) -> Path:
        p = Path(self.data_dir) / "vectors"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

# Ensure base data dir exists on import.
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
