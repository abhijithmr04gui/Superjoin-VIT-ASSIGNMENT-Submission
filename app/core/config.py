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

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    database_url: str = "sqlite:///./data/factlayer.db"
    data_dir: str = "./data"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    candidate_top_k: int = 8

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
