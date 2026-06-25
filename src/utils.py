"""Shared utilities: seeding, config, logging, JSONL I/O."""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # Load .env file (GROQ_API_KEY etc.)

import yaml
import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Pin all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_CACHE: dict | None = None


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration (cached after first call)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def reset_config_cache():
    """Reset cached config (useful in tests)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------

def get_openai_client(config: dict):
    """Create OpenAI-compatible client pointing at Groq."""
    from openai import OpenAI

    return OpenAI(
        base_url=config["llm"]["base_url"],
        api_key=os.getenv(config["llm"]["api_key_env"], ""),
    )


def get_instructor_client(config: dict):
    """Create Instructor-wrapped client for structured output."""
    import instructor

    return instructor.from_openai(
        get_openai_client(config), mode=instructor.Mode.JSON
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging and return root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger("genai_system")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_jsonl(filepath: str, record: dict):
    """Append a single JSON record to a JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(filepath: str) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    if not Path(filepath).exists():
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()