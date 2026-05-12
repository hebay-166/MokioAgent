from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_llm_config() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    required_keys = ["MODEL", "BASE_URL", "API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise RuntimeError(f"Missing .env values: {missing_text}")

    return {
        "model": os.environ["MODEL"],
        "base_url": os.environ["BASE_URL"],
        "api_key": os.environ["API_KEY"],
    }
