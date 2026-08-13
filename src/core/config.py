# src/core/config.py
# Loads settings.yaml and strategies.yaml.
# API keys come from .env (via python-dotenv), never from YAML.

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file at import time so all modules can use os.getenv()
load_dotenv()

# Project root is two levels up from this file (src/core/config.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_settings() -> dict:
    """Load config/settings.yaml and return as dict."""
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_strategies() -> dict:
    """Load config/strategies.yaml and return as dict."""
    path = PROJECT_ROOT / "config" / "strategies.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_scaling_tier(balance: float, settings: dict) -> dict:
    """Return the risk scaling tier for the current balance.

    Walks the scaling list in reverse (highest first) and returns
    the first tier whose min_balance is <= current balance.
    """
    tiers = sorted(settings["scaling"], key=lambda t: t["min_balance"], reverse=True)
    for tier in tiers:
        if balance >= tier["min_balance"]:
            return tier
    # Fallback to most conservative tier
    return settings["scaling"][0]
