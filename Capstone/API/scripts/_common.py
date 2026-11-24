"""Shared helpers for the API catalog scripts.

Every script in this folder imports `get`, `key`, and `show` from here so the
individual examples stay short and focused on the API itself.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

# Some APIs (SEC, NWS, Nominatim, Reddit) reject requests without a descriptive User-Agent.
USER_AGENT = os.getenv("API_USER_AGENT", "GWU-DATS-Capstone-API-Catalog/1.0 (contact: your_email@gwu.edu)")


def _load_dotenv():
    """Minimal .env loader (KEY=VALUE lines) so python-dotenv is optional."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def key(env_var, default=None):
    """Return an API key from the environment / .env, or exit with a helpful message."""
    value = os.getenv(env_var, default)
    if not value:
        print(f"[skip] Set {env_var} in API/.env to run this example (see README for sign-up steps).")
        sys.exit(2)
    return value


def get(url, params=None, headers=None, retries=3, as_json=True, timeout=30):
    """GET with a User-Agent, simple retry on 429/5xx, and JSON decoding."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(6 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json() if as_json else r.text
    return None


def show(obj, max_items=5, max_str=160):
    """Pretty-print a JSON-like object, truncating long lists and strings."""

    def trim(o, depth=0):
        if isinstance(o, dict):
            return {k: trim(v, depth + 1) for k, v in o.items()}
        if isinstance(o, list):
            out = [trim(v, depth + 1) for v in o[:max_items]]
            if len(o) > max_items:
                out.append(f"... ({len(o) - max_items} more)")
            return out
        if isinstance(o, str) and len(o) > max_str:
            return o[:max_str] + "..."
        return o

    print(json.dumps(trim(obj), indent=2, ensure_ascii=False, default=str))
