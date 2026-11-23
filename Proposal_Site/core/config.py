"""Central configuration: repo paths and secrets access."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

# Proposal_Site/core/config.py -> repo root is two levels up from Proposal_Site
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "Report_Gen" / "script.py"

ARXIV = "Report_Gen/Arxiv"
OFFICIAL = f"{ARXIV}/Proposals"   # your proposals (Report_Gen pipeline)
EXTERNAL = f"{ARXIV}/External"    # proposals submitted through the website

SEMESTERS = ["Spring", "Summer", "Fall"]
SEMESTER_ORDER = {s: i for i, s in enumerate(SEMESTERS)}


def secret(section: str, key: str, default=None):
    """Read st.secrets[section][key] without crashing when no secrets file exists."""
    try:
        return st.secrets[section][key]
    except Exception:
        return default


def _top(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def github_token() -> str | None:
    """Accept [github] token = ..., or top-level GITHUB_TOKEN / github_token."""
    tok = secret("github", "token") or _top("GITHUB_TOKEN") or _top("github_token")
    return str(tok).strip() if tok else None


def secret_sections() -> list[str]:
    """Names (never values) of the configured secrets, for diagnostics."""
    try:
        return [f"[{k}] " + ", ".join(v.keys()) if hasattr(v, "keys") else k for k, v in st.secrets.items()]
    except Exception:
        return []


def on_streamlit_cloud() -> bool:
    return REPO_ROOT.as_posix().startswith("/mount/src")


GITHUB_REPO = secret("github", "repo", "amir-jafari/Capstone")
GITHUB_BRANCH = secret("github", "branch", "main")
ADVISOR_NAME = secret("site", "advisor_name", "Amir Jafari")
ADVISOR_EMAIL = secret("site", "advisor_email", "ajafari@gwu.edu")
SITE_URL = secret("site", "url", "https://capstone-course.streamlit.app/")
SITE_TITLE = secret("site", "title", "GWU Data Science Capstone Proposals")


def github_folder_url(folder: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{folder}"
