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


GITHUB_REPO = secret("github", "repo", "amir-jafari/Capstone")
GITHUB_BRANCH = secret("github", "branch", "main")
ADVISOR_NAME = secret("site", "advisor_name", "Amir Jafari")
ADVISOR_EMAIL = secret("site", "advisor_email", "ajafari@gwu.edu")
SITE_TITLE = secret("site", "title", "GWU Data Science Capstone Proposals")


def github_folder_url(folder: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{folder}"
