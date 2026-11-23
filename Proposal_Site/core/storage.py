"""Storage backends.

* LocalStorage  - reads/writes the repo checkout on disk (local development).
* GitHubStorage - reads/writes the GitHub repo through the REST API, so files
  submitted on the deployed site are committed straight into the repo.

All paths are POSIX paths relative to the repo root, e.g.
"Report_Gen/Arxiv/External/2027/Spring/1/input.json".
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests
import streamlit as st

from core import config

API = "https://api.github.com"


class LocalStorage:
    mode = "local"

    def __init__(self, root: Path):
        self.root = root

    def list_files(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        out = []
        for dirpath, _, files in os.walk(base):
            for f in files:
                out.append(Path(dirpath, f).relative_to(self.root).as_posix())
        return out

    def read(self, path: str) -> bytes | None:
        p = self.root / path
        return p.read_bytes() if p.is_file() else None

    def commit(self, files: dict[str, bytes | None], message: str) -> None:
        for path, content in files.items():
            p = self.root / path
            if content is None:
                if p.exists():
                    p.unlink()
                # prune empty parent folders inside Arxiv
                parent = p.parent
                while parent != self.root and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)


# ---------------------------------------------------------------- GitHub ---
def _headers(token: str, raw: bool = False) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _check(r: requests.Response) -> requests.Response:
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API {r.status_code}: {r.text[:300]}")
    return r


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_tree(_token: str, repo: str, branch: str) -> dict[str, str]:
    """path -> blob sha for every file on the branch."""
    r = _check(requests.get(f"{API}/repos/{repo}/git/trees/{branch}",
                            params={"recursive": "1"}, headers=_headers(_token), timeout=30))
    return {e["path"]: e["sha"] for e in r.json()["tree"] if e["type"] == "blob"}


@st.cache_data(max_entries=2000, show_spinner=False)
def _fetch_blob(_token: str, repo: str, sha: str) -> bytes:
    # blobs are immutable (content-addressed), so they can be cached forever
    r = _check(requests.get(f"{API}/repos/{repo}/git/blobs/{sha}",
                            headers=_headers(_token, raw=True), timeout=30))
    return r.content


class GitHubStorage:
    mode = "github"

    def __init__(self, token: str, repo: str, branch: str):
        self.token, self.repo, self.branch = token, repo, branch

    def _tree(self) -> dict[str, str]:
        return _fetch_tree(self.token, self.repo, self.branch)

    def list_files(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        return [p for p in self._tree() if p.startswith(prefix)]

    def read(self, path: str) -> bytes | None:
        sha = self._tree().get(path)
        return _fetch_blob(self.token, self.repo, sha) if sha else None

    def commit(self, files: dict[str, bytes | None], message: str) -> None:
        """Create ONE commit containing all file additions/updates/deletions."""
        h, base = _headers(self.token), f"{API}/repos/{self.repo}/git"
        for attempt in range(3):
            head = _check(requests.get(f"{base}/ref/heads/{self.branch}", headers=h, timeout=30)).json()["object"]["sha"]
            base_tree = _check(requests.get(f"{base}/commits/{head}", headers=h, timeout=30)).json()["tree"]["sha"]
            entries = []
            for path, content in files.items():
                if content is None:
                    entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
                    continue
                blob = _check(requests.post(f"{base}/blobs", headers=h, timeout=60, json={
                    "content": base64.b64encode(content).decode(), "encoding": "base64"})).json()["sha"]
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob})
            tree = _check(requests.post(f"{base}/trees", headers=h, timeout=60,
                                        json={"base_tree": base_tree, "tree": entries})).json()["sha"]
            new = _check(requests.post(f"{base}/commits", headers=h, timeout=30,
                                       json={"message": message, "tree": tree, "parents": [head]})).json()["sha"]
            r = requests.patch(f"{base}/refs/heads/{self.branch}", headers=h, timeout=30, json={"sha": new})
            if r.status_code == 422 and attempt < 2:   # branch moved underneath us; retry
                continue
            _check(r)
            break
        _fetch_tree.clear()


def get_storage():
    token = config.github_token()
    if token:
        return GitHubStorage(token, config.GITHUB_REPO, config.GITHUB_BRANCH)
    if config.on_streamlit_cloud():
        return ReadOnlyStorage(config.REPO_ROOT)
    return LocalStorage(config.REPO_ROOT)


class ReadOnlyStorage(LocalStorage):
    """Used on Streamlit Cloud when no GitHub token is configured: never pretend to save."""
    mode = "read-only (no GitHub token found in secrets)"

    def commit(self, files, message):
        raise RuntimeError(
            "The site has no GitHub token configured, so nothing can be saved. "
            "Add a [github] section with token = \"...\" in the app's Secrets and reboot the app.")
