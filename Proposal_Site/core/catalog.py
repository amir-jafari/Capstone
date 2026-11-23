"""Load proposals (official + external) from storage into Proposal objects."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core import config
from core.render import plain_summary

PATH_RE = re.compile(
    rf"^{re.escape(config.ARXIV)}/(Proposals|External)/(\d{{4}})/({'|'.join(config.SEMESTERS)})/(\d+)/input\.json$"
)


@dataclass
class Proposal:
    kind: str            # "official" | "external"
    year: int
    semester: str
    number: int
    folder: str          # repo-relative folder path
    data: dict
    review: dict | None = None

    @property
    def slug(self) -> str:
        pre = "ext-" if self.kind == "external" else ""
        return f"{pre}{self.year}-{self.semester}-{self.number}"

    @property
    def code(self) -> str:
        return f"{'E' if self.kind == 'external' else 'P'}-{self.year}-{self.semester}-{self.number}"

    @property
    def title(self) -> str:
        return (self.data.get("project_name") or "Untitled proposal").strip()

    @property
    def term(self) -> str:
        return f"{self.semester} {self.year}"

    @property
    def proposer(self) -> str:
        return (self.data.get("Proposed by") or "").strip()

    @property
    def status(self) -> str:
        return "official" if self.kind == "official" else (self.review or {}).get("status", "pending")

    @property
    def is_public(self) -> bool:
        return self.kind == "official" or self.status == "approved"

    @property
    def summary(self) -> str:
        return plain_summary(self.data.get("Objective", ""))

    def file(self, name: str) -> str:
        return f"{self.folder}/{name}"

    @property
    def png(self) -> str:
        return self.file(f"{self.year}_{self.semester}_{self.number}.png")

    def search_blob(self) -> str:
        d = self.data
        return " ".join(str(d.get(k, "")) for k in
                        ("project_name", "Objective", "Dataset", "Approach", "Proposed by")).lower()


def _load_json(store, path: str) -> dict | None:
    raw = store.read(path)
    if raw is None:
        return None
    try:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:          # some older proposals were saved as cp1252
            text = raw.decode("cp1252", errors="replace")
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_all(store) -> list[Proposal]:
    out = []
    for path in store.list_files(config.ARXIV):
        m = PATH_RE.match(path)
        if not m:
            continue
        data = _load_json(store, path)
        if data is None:
            continue
        kind = "official" if m.group(1) == "Proposals" else "external"
        folder = path.rsplit("/", 1)[0]
        review = _load_json(store, f"{folder}/review.json") if kind == "external" else None
        out.append(Proposal(kind, int(m.group(2)), m.group(3), int(m.group(4)), folder,
                            data, review))
    out.sort(key=lambda p: (-p.year, -config.SEMESTER_ORDER[p.semester], p.kind != "official", p.number))
    return out


def next_number(store, base: str, year: int, semester: str) -> int:
    nums = [int(m.group(1)) for p in store.list_files(f"{base}/{year}/{semester}")
            if (m := re.match(rf"^{re.escape(base)}/{year}/{semester}/(\d+)/", p))]
    return max(nums, default=0) + 1


def find(proposals: list[Proposal], slug: str) -> Proposal | None:
    return next((p for p in proposals if p.slug == slug), None)
