"""Create / update / review / promote external proposals. All writes go through store.commit()."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import secrets
from datetime import datetime, timezone

from core import config
from core.catalog import Proposal, next_number
from core.render import build_md_html


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


def check_code(review: dict | None, code: str) -> bool:
    return bool(review) and hmac.compare_digest(review.get("access_code_sha256", ""), _hash(code))


def to_png(upload_bytes: bytes | None) -> bytes | None:
    if not upload_bytes:
        return None
    from PIL import Image
    img = Image.open(io.BytesIO(upload_bytes))
    img.thumbnail((2400, 2400))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _proposal_files(folder: str, data: dict, png: bytes | None) -> dict[str, bytes]:
    md, html = build_md_html(data)
    v, y, s = data["Version"], data["Year"], data["Semester"]
    files = {
        f"{folder}/input.json": json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
        f"{folder}/proposal_{v}.md": md,
        f"{folder}/proposal_{v}.html": html,
    }
    if png:
        files[f"{folder}/{y}_{s}_{v}.png"] = png
    return files


def create(store, fields: dict, png: bytes | None) -> tuple[str, str, str]:
    """Save a new external submission. Returns (code, access_code, folder)."""
    y, s = int(fields["Year"]), fields["Semester"]
    n = next_number(store, config.EXTERNAL, y, s)
    folder = f"{config.EXTERNAL}/{y}/{s}/{n}"
    data = {"Version": str(n), "Year": str(y), "Semester": s, **{k: v for k, v in fields.items()
                                                                   if k not in ("Year", "Semester")}}
    access = secrets.token_urlsafe(8)
    review = {"status": "pending", "submitted_at": now(), "updated_at": now(),
              "access_code_sha256": _hash(access), "promoted_to": None,
              "history": [{"at": now(), "by": data.get("Proposed by", "submitter"),
                           "action": "submitted", "comment": "", "public": True}]}
    files = _proposal_files(folder, data, png)
    files[f"{folder}/review.json"] = json.dumps(review, indent=2).encode()
    store.commit(files, f"External proposal submitted: {y} {s} #{n} - {data['project_name'][:60]}")
    return f"E-{y}-{s}-{n}", access, folder


def resubmit(store, p: Proposal, fields: dict, png: bytes | None, note: str):
    data = {**p.data, **{k: v for k, v in fields.items() if k not in ("Year", "Semester")}}
    review = dict(p.review or {})
    review["status"] = "pending"
    review["updated_at"] = now()
    review.setdefault("history", []).append(
        {"at": now(), "by": data.get("Proposed by", "submitter"), "action": "resubmitted",
         "comment": note, "public": True})
    files = _proposal_files(p.folder, data, png)
    files[f"{p.folder}/review.json"] = json.dumps(review, indent=2).encode()
    store.commit(files, f"External proposal resubmitted: {p.code}")


def review(store, p: Proposal, status: str, comment: str, public: bool, admin: str):
    r = dict(p.review or {})
    r["status"] = status
    r["updated_at"] = now()
    r.setdefault("history", []).append(
        {"at": now(), "by": admin, "action": status, "comment": comment, "public": public})
    store.commit({f"{p.folder}/review.json": json.dumps(r, indent=2).encode()},
                 f"Review {p.code}: {status}")


def promote(store, p: Proposal, admin: str) -> str:
    """Copy an approved external proposal into the official Proposals numbering."""
    y, s = p.year, p.semester
    n = next_number(store, config.OFFICIAL, y, s)
    folder = f"{config.OFFICIAL}/{y}/{s}/{n}"
    data = {**p.data, "Version": str(n)}
    files = _proposal_files(folder, data, store.read(p.png))
    r = dict(p.review or {})
    r["promoted_to"] = folder
    r.setdefault("history", []).append(
        {"at": now(), "by": admin, "action": "promoted", "public": True,
         "comment": f"Added to the official proposals as {y} {s} #{n}."})
    files[f"{p.folder}/review.json"] = json.dumps(r, indent=2).encode()
    store.commit(files, f"Promote {p.code} to official proposals {y}/{s}/{n}")
    return folder


def delete(store, p: Proposal):
    store.commit({f: None for f in store.list_files(p.folder)}, f"Delete external proposal {p.code}")


def delete_many(store, items: list[Proposal]):
    files = {f: None for p in items for f in store.list_files(p.folder)}
    store.commit(files, f"Delete {len(items)} external proposal(s): " + ", ".join(p.code for p in items))
