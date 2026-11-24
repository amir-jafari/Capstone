"""Create / update / review / promote external proposals. All writes go through store.commit()."""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone

from core import config, fmt, validate
from core.catalog import Proposal, lowest_free_number, next_number
from core.render import build_md_html


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    """Same files, key order and text layout as a Report_Gen proposal folder."""
    data = fmt.canonical(data)
    errors, _ = validate.check_data(data)
    if not png:
        errors.append("a figure (PNG/JPG) is required")
    if errors:   # never write a proposal that breaks the standard format
        raise ValueError("The proposal does not follow the standard format: " + "; ".join(errors))
    md, html = build_md_html(data)
    v, y, s = data["Version"], data["Year"], data["Semester"]
    base = "Proposals" if folder.startswith(config.OFFICIAL) else "External"
    files = {
        f"{folder}/input.json": fmt.input_json(data),
        f"{folder}/json_gen.py": fmt.json_gen_py(data, base),
        f"{folder}/proposal_{v}.md": md,
        f"{folder}/proposal_{v}.html": html,
    }
    if png:
        files[f"{folder}/{y}_{s}_{v}.png"] = png
    return files


def create(store, fields: dict, png: bytes | None) -> tuple[int, str, int]:
    """Save a new external submission. Returns (year, semester, number)."""
    y, s = int(fields["Year"]), fields["Semester"]
    n = lowest_free_number(store, config.EXTERNAL, y, s)   # fills gaps left by deletions
    folder = f"{config.EXTERNAL}/{y}/{s}/{n}"
    data = {"Version": str(n), "Year": str(y), "Semester": s, **{k: v for k, v in fields.items()
                                                                   if k not in ("Year", "Semester")}}
    review = {"status": "pending", "submitted_at": now(), "updated_at": now(),
              "promoted_to": None,
              "history": [{"at": now(), "by": data.get("Proposed by", "submitter"),
                           "action": "submitted", "comment": "", "public": True}]}
    files = _proposal_files(folder, data, png)
    files[f"{folder}/review.json"] = json.dumps(review, indent=2).encode()
    store.commit(files, f"External proposal submitted: {y} {s} #{n} - {data['project_name'][:60]}")
    return y, s, n


def resubmit(store, p: Proposal, fields: dict, png: bytes | None, note: str):
    """The submitter edited the proposal: save it and send it back to review (status pending).
    If it was approved, its copy in Proposals/ is taken down until it is approved again; the
    number is remembered (`official_number`) so re-approving puts it back in the same place."""
    data = {**p.data, **{k: v for k, v in fields.items() if k not in ("Year", "Semester")}}
    review = dict(p.review or {})
    action = "edited" if review.get("status", "pending") == "pending" else "resubmitted"
    copy = _official_copy(store, p)        # must run before the title can change
    review["status"] = "pending"
    review["updated_at"] = now()
    review.setdefault("history", []).append(
        {"at": now(), "by": data.get("Proposed by", "submitter"), "action": action,
         "comment": note, "public": True})
    files = _proposal_files(p.folder, data, png)
    if copy:
        review["official_number"] = int(review["promoted_to"].rsplit("/", 1)[1])
        review["promoted_to"] = None
        files.update({f: None for f in copy})
    files[f"{p.folder}/review.json"] = json.dumps(review, indent=2).encode()
    store.commit(files, f"External proposal resubmitted: {p.code}")


def _official_copy(store, p: Proposal) -> list[str]:
    """Files of the official copy made when `p` was approved ([] if none / it no longer matches).

    Safety: only a folder Proposals/{year}/{semester}/{N} whose project name still matches the
    external proposal is ever touched, so a manually made official proposal is never deleted."""
    folder = (p.review or {}).get("promoted_to")
    if not folder or not re.fullmatch(rf"{re.escape(config.OFFICIAL)}/{p.year}/{p.semester}/\d+", folder):
        return []
    raw = store.read(f"{folder}/input.json")
    try:
        name = json.loads(raw.decode("utf-8", "replace")).get("project_name", "") if raw else None
    except json.JSONDecodeError:
        name = None
    if name is None or name.strip() != p.title:
        return []
    return store.list_files(folder)


def _publish_files(store, p: Proposal, r: dict, admin: str) -> dict[str, bytes]:
    """Copy `p` into Proposals/{year}/{semester}/{next number}/ and record it in review dict `r`."""
    y, s = p.year, p.semester
    old = r.pop("official_number", None)              # its number before it was edited, if free
    if old and not store.list_files(f"{config.OFFICIAL}/{y}/{s}/{old}"):
        n = old
    else:
        n = next_number(store, config.OFFICIAL, y, s)  # highest existing number + 1
    folder = f"{config.OFFICIAL}/{y}/{s}/{n}"
    files = _proposal_files(folder, {**p.data, "Version": str(n)}, store.read(p.png))
    r["promoted_to"] = folder
    r.setdefault("history", []).append(
        {"at": now(), "by": admin, "action": "promoted", "public": True,
         "comment": f"Added to the official proposals as {y} {s} #{n}."})
    return files


def review(store, p: Proposal, status: str, comment: str, public: bool, admin: str) -> str | None:
    """Save a decision. Approving copies the proposal into Proposals/ with the next number;
    moving an approved proposal to another status removes that copy again.
    Returns the official folder when one was created."""
    r = dict(p.review or {})
    r["status"] = status
    r["updated_at"] = now()
    r.setdefault("history", []).append(
        {"at": now(), "by": admin, "action": status, "comment": comment, "public": public})
    files, created = {}, None
    existing = _official_copy(store, p)
    if status == "approved":
        if not existing:
            files = _publish_files(store, p, r, admin)
            created = r["promoted_to"]
    elif existing:
        files = {f: None for f in existing}
        r["promoted_to"] = None
    files[f"{p.folder}/review.json"] = json.dumps(r, indent=2).encode()
    msg = f"Review {p.code}: {status}" + (f" -> {created}" if created else "")
    store.commit(files, msg)
    return created


def promote(store, p: Proposal, admin: str) -> str:
    """Copy an already-approved proposal into Proposals/ (for ones approved before auto-copy)."""
    r = dict(p.review or {})
    files = _publish_files(store, p, r, admin)
    files[f"{p.folder}/review.json"] = json.dumps(r, indent=2).encode()
    store.commit(files, f"Promote {p.code} to official proposals {r['promoted_to']}")
    return r["promoted_to"]


def delete(store, p: Proposal):
    """Delete the external folder and its official copy in Proposals/ (if any)."""
    files = {f: None for f in store.list_files(p.folder) + _official_copy(store, p)}
    store.commit(files, f"Delete external proposal {p.code}")


def delete_many(store, items: list[Proposal]):
    files = {f: None for p in items for f in store.list_files(p.folder) + _official_copy(store, p)}
    store.commit(files, f"Delete {len(items)} external proposal(s): " + ", ".join(p.code for p in items))
