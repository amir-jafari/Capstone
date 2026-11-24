"""One standard for every proposal folder, whether it came from Report_Gen/json_gen.py or the website.

errors   -> the proposal does not follow the format (website refuses to save)
warnings -> worth fixing, but the proposal still works
"""
from __future__ import annotations

import ast
import json
import re

from core import config, fmt
from core.render import WEEK_RE

REQUIRED = ["Version", "Year", "Semester", "project_name", "Objective", "Dataset", "Rationale", "Approach",
            "Timeline", "Expected Number Students", "Proposed by", "Proposed by email", "instructor",
            "instructor_email"]
OPTIONAL = [k for k in fmt.KEY_ORDER if k not in REQUIRED]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def check_data(data: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if not isinstance(data, dict):
        return ["input.json is not a JSON object"], []
    for k in REQUIRED:
        if k not in data:
            errors.append(f'missing field "{k}"')
        elif not isinstance(data[k], str):
            errors.append(f'"{k}" must be a string (got {type(data[k]).__name__})')
        elif not data[k].strip():
            errors.append(f'"{k}" is empty')
    missing = [k for k in OPTIONAL if k not in data]
    if missing:
        warnings.append("missing optional fields: " + ", ".join(missing))
    for k in OPTIONAL:
        if k in data and not isinstance(data[k], str):
            errors.append(f'"{k}" must be a string (got {type(data[k]).__name__})')
    extra = [k for k in data if k not in fmt.KEY_ORDER]
    if extra:
        warnings.append(f"unknown fields: {', '.join(extra)}")
    g = lambda k: str(data.get(k, "")).strip()
    if g("Version") and not g("Version").isdigit():
        errors.append(f'"Version" must be a number (got "{g("Version")}")')
    if g("Year") and not re.fullmatch(r"20\d\d", g("Year")):
        errors.append(f'"Year" must be a 4-digit year (got "{g("Year")}")')
    if g("Semester") and g("Semester") not in config.SEMESTERS:
        errors.append(f'"Semester" must be one of {config.SEMESTERS} (got "{g("Semester")}")')
    for k in ("Proposed by email", "instructor_email"):
        if g(k) and not all(EMAIL_RE.match(e.strip()) for e in g(k).split(",")):   # comma-separated allowed
            errors.append(f'"{k}" is not a valid email ("{g(k)}")')
    if g("Timeline") and sum(bool(WEEK_RE.match(l.strip().replace("**", "")))
                             for l in g("Timeline").splitlines()) < 2:
        warnings.append('Timeline has no "Week N: ..." lines (it will render as plain text, not a timeline)')
    if g("Objective") and not re.search(r"^\s*1[.)]\s", g("Objective"), re.M):
        warnings.append('Objective has no numbered "Key Objectives" list')
    if list(k for k in data if k in fmt.KEY_ORDER) != [k for k in fmt.KEY_ORDER if k in data]:
        warnings.append("fields are not in the standard json_gen.py order")
    return errors, warnings


def _json_gen_data(src: str) -> dict | None:
    """Extract the data_to_save literal from a json_gen.py without running it."""
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "data_to_save" for t in node.targets):
                return ast.literal_eval(node.value)
    except (SyntaxError, ValueError):
        return None
    return None


def check_folder(store, folder: str) -> tuple[list[str], list[str]]:
    """Validate a whole proposal folder: input.json, generated files, figure, json_gen.py."""
    raw = store.read(f"{folder}/input.json")
    if raw is None:
        return ["input.json not found"], []
    try:
        try:
            data = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            data = json.loads(raw.decode("cp1252"))
            raise_enc = True
        else:
            raise_enc = False
    except json.JSONDecodeError as e:
        return [f"input.json is not valid JSON: {e}"], []
    errors, warnings = check_data(data)
    if raise_enc:
        warnings.append("input.json is not UTF-8 (saved as Windows-1252)")

    m = re.search(r"/(\d{4})/(\w+)/(\d+)$", folder)
    if m and isinstance(data, dict):
        y, s, n = m.groups()
        for k, v in (("Year", y), ("Semester", s), ("Version", n)):
            if str(data.get(k, "")).strip() != v:
                errors.append(f'"{k}" is "{data.get(k)}" but the folder says {v}')
    v = str(data.get("Version", "")).strip() if isinstance(data, dict) else ""
    files = {f.rsplit("/", 1)[1] for f in store.list_files(folder)}
    for name in (f"proposal_{v}.md", f"proposal_{v}.html"):
        if name not in files:
            errors.append(f"{name} is missing (run Report_Gen/script.py)")
    if m and f"{m.group(1)}_{m.group(2)}_{m.group(3)}.png" not in files:
        errors.append(f"figure {m.group(1)}_{m.group(2)}_{m.group(3)}.png is missing (required)")
    if "json_gen.py" not in files:
        warnings.append("json_gen.py copy is missing")
    else:
        jg = _json_gen_data(store.read(f"{folder}/json_gen.py").decode("utf-8", "replace"))
        if jg is None:
            warnings.append("json_gen.py could not be parsed")
        elif jg != data:
            diff = [k for k in set(jg) | set(data) if jg.get(k) != data.get(k)]
            warnings.append(f"json_gen.py and input.json differ in: {', '.join(sorted(diff))}")
    return errors, warnings
