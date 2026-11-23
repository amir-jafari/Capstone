"""Text cleaning for display, and md/html generation via Report_Gen/script.py."""
from __future__ import annotations

import base64
import html
import importlib.util
import os
import re
import tempfile
import textwrap

from core import config

LIST_RE = re.compile(r"^(\d+[.)]|[-*•])\s+")
WEEK_RE = re.compile(r"^((?:Week|Weeks|Phase|Month)\s*[\d\-–\s]+):\s*(.*)$", re.I)


def _escape(s: str) -> str:
    # Streamlit markdown treats $...$ as LaTeX and ~...~ as strikethrough
    return s.replace("$", r"\$").replace("~", r"\~")


def _is_heading(line: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", line)
    return line.endswith(":") and len(line) < 110 and (
        (letters.isupper() and len(letters) > 3) or len(line.split()) <= 6)


def clean_field(text: str) -> str:
    """Turn the free-form, indented triple-quoted JSON text into tidy markdown."""
    if not text:
        return ""
    text = textwrap.dedent(str(text).replace("\t", "    ")).strip("\n")
    blocks: list[tuple[str, str]] = []   # (kind, text) where kind in list|line|head|blank
    for raw in text.splitlines():
        if not raw.strip():
            blocks.append(("blank", ""))
            continue
        stripped = raw.strip()
        indented = raw[:1].isspace()
        if indented and blocks and blocks[-1][0] in ("list", "line") and not LIST_RE.match(stripped):
            kind, prev = blocks[-1]
            blocks[-1] = (kind, prev + " " + stripped)          # continuation line
        elif LIST_RE.match(stripped):
            blocks.append(("list", stripped))
        elif _is_heading(stripped):
            blocks.append(("head", stripped))
        else:
            blocks.append(("line", stripped))

    out: list[str] = []
    prev = "blank"
    for kind, s in blocks:
        s = _escape(s)
        if kind == "blank":
            out.append("")
        elif kind == "head":
            out += ["", f"**{s}**", ""]
        elif kind == "list":
            if prev not in ("list", "blank"):
                out.append("")
            out.append(s)
        else:
            m = WEEK_RE.match(s)
            if m:
                s = f"**{m.group(1).strip()}:** {m.group(2).strip()}"
            if prev == "list":
                out.append("")
            out.append(s + "  ")   # hard line break keeps line structure
        prev = kind if kind != "blank" else "blank"
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def plain_summary(text: str, n: int = 210) -> str:
    t = re.sub(r"\s+", " ", textwrap.dedent(str(text or ""))).strip()
    t = re.split(r"Key Objectives:|Objectives:", t)[0].strip() or t
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


# ------------------------------------------------- Report_Gen integration ---
def _report_gen():
    spec = importlib.util.spec_from_file_location("report_gen_script", config.SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Blank(dict):
    def __missing__(self, key):
        return ""


def build_md_html(data: dict) -> tuple[bytes, bytes]:
    """Generate proposal_{N}.md / .html with the exact same template as Report_Gen/script.py."""
    rg = _report_gen()
    data = _Blank(data)   # missing optional keys render as empty instead of KeyError
    with tempfile.TemporaryDirectory() as tmp:
        out = tmp + os.sep
        rg.generate_readme(data, out)
        md_path = out + f"proposal_{data['Version']}.md"
        html_path = out + f"proposal_{data['Version']}.html"
        rg.convert_md_to_html(md_path, html_path)
        with open(md_path, "rb") as f1, open(html_path, "rb") as f2:
            return f1.read(), f2.read()


PAGE_CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:860px;margin:2rem auto;padding:0 1.2rem;
line-height:1.6;color:#1d2733;background:#fff}
h1{color:#033C5A;border-bottom:3px solid #AA9868;padding-bottom:.3rem}
h2{color:#033C5A;margin-top:2rem}h3,h4{color:#4a5866;margin:.2rem 0}
img{max-width:100%;border:1px solid #e3e7eb;border-radius:8px}
a{color:#0b5d8a}
"""


def standalone_html(html_body: str, png: bytes | None, png_name: str) -> str:
    """Wrap the generated html fragment as a full styled page with the figure embedded."""
    if png:
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        html_body = html_body.replace(f'src="{png_name}"', f'src="{uri}"')
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Capstone Proposal</title><style>{PAGE_CSS}</style></head>"
            f"<body>{html_body}</body></html>")


def esc(s: str) -> str:
    return html.escape(str(s or ""))
