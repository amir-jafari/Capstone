"""Run every API example in scripts/ and regenerate README.md with the code and its output.

Usage:
    python build_readme.py              # run all scripts, save outputs to samples/, rebuild README
    python build_readme.py --no-run     # rebuild README from the saved samples/ only
    python build_readme.py --only nasa  # re-run scripts whose path contains "nasa"
"""
import argparse
import datetime as dt
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
SAMPLES = ROOT / "samples"

DOMAINS = [
    ("news_media", "📰", "News & Media"),
    ("social_media", "💬", "Social Media"),
    ("weather_environment", "🌦️", "Weather, Climate & Environment"),
    ("finance_economics", "💵", "Finance & Economics"),
    ("census_demographics", "👥", "Census & Demographics"),
    ("politics_government", "🏛️", "Political Science & Government"),
    ("health", "🩺", "Health & Social Science"),
    ("education", "🎓", "Education"),
    ("science_research", "🔬", "Science & Research"),
    ("ai_ml", "🤖", "AI & Machine Learning"),
    ("engineering_energy", "⚙️", "Engineering, Energy & Tech"),
    ("transportation_geo", "🗺️", "Transportation & Geospatial"),
    ("sports", "🏅", "Sports"),
    ("culture_misc", "🎬", "Culture, Books & Entertainment"),
]


def load_meta(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(mod)
    return mod.META


def run(path):
    """Return (status, output). status: live | skipped | error."""
    try:
        p = subprocess.run([sys.executable, str(path)], cwd=SCRIPTS, capture_output=True, text=True, timeout=120,
                           env={**__import__("os").environ, "PYTHONPATH": str(SCRIPTS)})
    except subprocess.TimeoutExpired:
        return "error", "Timed out after 120 s"
    if p.returncode == 0:
        return "live", p.stdout.strip()
    if p.returncode == 2 and p.stdout.startswith("[skip]"):
        return "skipped", p.stdout.strip()
    err = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
    return "error", err


def anchor(text):
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).replace(" ", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    SAMPLES.mkdir(exist_ok=True)

    entries = []
    for folder, icon, title in DOMAINS:
        for path in sorted((SCRIPTS / folder).glob("*.py")):
            rel = path.relative_to(ROOT)
            sample_file = SAMPLES / f"{folder}__{path.stem}.txt"
            if not args.no_run and args.only in str(rel):
                status, out = run(path)
                print(f"{status:8s} {rel}")
                if status == "live":
                    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                    sample_file.write_text(f"# live {stamp}\n{out}\n")
            entries.append((folder, path, load_meta(path), sample_file))

    lines = [HEADER]
    # --- summary table
    lines += ["## Catalog at a Glance", "",
              "| # | Domain | API | Auth | Key / env var | Script |", "|--:|:--|:--|:--|:--|:--|"]
    for n, (folder, path, m, _) in enumerate(entries, 1):
        dom = next(f"{i} {t}" for f, i, t in DOMAINS if f == folder)
        keyless = m["auth"].lower().startswith("none") or "demo" in m["auth"].lower()
        badge = "🟢 no key" if keyless else "🔑 key"
        lines.append(f"| {n} | {dom} | [{m['name']}](#{anchor(m['name'])}) | {badge} | "
                     f"{('`' + m['env'] + '`') if m.get('env') else '—'} | "
                     f"[`{path.name}`](scripts/{folder}/{path.name}) |")
    lines += ["", "🟢 = runs out of the box (no key, or a public demo key)  ·  🔑 = free registration required", "",
              "---", ""]

    # --- per-domain sections
    for folder, icon, title in DOMAINS:
        group = [e for e in entries if e[0] == folder]
        if not group:
            continue
        lines += [f"## {icon} {title}", ""]
        for _, path, m, sample_file in group:
            # Show the script minus its META block and the sys.path shim line.
            code = re.sub(r"^META = dict\(.*?^\)\n", "", path.read_text(), flags=re.S | re.M)
            code = re.sub(r"^import sys, pathlib; sys.path.insert.*\n", "", code, flags=re.M).strip()
            code = code.replace('\n\nif __name__ == "__main__":\n    main()', "").strip()

            lines += [f"### {m['name']}", "",
                      f"| | |", "|:--|:--|",
                      f"| **Docs** | {m['docs']} |",
                      f"| **Auth** | {m['auth']} |",
                      f"| **Register / token** | {m['signup']} |"]
            if m.get("env"):
                lines.append(f"| **Env var** | `{m['env']}` |")
            lines += [f"| **Script** | [`scripts/{folder}/{path.name}`](scripts/{folder}/{path.name}) |", "",
                      "**What you can get:**", ""]
            lines += [f"- {p}" for p in m["provides"]]
            if m.get("ideas"):
                lines += ["", f"**Capstone ideas:** {m['ideas']}"]
            if m.get("params"):
                cell = lambda t: t.replace("|", "\\|")  # escape pipes so they don't split table cells
                lines += ["", f"**Endpoint:** `{cell(m['endpoint'])}`", "",
                          "| Parameter | Description | Example |", "|:--|:--|:--|"]
                lines += [f"| `{cell(a)}` | {cell(b)} | `{cell(c)}` |" for a, b, c in m["params"]]
            lines += ["", "```python", code, "```", ""]

            if sample_file.exists():
                txt = sample_file.read_text()
                stamp = txt.splitlines()[0].replace("# live ", "")
                body = "\n".join(txt.splitlines()[1:])
                lines += [f"<details><summary><b>Sample output</b> — live run {stamp}</summary>", "",
                          "```json", body, "```", "</details>", ""]
            elif m.get("sample"):
                lines += ["<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was "
                          "rate-limited; shape follows the official docs, values are examples)</summary>", "",
                          "```json", m["sample"], "```", "</details>", ""]
            lines += ["---", ""]

    (ROOT / "README.md").write_text("\n".join(lines))
    print(f"README.md written with {len(entries)} APIs")


HEADER = """# Public API Catalog for Data Science Capstones

> A curated, runnable list of free APIs across news, social media, weather, finance, census,
> political science, health, education, science, AI & machine learning, engineering, transportation,
> sports, and culture.
> Every entry has: how to register / get a token, what data you can get, the endpoint and its
> **parameters**, a short Python example, and a **sample output**.

## Quick Start

```bash
cd API
pip install -r requirements.txt
cp .env.example .env            # paste in any keys you register for
python scripts/weather_environment/open_meteo.py   # run any single example
python build_readme.py          # re-run everything & regenerate this README
```

- Scripts share helpers in [`scripts/_common.py`](scripts/_common.py): `get()` (User-Agent + retry on 429),
  `key()` (reads `.env`), and `show()` (truncated pretty-print).
- Sample outputs marked **live run** were produced by actually running the script (saved in [`samples/`](samples/)).
  Outputs marked **⚠️ illustrative** could not be run live (key required or rate-limited); they show the response shape from the official docs.
- **One key, many APIs:** a free [api.data.gov key](https://api.data.gov/signup/) works for Congress.gov,
  OpenFEC, Regulations.gov, College Scorecard, NASA, and NLR (formerly NREL). Set it as `DATA_GOV_API_KEY`.
- Always check each API's terms of use (non-commercial clauses, attribution, rate limits) before
  publishing a dataset built from it, and **never commit `.env`** (it is git-ignored).
"""

if __name__ == "__main__":
    main()
