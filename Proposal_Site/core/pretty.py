"""Beautiful HTML rendering of a single proposal (used by the proposal, status and admin pages)."""
from __future__ import annotations

import base64
import re
import textwrap

import mistune

from core.render import WEEK_RE, clean_field, esc

PAREN_RE = re.compile(r"^\((\d+(?:\s*[-–]\s*\d+)?\s*(?:weeks?|days?|months?))\)\s*(.+)$", re.I)
BRACKET_RE = re.compile(r"^\[(.+?)\]\s*[:\-–]?\s*(\d+(?:\s*[-–]\s*\d+)?\s*(?:weeks?|days?|months?))\.?$", re.I)
_md = mistune.create_markdown(escape=True, plugins=["table", "url"])

SECTIONS = [
    ("Objective", "Objective", "🎯"), ("Dataset", "Dataset", "🗂️"), ("Rationale", "Rationale", "💡"),
    ("Approach", "Approach", "🛠️"), ("Timeline", "Timeline", "🗓️"),
    ("Research Contributions", "Research Contributions", "🔬"),
    ("Possible Issues", "Possible Issues", "⚠️"), ("Additional Resources", "Additional Resources", "📚"),
]

STATUS = {
    "official": ("Program proposal", "#cfe3ef"), "approved": ("Approved", "#bfe8cf"),
    "pending": ("Pending review", "#f6e2a6"), "changes_requested": ("Changes requested", "#f7cfa6"),
    "rejected": ("Not accepted", "#f3b9b9"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');
.pp{font-family:Inter,system-ui,sans-serif;color:#1d2733;}
.pp *{box-sizing:border-box;}
.pp-hero{background:linear-gradient(125deg,#022a40 0%,#033C5A 45%,#0b5d8a 100%);color:#fff;border-radius:22px;
  padding:2.4rem 2.6rem 2.1rem;position:relative;overflow:hidden;margin-bottom:1.3rem;}
.pp-hero:before{content:"";position:absolute;right:-110px;top:-120px;width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(170,152,104,.55),rgba(170,152,104,0) 70%);}
.pp-hero:after{content:"";position:absolute;left:-60px;bottom:-140px;width:260px;height:260px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.10),rgba(255,255,255,0) 70%);}
.pp-eyebrow{position:relative;z-index:1;font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#d9c79a;}
.pp-title{position:relative;z-index:1;font-family:'Source Serif 4',Georgia,serif;font-size:2.15rem;line-height:1.2;
  font-weight:700;margin:.55rem 0 1rem;color:#fff;max-width:980px;}
.pp-tags{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1.3rem;}
.pp-tag{font-size:.74rem;font-weight:600;padding:.25rem .7rem;border-radius:999px;background:rgba(255,255,255,.14);
  color:#fff;border:1px solid rgba(255,255,255,.22);}
.pp-tag.st{color:#14212d;border:none;}
.pp-people{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:1.6rem;font-size:.92rem;color:#dfeaf1;}
.pp-people b{display:block;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#a9c3d3;font-weight:600;margin-bottom:.1rem;}
.pp-people span{color:#fff;font-weight:600;}
.pp-people a{color:#fff;text-decoration:none;border-bottom:1px dotted rgba(255,255,255,.5);}
.pp-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:0 0 1.4rem;}
.pp-fact{background:#fff;border:1px solid #e3e9ee;border-radius:14px;padding:.85rem 1rem;box-shadow:0 1px 2px rgba(3,60,90,.04);}
.pp-fact b{display:block;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#7a8794;font-weight:600;}
.pp-fact span{font-size:1.02rem;font-weight:700;color:#033C5A;}
.pp-toc{display:flex;flex-wrap:wrap;gap:.45rem;margin:0 0 1.4rem;}
.pp-toc a{font-size:.82rem;font-weight:600;text-decoration:none;color:#033C5A;background:#eef3f7;border:1px solid #dbe5ec;
  padding:.35rem .8rem;border-radius:999px;}
.pp-toc a:hover{background:#033C5A;color:#fff;}
.pp-sec{background:#fff;border:1px solid #e3e9ee;border-radius:18px;padding:1.6rem 1.9rem 1.3rem;margin-bottom:1.1rem;
  box-shadow:0 2px 10px rgba(3,60,90,.05);scroll-margin-top:80px;}
.pp-sec-h{display:flex;align-items:center;gap:.8rem;margin-bottom:1rem;padding-bottom:.8rem;border-bottom:1px solid #eef2f5;}
.pp-num{flex:none;width:2.1rem;height:2.1rem;border-radius:50%;background:linear-gradient(135deg,#AA9868,#c9b78a);color:#fff;
  font-weight:700;display:flex;align-items:center;justify-content:center;font-size:.95rem;}
.pp-sec-h h2{font-family:'Source Serif 4',Georgia,serif;font-size:1.45rem;margin:0;padding:0;color:#033C5A;font-weight:700;}
.pp-sec-h .ic{margin-left:auto;font-size:1.3rem;opacity:.85;}
.pp-body{font-size:1rem;line-height:1.72;color:#2b3640;}
.pp-body p{margin:0 0 .85rem;}
.pp-body h4{font-size:.8rem;letter-spacing:.09em;text-transform:uppercase;color:#8a7440;margin:1.3rem 0 .55rem;font-weight:700;}
.pp-body h4:first-child{margin-top:0;}
.pp-body a{color:#0b5d8a;word-break:break-word;}
.pp-body ul{padding-left:1.25rem;margin:.2rem 0 .9rem;}
.pp-body ul li{margin:.25rem 0;}
.pp-body ul li::marker{color:#AA9868;}
.pp-body ol{list-style:none;counter-reset:pp;padding:0;margin:.3rem 0 1rem;}
.pp-body ol>li{counter-increment:pp;position:relative;padding:.75rem 1rem .75rem 3.1rem;margin:.5rem 0;background:#f7f9fb;
  border:1px solid #e8eef2;border-radius:12px;}
.pp-body ol>li:before{content:counter(pp);position:absolute;left:.85rem;top:.75rem;width:1.6rem;height:1.6rem;border-radius:50%;
  background:#033C5A;color:#fff;font-size:.8rem;font-weight:700;display:flex;align-items:center;justify-content:center;}
.pp-body code{background:#eef3f7;padding:.05rem .35rem;border-radius:5px;font-size:.88em;}
.pp-lead{font-size:1.08rem;color:#1d2733;}
.pp-fig{margin:1.2rem 0 .4rem;background:#f7f9fb;border:1px solid #e3e9ee;border-radius:14px;padding:1rem;text-align:center;}
.pp-fig img{max-width:100%;max-height:560px;object-fit:contain;border-radius:8px;}
.pp-fig figcaption{font-size:.82rem;color:#6b7785;margin-top:.55rem;}
.pp-tl{position:relative;margin:.2rem 0 .6rem;padding-left:1.6rem;}
.pp-tl:before{content:"";position:absolute;left:.45rem;top:.4rem;bottom:.4rem;width:2px;
  background:linear-gradient(#AA9868,#033C5A);border-radius:2px;}
.pp-tl-i{position:relative;margin:0 0 .75rem;}
.pp-tl-i:before{content:"";position:absolute;left:-1.42rem;top:.42rem;width:.85rem;height:.85rem;border-radius:50%;
  background:#fff;border:3px solid #033C5A;}
.pp-tl-l{display:inline-block;font-size:.74rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#033C5A;
  background:#e6eff5;padding:.15rem .6rem;border-radius:999px;margin-bottom:.25rem;}
.pp-tl-t{font-size:.97rem;line-height:1.6;color:#2b3640;}
@media (max-width:700px){.pp-hero{padding:1.6rem 1.3rem}.pp-title{font-size:1.55rem}.pp-sec{padding:1.2rem 1.1rem}}
</style>
"""


def _html(md_text: str) -> str:
    out = _md(md_text)
    return out.replace("<a href=", '<a target="_blank" rel="noopener" href=')


def _body(text: str) -> str:
    return _html(clean_field(text, heading="h4"))


def _timeline(text: str) -> str | None:
    lines = textwrap.dedent(str(text or "")).strip("\n").splitlines()
    items, before, after = [], [], []
    for raw in lines:
        s = re.sub(r"^[-*•]\s+", "", raw.strip()).replace("**", "").rstrip()
        if not s:
            continue
        m = WEEK_RE.match(s) or PAREN_RE.match(s)
        mb = BRACKET_RE.match(s)
        if m:
            items.append([m.group(1).strip(), m.group(2).strip().rstrip(".")])
        elif mb:
            items.append([mb.group(2).strip(), mb.group(1).replace("`", "").strip()])
        elif items and raw[:1].isspace() and not after:
            items[-1][1] += " " + s
        elif items:
            after.append(s)
        else:
            before.append(s)
    if len(items) < 2:
        return None
    html = _body("\n".join(before)) if before else ""
    html += '<div class="pp-tl">' + "".join(
        f'<div class="pp-tl-i"><span class="pp-tl-l">{esc(lbl)}</span><div class="pp-tl-t">{esc(txt)}</div></div>'
        for lbl, txt in items) + "</div>"
    if after:
        html += _body("\n".join(after))
    return html


def _objective(text: str) -> str:
    html = _body(text)
    # first paragraph reads as the lead
    return html.replace("<p>", '<p class="pp-lead">', 1)


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def team_size(raw) -> str:
    """'RECOMMENDED: 2-3 students ...' -> '2–3 students'; 'One or maximum two ...' -> '1–2 students'."""
    txt = " ".join(str(raw or "").split()).replace("*", "")
    if re.fullmatch(r"\d{1,2}(\s*[-–]\s*\d{1,2})?", txt):
        txt += " students"
    num = r"(\d{1,2}|one|two|three|four|five|six|seven|eight)"
    m = re.search(num + r"(?:\s*(?:-|–|to|or(?: maximum)?)\s*" + num + r")?(?:\s+\w+){0,2}?\s+students?\b",
                  txt[:400], re.I)
    if not m:
        return txt[:40]
    vals = [WORDS.get(g.lower()) or int(g) for g in m.groups() if g]
    vals = [v for v in vals if 1 <= v <= 10] or vals
    lo, hi = min(vals), max(vals)
    n = str(lo) if lo == hi else f"{lo}–{hi}"
    return f"{n} student{'s' if n != '1' else ''}"


def header_html(p) -> str:
    d = p.data
    st_label, st_bg = STATUS.get(p.status, (p.status, "#ddd"))
    tags = f'<span class="pp-tag st" style="background:{st_bg}">{esc(st_label)}</span>'
    if p.kind == "external":
        tags += '<span class="pp-tag">Community</span>'
    tags += "".join(f'<span class="pp-tag">{esc(t)}</span>' for t in p.tags)
    people = []
    if d.get("Proposed by"):
        email = (d.get("Proposed by email") or "").strip()
        who = esc(d["Proposed by"]) + (f' · <a href="mailto:{esc(email)}">{esc(email)}</a>' if email else "")
        people.append(f"<div><b>Proposed by</b><span>{who}</span></div>")
    if d.get("instructor"):
        people.append(f"<div><b>Advisor</b><span>{esc(d['instructor'])}</span></div>")
    if (d.get("collaborator") or "").strip():
        people.append(f"<div><b>Partner</b><span>{esc(d['collaborator'])}</span></div>")
    return (f'<div class="pp"><div class="pp-hero"><div class="pp-eyebrow">{esc(p.term)} · Proposal #{p.number}'
            f' · GWU Data Science Capstone</div><div class="pp-title">{esc(p.title)}</div>'
            f'<div class="pp-tags">{tags}</div><div class="pp-people">{"".join(people)}</div></div></div>')


def body_html(p, png: bytes | None) -> str:
    d = p.data
    facts = [("Semester", p.term), ("Team size", team_size(d.get("Expected Number Students", ""))),
             ("Proposal", f"#{p.number}"), ("Funding", (d.get("funding_opportunity") or "").strip()),
             ("GitHub", re.sub(r"^https?://(www\.)?github\.com/", "", (d.get("github_repo") or "").strip()))]
    facts_html = "".join(f'<div class="pp-fact"><b>{esc(a)}</b><span>{esc(b[:60])}</span></div>'
                         for a, b in facts if b and b.strip().lower() not in ("none", "n/a", "na", "-", "tbd"))

    present = [(lbl, k, ic) for lbl, k, ic in SECTIONS if str(d.get(k, "")).strip()]
    toc = "".join(f'<a href="#pp-{p.slug}-{i}">{ic} {esc(lbl)}</a>' for i, (lbl, _, ic) in enumerate(present, 1))

    secs = []
    for i, (lbl, k, ic) in enumerate(present, 1):
        text = d.get(k, "")
        if k == "Objective":
            inner = _objective(text)
            if png:
                uri = "data:image/png;base64," + base64.b64encode(png).decode()
                inner += (f'<figure class="pp-fig"><img src="{uri}" alt="Project overview figure">'
                          f'<figcaption>Figure 1 · Project overview</figcaption></figure>')
        elif k == "Timeline":
            inner = _timeline(text) or _body(text)
        else:
            inner = _body(text)
        secs.append(f'<section class="pp-sec" id="pp-{p.slug}-{i}"><div class="pp-sec-h"><div class="pp-num">{i}</div>'
                    f'<h2>{esc(lbl)}</h2><span class="ic">{ic}</span></div><div class="pp-body">{inner}</div></section>')

    return (f'<div class="pp"><div class="pp-facts">{facts_html}</div><div class="pp-toc">{toc}</div>'
            f'{"".join(secs)}</div>')
