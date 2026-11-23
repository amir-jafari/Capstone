"""Shared UI pieces: CSS, chips, proposal cards and the full proposal view."""
from __future__ import annotations

import json

import streamlit as st

from core import config, pretty
from core.catalog import Proposal
from core.render import esc, standalone_html

NAVY, GOLD = "#033C5A", "#AA9868"

STATUS_STYLE = {
    "official": ("Official", "#033C5A", "rgba(3,60,90,.10)"),
    "approved": ("Approved", "#1e7b45", "rgba(30,123,69,.12)"),
    "pending": ("Pending review", "#8a6100", "rgba(214,158,0,.15)"),
    "changes_requested": ("Changes requested", "#9a4a00", "rgba(230,120,20,.15)"),
    "rejected": ("Not accepted", "#a12a2a", "rgba(200,40,40,.12)"),
}

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #0b5d8a 65%, #3d7fa6 100%);
  color: #fff; padding: 2.4rem 2.4rem 2rem; border-radius: 20px; margin-bottom: 1.4rem;
  position: relative; overflow: hidden;}}
.hero:after {{content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px;
  border-radius:50%; background: rgba(170,152,104,.35);}}
.hero h1 {{color:#fff; font-size:2.4rem; line-height:1.15; margin:0 0 .5rem; padding:0;}}
.hero p {{color:#dfeaf1; font-size:1.07rem; max-width:780px; margin:0;}}
.hero .eyebrow {{color:{GOLD}; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.8rem;}}
.stat {{background: rgba(3,60,90,.05); border:1px solid rgba(3,60,90,.10); border-radius:14px;
  padding:.9rem 1.1rem;}}
.stat .n {{font-size:1.9rem; font-weight:800; color:{NAVY}; line-height:1.1;}}
.stat .l {{font-size:.8rem; color:#5b6875; text-transform:uppercase; letter-spacing:.06em;}}
.chip {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600;
  margin:0 5px 5px 0; background: rgba(3,60,90,.08); color:{NAVY}; white-space:nowrap;}}
.chip.gold {{background: rgba(170,152,104,.20); color:#6d5b2c;}}
.card-meta {{font-size:.74rem; color:#6b7785; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.25rem;}}
.card-title {{font-weight:700; font-size:1.02rem; line-height:1.35; color:#14212d; min-height:2.75em; margin-bottom:.35rem;}}
.card-sum {{font-size:.88rem; color:#47525e; line-height:1.45; min-height:5.1em; margin-bottom:.5rem;}}
.term-head {{display:flex; align-items:baseline; gap:.7rem; margin:1.6rem 0 .6rem; border-bottom:2px solid {GOLD}; padding-bottom:.3rem;}}
.term-head h3 {{margin:0; padding:0; color:{NAVY};}}
.term-head span {{color:#6b7785; font-size:.9rem;}}
.p-title {{font-size:2rem; font-weight:800; color:{NAVY}; line-height:1.2; margin:.2rem 0 .6rem;}}
.p-meta {{color:#4a5866; font-size:.98rem; margin-bottom:.4rem;}}
.glance {{background: rgba(3,60,90,.04); border:1px solid rgba(3,60,90,.12); border-radius:14px; padding:1rem 1.1rem;}}
.glance dt {{font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:#6b7785; margin-top:.55rem;}}
.glance dd {{margin:0; font-weight:600; color:#1d2733; word-break:break-word;}}
.sec h2 {{color:{NAVY};}}
.comment {{border-left:3px solid {GOLD}; padding:.4rem .8rem; margin:.4rem 0; background:rgba(170,152,104,.08); border-radius:0 8px 8px 0;}}
.comment small {{color:#6b7785;}}
.st-key-hp_website {{position:absolute; left:-9999px; height:0; overflow:hidden;}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def chip(text: str, gold: bool = False) -> str:
    return f'<span class="chip{" gold" if gold else ""}">{esc(text)}</span>'


def status_chip(status: str) -> str:
    label, fg, bg = STATUS_STYLE.get(status, (status, "#444", "#eee"))
    return f'<span class="chip" style="color:{fg};background:{bg}">{esc(label)}</span>'


def open_proposal(slug: str):
    st.query_params["p"] = slug


def card(p: Proposal, key_prefix: str = "c"):
    with st.container(border=True):
        badge = chip("Community", gold=True) if p.kind == "external" else ""
        title = p.title if len(p.title) <= 120 else p.title[:117].rsplit(" ", 1)[0] + "…"
        st.markdown(
            f'<div class="card-meta">{esc(p.term)} · #{p.number}</div>'
            f'<div class="card-title">{esc(title)}</div>'
            f'<div class="card-sum">{esc(p.summary)}</div>'
            f'<div>{badge}{"".join(chip(t) for t in p.tags[:3])}</div>',
            unsafe_allow_html=True)
        st.button("View proposal →", key=f"{key_prefix}_{p.slug}", on_click=open_proposal,
                  args=(p.slug,), width="stretch")


def proposal_view(p: Proposal, store, show_downloads: bool = True, key: str = "v"):
    png = store.read(p.png)
    st.html(pretty.CSS + pretty.header_html(p))

    if show_downloads:
        n_ = p.number
        md = store.read(p.file(f"proposal_{n_}.md"))
        htm = store.read(p.file(f"proposal_{n_}.html"))
        cols = st.columns(4)
        if htm:
            page = standalone_html(htm.decode("utf-8", "replace"), png, p.png.rsplit("/", 1)[1])
            cols[0].download_button("⬇ Download HTML", page, f"{p.code}.html", "text/html",
                                    width="stretch", key=f"{key}_h_{p.slug}")
        if md:
            cols[1].download_button("⬇ Download Markdown", md, f"proposal_{n_}.md", "text/markdown",
                                    width="stretch", key=f"{key}_m_{p.slug}")
        cols[2].download_button("⬇ input.json", json.dumps(p.data, indent=2), "input.json",
                                "application/json", width="stretch", key=f"{key}_j_{p.slug}")
        if p.is_public:
            cols[3].link_button("View on GitHub ↗", config.github_folder_url(p.folder), width="stretch")
        st.write("")

    st.html(pretty.body_html(p, png))


def textwrap_short(s, n: int = 160) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def comments_view(review: dict | None, public_only: bool = True):
    hist = (review or {}).get("history", [])
    shown = 0
    for h in hist:
        if public_only and not h.get("public", True):
            continue
        if not (h.get("comment") or h.get("action")):
            continue
        label = h.get("action", "").replace("_", " ")
        st.markdown(f'<div class="comment"><small>{esc(h.get("at", "")[:16].replace("T", " "))} · '
                    f'{esc(h.get("by", ""))} · <b>{esc(label)}</b></small><br>{esc(h.get("comment", ""))}</div>',
                    unsafe_allow_html=True)
        shown += 1
    if not shown:
        st.caption("No activity yet.")
