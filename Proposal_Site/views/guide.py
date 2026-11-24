"""How it works: what each tab does and how to get a proposal approved."""
import streamlit as st

from core import ui

st.markdown("""<div class="hero"><div class="eyebrow">The George Washington University · Data Science Program</div>
<h1>How it works</h1>
<p>What each tab of this site is for, and the steps to take a project idea to an approved,
published capstone proposal.</p></div>""", unsafe_allow_html=True)

TABS = [
    ("🏠", "Proposals", "Everyone",
     "Browse every published capstone proposal by year and semester and open any one to read "
     "its objective, data, approach and timeline. A good place to see the expected depth before you write."),
    ("💡", "Propose a project", "Partners, faculty, students",
     "Fill in the proposal form — the grey text in each box is a full example. Pick the target year and "
     "semester and upload one figure of your idea. You get a proposal number when you submit."),
    ("🔎", "Check submission", "Submitters",
     "No login. Choose your semester and proposal number to see its status and the reviewer's comments. "
     "Confirm your email to edit the proposal if changes are requested."),
    ("🔐", "Reviewer", "Program faculty only",
     "Password-protected. The program reviews each submission and approves it, asks for changes, "
     "or declines it, with comments you can read on Check submission."),
]

STEPS = [
    ("Look at examples", "Open a few proposals on the <i>Proposals</i> tab to see the level of detail expected."),
    ("Submit the form", "On <i>Propose a project</i> fill in every required field (*) — Objective, Dataset, "
                        "Rationale and Approach in some detail, a week-by-week Timeline — and upload a figure."),
    ("Note your number", "After submitting you see e.g. <i>Spring 2027 · Proposal #3</i>. "
                         "Keep it — it is how you find your proposal again."),
    ("Follow the review", "Check <i>Check submission</i>. If the status is <i>Changes requested</i>, read the "
                          "comments, click <i>Modify this proposal</i>, confirm your email and resubmit."),
    ("Get published", "Once <b style='display:inline'>approved</b>, your proposal is listed here on the "
                      "Proposals page for students to choose."),
]


def _how_it_works():
    cards = "".join(f'<div class="tab-card"><div class="ti">{i}</div><div class="tn">{ui.esc(n)}</div>'
                    f'<div class="tw">{ui.esc(w)}</div><div class="td">{ui.esc(d)}</div></div>'
                    for i, n, w, d in TABS)
    steps = "".join(f'<div class="step"><b>{t}</b><span>{d}</span></div>' for t, d in STEPS)
    legend = " ".join(f"{ui.status_chip(s)} {txt}" for s, txt in [
        ("pending", "waiting for the reviewer ·"),
        ("changes_requested", "edit and resubmit (it goes back to pending) ·"),
        ("approved", "published on the Proposals tab ·"),
        ("rejected", "not accepted this time (cannot be edited)")])
    st.markdown(
        f'<div class="how-h">How this site works</div>'
        f'<div class="how-sub">Use the tabs at the top of the page.</div><div class="tabs-grid">{cards}</div>'
        f'<div class="how-h">How to get your proposal approved</div>'
        f'<div class="how-sub">From idea to a published capstone project in five steps.</div>'
        f'<div class="steps">{steps}</div><div class="legend"><b>Statuses:</b> {legend}</div>',
        unsafe_allow_html=True)

_how_it_works()
