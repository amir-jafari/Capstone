"""Landing page: browse proposals by year and semester, and the single proposal view (?p=<slug>)."""
from collections import OrderedDict

import streamlit as st

from core import catalog, config, data, ui


def _detail(slug: str, items):
    p = catalog.find(items, slug)
    if st.button("← Back to all proposals"):
        del st.query_params["p"]
        st.rerun()
    if not p or not p.is_public:
        st.warning("This proposal was not found or is not public yet.")
        return
    ui.proposal_view(p, data.store())


def _hero(items):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">The George Washington University · Data Science Program</div>
        <h1>{ui.esc(config.SITE_TITLE)}</h1>
        <p>Explore every capstone project proposed for the program — objectives, datasets, approaches and
        timelines — by year and semester.</p></div>""",
        unsafe_allow_html=True)
    terms = {(p.year, p.semester) for p in items}
    stats = [(len(items), "Proposals"), (len(terms), "Semesters"),
             (len({p.year for p in items}), "Years")]
    for c, (n, label) in zip(st.columns(3), stats):
        c.markdown(f'<div class="stat"><div class="n">{n}</div><div class="l">{label}</div></div>',
                   unsafe_allow_html=True)
    st.write("")


TABS = [
    ("🏠", "Proposals", "Everyone",
     "This page. Browse every published capstone proposal by year and semester and open any one to read "
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
    ("Look at examples", "Open a few proposals below to see the level of detail expected."),
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
        ("approved", "published on this page ·"),
        ("rejected", "not accepted this time (cannot be edited)")])
    st.markdown(
        f'<div class="how-h">How this site works</div>'
        f'<div class="how-sub">Use the tabs at the top of the page.</div><div class="tabs-grid">{cards}</div>'
        f'<div class="how-h">How to get your proposal approved</div>'
        f'<div class="how-sub">From idea to a published capstone project in five steps.</div>'
        f'<div class="steps">{steps}</div><div class="legend"><b>Statuses:</b> {legend}</div>'
        f'<div class="how-h">Browse proposals</div>',
        unsafe_allow_html=True)


def page():
    items = [p for p in data.proposals() if p.is_public]
    if slug := st.query_params.get("p"):
        _detail(slug, items)
        return

    _hero(items)
    _how_it_works()

    years = sorted({p.year for p in items}, reverse=True)
    c1, c2, _ = st.columns([1, 1, 2])
    year = c1.selectbox("Year", ["All years"] + years)
    sem = c2.selectbox("Semester", ["All semesters"] + [s for s in config.SEMESTERS
                                                         if any(p.semester == s for p in items)])
    shown = [p for p in items
             if (year == "All years" or p.year == year) and (sem == "All semesters" or p.semester == sem)]
    if not shown:
        st.info("No proposals for this selection.")
        return

    groups = OrderedDict()
    for p in shown:
        groups.setdefault(p.term, []).append(p)
    # One collapsible group per semester, all closed by default (open only if the filter leaves one).
    st.markdown("""<style>
      [data-testid="stExpander"] summary p{font-size:1.2rem;color:#033C5A;}
      [data-testid="stExpander"] details{border-radius:14px;}
    </style>""", unsafe_allow_html=True)
    st.caption("Click a semester to show or hide its proposals.")
    for term, ps in groups.items():
        label = f"**{term}** · {len(ps)} proposal{'s' if len(ps) != 1 else ''}"
        with st.expander(label, expanded=(len(groups) == 1)):
            for i in range(0, len(ps), 3):
                for col, p in zip(st.columns(3), ps[i:i + 3]):
                    with col:
                        ui.card(p)


page()
