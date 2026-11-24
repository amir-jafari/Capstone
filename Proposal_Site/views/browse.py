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


def page():
    items = [p for p in data.proposals() if p.is_public]
    if slug := st.query_params.get("p"):
        _detail(slug, items)
        return

    _hero(items)

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
    # One collapsible group per semester: the newest is open, older ones closed (all open if only one).
    st.markdown("""<style>
      [data-testid="stExpander"] summary p{font-size:1.2rem;color:#033C5A;}
      [data-testid="stExpander"] details{border-radius:14px;}
    </style>""", unsafe_allow_html=True)
    st.caption("Click a semester to show or hide its proposals.")
    for idx, (term, ps) in enumerate(groups.items()):
        label = f"**{term}** · {len(ps)} proposal{'s' if len(ps) != 1 else ''}"
        with st.expander(label, expanded=(idx == 0 or len(groups) == 1)):
            for i in range(0, len(ps), 3):
                for col, p in zip(st.columns(3), ps[i:i + 3]):
                    with col:
                        ui.card(p)


page()
