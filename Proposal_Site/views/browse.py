"""Landing page + searchable gallery + single proposal view (?p=<slug>)."""
from collections import Counter, OrderedDict

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
        timelines — or propose a project of your own for a student team to take on.</p></div>""",
        unsafe_allow_html=True)
    terms = {(p.year, p.semester) for p in items}
    topics = Counter(t for p in items for t in p.tags)
    stats = [(len(items), "Proposals"), (len(terms), "Semesters"),
             (sum(p.kind == "external" for p in items), "Community projects"), (len(topics), "Topic areas")]
    cols = st.columns(4)
    for c, (n, l) in zip(cols, stats):
        c.markdown(f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>',
                   unsafe_allow_html=True)
    st.write("")
    a, b, _ = st.columns([1.2, 1.2, 3])
    if a.button("💡 Propose a project", type="primary", width="stretch"):
        st.switch_page("views/submit.py")
    if b.button("🔎 Check a submission", width="stretch"):
        st.switch_page("views/status.py")


def page():
    items = [p for p in data.proposals() if p.is_public]
    if slug := st.query_params.get("p"):
        _detail(slug, items)
        return

    _hero(items)

    st.markdown("#### Browse proposals")
    f1, f2, f3, f4 = st.columns([2.4, 1.2, 1.1, 1.1])
    q = f1.text_input("Search", placeholder="Search titles, datasets, methods, people…",
                      label_visibility="collapsed")
    years = sorted({p.year for p in items}, reverse=True)
    ysel = f2.multiselect("Year", years, placeholder="All years", label_visibility="collapsed")
    ssel = f3.selectbox("Semester", ["All semesters", "Spring", "Fall"], label_visibility="collapsed")
    src = f4.selectbox("Source", ["All sources", "Program", "Community"], label_visibility="collapsed")
    topic_counts = Counter(t for p in items for t in p.tags)
    topics = st.pills("Topics", [t for t, _ in topic_counts.most_common()], selection_mode="multi",
                      label_visibility="collapsed")

    words = q.lower().split()
    shown = [p for p in items
             if all(w in p.search_blob() for w in words)
             and (not ysel or p.year in ysel)
             and (ssel == "All semesters" or p.semester == ssel)
             and (src == "All sources" or (src == "Community") == (p.kind == "external"))
             and (not topics or set(topics) & set(p.tags))]

    if not shown:
        st.info("No proposals match these filters.")
        return
    st.caption(f"Showing {len(shown)} of {len(items)} proposals")

    groups = OrderedDict()
    for p in shown:
        groups.setdefault(p.term, []).append(p)
    for term, ps in groups.items():
        st.markdown(f'<div class="term-head"><h3>{term}</h3><span>{len(ps)} proposal'
                    f'{"s" if len(ps) != 1 else ""}</span></div>', unsafe_allow_html=True)
        for i in range(0, len(ps), 3):
            cols = st.columns(3)
            for col, p in zip(cols, ps[i:i + 3]):
                with col:
                    ui.card(p)


page()
