"""The proposal form, shared by the Submit page and the resubmission flow."""
from __future__ import annotations

import re
import textwrap
from datetime import date

import streamlit as st

from core import config

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_FIG = 5 * 1024 * 1024

HELP = {
    "Objective": "What should the team build or find out? A short overview followed by a numbered list of key objectives works well.",
    "Dataset": "Which data will be used? Include links, size, and access conditions (public / requires agreement).",
    "Rationale": "Why does this matter? The problem, who benefits, and what gap it fills.",
    "Approach": "How will the team tackle it? Methods, models, tools, and evaluation plan — numbered steps are great.",
    "Timeline": "One line per week or phase, e.g.\nWeek 1: Literature review and data access\nWeek 2-3: Baseline models…",
    "Research Contributions": "Optional: expected novel findings, publications, open-source artifacts.",
    "Possible Issues": "Risks and mitigations: data access, compute, scope.",
    "Additional Resources": "Optional: papers, repos, tutorials the team should start from.",
}


def term_options() -> list[tuple[int, str]]:
    t = date.today()
    y = t.year
    seq = [(y, "Spring"), (y, "Fall"), (y + 1, "Spring"), (y + 1, "Fall"), (y + 2, "Spring")]
    cur = (y, "Spring") if t.month <= 5 else (y, "Fall")
    order = lambda x: (x[0], config.SEMESTER_ORDER[x[1]])
    return [x for x in seq if order(x) > order(cur)][:3]


def _clean(s: str) -> str:
    return textwrap.dedent(s or "").strip()


def proposal_form(key: str, prefill: dict | None = None, lock_term: bool = False,
                  submit_label: str = "Submit proposal"):
    """Returns (fields | None, png_bytes | None, extra_note)."""
    pre = prefill or {}
    with st.form(key, border=True):
        st.markdown("##### 1 · About you")
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name *", pre.get("Proposed by", ""))
        email = c2.text_input("Your email *", pre.get("Proposed by email", ""))
        c3, c4 = st.columns(2)
        collab = c3.text_input("Organization / collaborator", pre.get("collaborator", ""),
                               help="Company, lab, agency, or department sponsoring the project.")
        funding = c4.text_input("Funding opportunity", pre.get("funding_opportunity", ""),
                                help="Optional: any funding, internship, or compute support attached.")

        st.markdown("##### 2 · Project")
        opts = term_options()
        if lock_term:
            st.caption(f"Target semester: **{pre.get('Semester')} {pre.get('Year')}**")
            term = (int(pre["Year"]), pre["Semester"])
        else:
            term = st.selectbox("Target semester *", opts, format_func=lambda x: f"{x[1]} {x[0]}")
        title = st.text_input("Project title *", pre.get("project_name", ""), max_chars=250)
        c5, c6 = st.columns(2)
        students = c5.selectbox("Expected number of students *", ["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"],
                                index=["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"].index(
                                    str(pre.get("Expected Number Students", "2")).strip())
                                if str(pre.get("Expected Number Students", "2")).strip()
                                in ["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"] else 1)
        repo = c6.text_input("GitHub repo (optional)", pre.get("github_repo", ""), placeholder="owner/repo")

        vals = {}
        for k, h, req in [("Objective", 220, True), ("Dataset", 160, True), ("Rationale", 160, True),
                          ("Approach", 220, True), ("Timeline", 200, True),
                          ("Research Contributions", 120, False), ("Possible Issues", 120, False),
                          ("Additional Resources", 100, False)]:
            vals[k] = st.text_area(f"{k}{' *' if req else ''}", _clean(pre.get(k, "")), height=h,
                                   help=HELP[k], placeholder=HELP[k])

        st.markdown("##### 3 · Figure (optional)")
        fig = st.file_uploader("A diagram of the pipeline or idea (PNG/JPG, max 5 MB)",
                               type=["png", "jpg", "jpeg"])
        note = ""
        if prefill:
            note = st.text_area("What did you change? (visible to the reviewer)", height=80)
        hp = st.text_input("Website", key="hp_website")   # honeypot, hidden by CSS
        agree = st.checkbox("I understand this proposal (including my name and email) will be stored "
                            "in the program's public GitHub repository.", value=bool(prefill))
        sent = st.form_submit_button(submit_label, type="primary", width="stretch")

    if not sent:
        return None, None, ""
    errors = []
    if hp:
        errors.append("Submission rejected.")
    if len(name.strip()) < 2:
        errors.append("Please enter your name.")
    if not EMAIL_RE.match(email.strip()):
        errors.append("Please enter a valid email.")
    if len(title.strip()) < 10:
        errors.append("The title should be at least 10 characters.")
    for k in ("Objective", "Dataset", "Rationale", "Approach"):
        if len(vals[k].strip()) < 80:
            errors.append(f"**{k}** needs more detail (at least 80 characters).")
    if len(vals["Timeline"].strip()) < 20:
        errors.append("Please add a timeline.")
    if fig is not None and fig.size > MAX_FIG:
        errors.append("The figure is larger than 5 MB.")
    if not agree:
        errors.append("Please confirm the public-repository notice.")
    if errors:
        st.error("\n".join(f"- {e}" for e in errors))
        return None, None, ""

    fields = {
        "Year": term[0], "Semester": term[1],
        "project_name": title.strip(),
        **{k: v.strip() for k, v in vals.items()},
        "Expected Number Students": students,
        "Proposed by": name.strip(), "Proposed by email": email.strip(),
        "instructor": config.ADVISOR_NAME, "instructor_email": config.ADVISOR_EMAIL,
        "collaborator": collab.strip(), "funding_opportunity": funding.strip(),
        "github_repo": repo.strip(),
    }
    return fields, (fig.getvalue() if fig else None), note.strip()
