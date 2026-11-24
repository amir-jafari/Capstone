"""The proposal form, shared by the Submit page and the resubmission flow."""
from __future__ import annotations

import re
import textwrap
from datetime import date

import streamlit as st

from core import config
from core.examples import EXAMPLE

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


FORM_SEMESTERS = ["Spring", "Fall"]   # semesters offered in the form
YEARS_AHEAD = 5                        # Year dropdown: this year .. this year + YEARS_AHEAD


def _term_key(term: tuple[int, str]) -> tuple[int, int]:
    return term[0], config.SEMESTER_ORDER[term[1]]


def current_term() -> tuple[int, str]:
    t = date.today()
    return (t.year, "Spring") if t.month <= 5 else (t.year, "Fall")


def next_term() -> tuple[int, str]:
    """First semester after the current one (the form's default)."""
    y, s = current_term()
    later = [(y, x) for x in FORM_SEMESTERS] + [(y + 1, x) for x in FORM_SEMESTERS]
    return next(t for t in later if _term_key(t) > _term_key((y, s)))


def year_options() -> list[int]:
    y = next_term()[0]                   # first year that still has an open semester
    return list(range(y, y + YEARS_AHEAD + 1))


def _clean(s: str) -> str:
    return textwrap.dedent(s or "").strip()


def proposal_form(key: str, prefill: dict | None = None, lock_term: bool = False,
                  submit_label: str = "Submit proposal", has_figure: bool = False):
    """Returns (fields | None, png_bytes | None, extra_note).

    A figure is required; `has_figure` = the proposal being edited already has one (keep it)."""
    pre = prefill or {}
    with st.form(key, border=True):
        st.markdown("##### 1 · About you")
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name *", pre.get("Proposed by", ""), placeholder=EXAMPLE["Proposed by"])
        email = c2.text_input("Your email *", pre.get("Proposed by email", ""),
                              placeholder=EXAMPLE["Proposed by email"])
        c3, c4 = st.columns(2)
        collab = c3.text_input("Organization / collaborator", pre.get("collaborator", ""),
                               placeholder=EXAMPLE["collaborator"],
                               help="Company, lab, agency, or department sponsoring the project.")
        funding = c4.text_input("Funding opportunity", pre.get("funding_opportunity", ""),
                                placeholder=EXAMPLE["funding_opportunity"],
                                help="Optional: any funding, internship, or compute support attached.")

        st.markdown("##### 2 · Project")
        if lock_term:
            st.caption(f"Target semester: **{pre.get('Semester')} {pre.get('Year')}**")
            term = (int(pre["Year"]), pre["Semester"])
        else:
            dy, ds = next_term()
            years = year_options()
            cy, cs = st.columns(2)
            ty = cy.selectbox("Target year *", years, index=years.index(dy))
            ts = cs.selectbox("Target semester *", FORM_SEMESTERS, index=FORM_SEMESTERS.index(ds))
            term = (int(ty), ts)
        title = st.text_input("Project title *", pre.get("project_name", ""), max_chars=250,
                              placeholder=EXAMPLE["project_name"])
        c5, c6 = st.columns(2)
        students = c5.selectbox("Expected number of students *", ["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"],
                                index=["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"].index(
                                    str(pre.get("Expected Number Students", "2")).strip())
                                if str(pre.get("Expected Number Students", "2")).strip()
                                in ["1", "2", "3", "4", "5", "1-2", "2-3", "3-4"] else 1)
        repo = c6.text_input("GitHub repo (optional)", pre.get("github_repo", ""), placeholder=EXAMPLE["github_repo"])

        st.caption("The grey text in each box is an example project in the program's proposal format — "
                   "replace it with your own. Keep the same structure (numbered Key Objectives, "
                   "one line per week in the Timeline) so every proposal reads the same.")
        vals = {}
        for k, h, req in [("Objective", 380, True), ("Dataset", 300, True), ("Rationale", 200, True),
                          ("Approach", 380, True), ("Timeline", 320, True),
                          ("Research Contributions", 160, False), ("Possible Issues", 160, False),
                          ("Additional Resources", 140, False)]:
            vals[k] = st.text_area(f"{k}{' *' if req else ''}", _clean(pre.get(k, "")), height=h,
                                   help=HELP[k], placeholder=EXAMPLE[k])

        st.markdown("##### 3 · Figure *")
        fig = st.file_uploader("A diagram of the pipeline or idea (PNG/JPG, max 5 MB)"
                               + (" — leave empty to keep the current figure" if has_figure else " *"),
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
    if not lock_term and _term_key(term) <= _term_key(current_term()):
        errors.append(f"The target semester must be after the current one ({' '.join(current_term()[::-1])}) "
                      f"— pick {next_term()[1]} {next_term()[0]} or later.")
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
    if fig is None and not has_figure:
        errors.append("Please upload a **figure** (a diagram of the pipeline or idea) — it is required.")
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
