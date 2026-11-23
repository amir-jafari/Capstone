"""Public page: anyone can check community submissions by semester and number — no login."""
import streamlit as st

from core import config, data, submissions, ui
from core.form import proposal_form

st.markdown("## 🔎 Check a submission")
st.caption("Pick the semester and proposal number to see its review status and the reviewer's comments.")

if msg := st.session_state.pop("status_flash", None):
    st.success(msg)

ext = [p for p in data.proposals() if p.kind == "external"]
if not ext:
    st.info("No community submissions yet.")
    st.stop()

terms = sorted({(p.year, p.semester) for p in ext},
               key=lambda t: (t[0], config.SEMESTER_ORDER[t[1]]), reverse=True)
qp = st.query_params
default_term = next((i for i, t in enumerate(terms)
                     if str(t[0]) == qp.get("y") and t[1] == qp.get("s")), 0)

c1, c2 = st.columns([1, 2])
term = c1.selectbox("Semester", terms, index=default_term, format_func=lambda t: f"{t[1]} {t[0]}")
in_term = sorted((p for p in ext if (p.year, p.semester) == term), key=lambda p: p.number)
nums = [p.number for p in in_term]
default_n = nums.index(int(qp["n"])) if qp.get("n", "").isdigit() and int(qp["n"]) in nums else 0
num = c2.selectbox("Proposal number", nums, index=default_n,
                   format_func=lambda n: f"#{n} — " + next(p.title for p in in_term if p.number == n)[:90])

p = next(x for x in in_term if x.number == num)
st.query_params.update({"y": str(p.year), "s": p.semester, "n": str(p.number)})   # shareable link

st.markdown(f"### {ui.esc(p.title)}")
st.markdown(f"{ui.status_chip(p.status)} &nbsp; {ui.esc(p.term)} · #{p.number} · proposed by "
            f"{ui.esc(p.proposer)}", unsafe_allow_html=True)
if p.status == "approved":
    st.success("Approved — this proposal is published on the Proposals page.")
elif p.status == "changes_requested":
    st.warning("The reviewer asked for changes — see the comments below.")
elif p.status == "rejected":
    st.error("This proposal was not accepted.")
else:
    st.info("Waiting for review.")

st.markdown("#### Review activity")
ui.comments_view(p.review)

with st.expander("View the proposal", expanded=p.status != "changes_requested"):
    ui.proposal_view(p, data.store(), show_downloads=False, key="status")

if p.status == "changes_requested":
    st.markdown("#### Edit and resubmit")
    email = st.text_input("Confirm the submitter's email to edit this proposal")
    if email:
        if email.strip().lower() != p.data.get("Proposed by email", "").strip().lower():
            st.error("That email doesn't match the one on this submission.")
        else:
            fields, fig, note = proposal_form("resubmit", prefill=p.data, lock_term=True,
                                              submit_label="Resubmit for review")
            if fields:
                try:
                    with st.spinner("Saving…"):
                        png = submissions.to_png(fig) if fig else data.store().read(p.png)
                        submissions.resubmit(data.store(), p, fields, png, note)
                    st.session_state["status_flash"] = "Resubmitted! The status is back to *Pending review*."
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save: {e}")
