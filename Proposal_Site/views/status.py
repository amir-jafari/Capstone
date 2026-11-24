"""Public page: anyone can check community submissions by semester and number — no login."""
import streamlit as st

from core import config, data, submissions, ui
from core.form import proposal_form

st.markdown("## 🔎 Check a submission")
st.caption("Choose the semester and proposal number, then click **Show** to see its review status, "
           "the reviewer's comments, and the full proposal.")

if msg := st.session_state.pop("status_flash", None):
    st.success(msg)

ext = [p for p in data.proposals() if p.kind == "external"]
if not ext:
    st.info("No community submissions yet.")
    st.stop()

# A shared link / redirect after submitting (?y=2027&s=Spring&n=1) shows that proposal directly.
qp = st.query_params
if qp.get("y") and qp.get("s") and qp.get("n", "").isdigit() and "status_show" not in st.session_state:
    st.session_state["status_show"] = (int(qp["y"]), qp["s"], int(qp["n"]))

terms = sorted({(p.year, p.semester) for p in ext},
               key=lambda t: (t[0], config.SEMESTER_ORDER[t[1]]), reverse=True)
shown = st.session_state.get("status_show")

with st.container(border=True):
    c1, c2, c3 = st.columns([1.2, 2.6, 0.8], vertical_alignment="bottom")
    t_idx = next((i for i, t in enumerate(terms) if shown and t == shown[:2]), None)
    term = c1.selectbox("Semester", terms, index=t_idx, placeholder="Select a semester",
                        format_func=lambda t: f"{t[1]} {t[0]}")
    in_term = sorted((p for p in ext if (p.year, p.semester) == term), key=lambda p: p.number) if term else []
    nums = [p.number for p in in_term]
    n_idx = nums.index(shown[2]) if shown and term == shown[:2] and shown[2] in nums else None
    num = c2.selectbox("Proposal", nums, index=n_idx, disabled=not term,
                       placeholder="Select a proposal" if term else "Select a semester first",
                       format_func=lambda n: f"#{n} — " + next(p.title for p in in_term if p.number == n)[:90])
    if c3.button("Show", type="primary", width="stretch", disabled=num is None):
        st.session_state["status_show"] = (term[0], term[1], num)
        st.query_params.update({"y": str(term[0]), "s": term[1], "n": str(num)})
        st.rerun()

if not shown:
    st.stop()
p = next((x for x in ext if (x.year, x.semester, x.number) == shown), None)
if not p:
    st.warning("That submission was not found.")
    st.stop()

if p.status == "approved":
    st.success("✅ Approved — this proposal is published on the Proposals page.")
elif p.status == "changes_requested":
    st.warning("✏️ The reviewer asked for changes — see the comments below.")
elif p.status == "rejected":
    st.error("This proposal was not accepted.")
else:
    st.info("⏳ Waiting for review.")

with st.container(border=True):
    st.markdown("##### 💬 Review activity")
    ui.comments_view(p.review)

if p.status == "changes_requested":
    with st.expander("✏️ Edit and resubmit", expanded=False):
        email = st.text_input("Confirm the submitter's email to edit this proposal")
        if email:
            if email.strip().lower() != p.data.get("Proposed by email", "").strip().lower():
                st.error("That email doesn't match the one on this submission.")
            else:
                fields, fig, note = proposal_form("resubmit", prefill=p.data, lock_term=True,
                                                  has_figure=data.store().read(p.png) is not None,
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

st.write("")
ui.proposal_view(p, data.store(), show_downloads=False, key="status")
