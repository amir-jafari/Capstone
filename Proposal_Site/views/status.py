"""Public page: submitters check their review status, read comments, and resubmit."""
import streamlit as st

from core import data, submissions, ui
from core.form import proposal_form

st.markdown("## 🔎 Check your submission")
st.caption("Enter the submission ID and access code you received when you submitted.")

with st.form("lookup_form"):
    c1, c2 = st.columns(2)
    sid = c1.text_input("Submission ID", placeholder="E-2027-Spring-1")
    code = c2.text_input("Access code", type="password")
    if st.form_submit_button("Look up", type="primary"):
        st.session_state["lookup"] = (sid.strip(), code.strip())

if "lookup" not in st.session_state:
    st.stop()

sid, code = st.session_state["lookup"]
p = next((x for x in data.proposals() if x.kind == "external" and x.code.lower() == sid.lower()), None)
if not p or not submissions.check_code(p.review, code):
    st.error("No submission matches that ID and access code.")
    st.stop()

if msg := st.session_state.pop("status_flash", None):
    st.success(msg)
st.markdown(f"### {ui.esc(p.title)}")
st.markdown(f"Status: {ui.status_chip(p.status)}", unsafe_allow_html=True)
if p.status == "approved":
    st.success("Your proposal was approved and is now published on the site.")
elif p.status == "changes_requested":
    st.warning("The reviewer asked for changes — see the comments below, then edit and resubmit.")

st.markdown("#### Review activity")
ui.comments_view(p.review)

if p.status == "changes_requested":
    st.markdown("#### Edit and resubmit")
    fields, fig, note = proposal_form("resubmit", prefill=p.data, lock_term=True, submit_label="Resubmit for review")
    if fields:
        try:
            with st.spinner("Saving…"):
                png = submissions.to_png(fig) if fig else data.store().read(p.png)
                submissions.resubmit(data.store(), p, fields, png, note)
            st.session_state["status_flash"] = "Resubmitted! The status is back to *Pending review*."
            st.rerun()
        except Exception as e:
            st.error(f"Could not save: {e}")
else:
    with st.expander("View your proposal"):
        ui.proposal_view(p, data.store(), show_downloads=False)
