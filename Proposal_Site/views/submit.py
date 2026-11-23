"""Public page: anyone can propose a capstone project."""
import streamlit as st

from core import data, submissions
from core.form import proposal_form

st.markdown("""<div class="hero"><div class="eyebrow">Industry · Government · Research partners</div>
<h1>Propose a capstone project</h1>
<p>Have a real problem and data? Propose it for a team of GWU Data Science master's students.
Your proposal is reviewed by the program; once approved it is published on this site.</p></div>""",
            unsafe_allow_html=True)

if done := st.session_state.get("submitted"):
    y, sem, n = done
    st.success(f"Thank you! Your proposal was submitted as **{sem} {y} · Proposal #{n}**.")
    st.markdown(f"To check its review status and the reviewer's comments, open **Check submission** "
                f"and choose **{sem} {y}** → **#{n}**. No login is needed.")
    c1, c2 = st.columns(2)
    if c1.button("Check status", width="stretch"):
        st.switch_page("views/status.py", query_params={"y": str(y), "s": sem, "n": str(n)})
    if c2.button("Submit another proposal", width="stretch"):
        del st.session_state["submitted"]
        st.rerun()
    st.stop()

with st.expander("What makes a strong proposal?", icon="💡"):
    st.markdown("""
- **A clear, testable objective** — what question will be answered or what will be built.
- **Accessible data** — public, or data you can share with the students under a clear agreement.
- **Right-sized scope** — one semester (~14 weeks) for a team of 2–4 students, with realistic compute.
- **A concrete approach and timeline** — milestones the team can check off week by week.
- Browse past proposals on the home page for examples of the expected depth.
""")

fields, fig, _ = proposal_form("new_proposal")
if fields:
    try:
        with st.spinner("Saving your proposal…"):
            png = submissions.to_png(fig)
            st.session_state["submitted"] = submissions.create(data.store(), fields, png)
        st.rerun()
    except Exception as e:
        st.error(f"Sorry, the proposal could not be saved: {e}")
