"""GWU Data Science Capstone Proposals - Streamlit site.

Run locally:   cd Proposal_Site && streamlit run streamlit_app.py
"""
import streamlit as st

from core import config, ui

st.set_page_config(page_title="Capstone Proposals · GWU Data Science", page_icon="🎓", layout="wide")
ui.inject_css()

pages = [
    st.Page("views/guide.py", title="How it works", icon="❓", url_path="how-it-works"),
    st.Page("views/browse.py", title="Proposals", icon="🏠", default=True),
    st.Page("views/submit.py", title="Propose a project", icon="💡", url_path="submit"),
    st.Page("views/status.py", title="Check submission", icon="🔎", url_path="status"),
    st.Page("views/admin.py", title="Reviewer", icon="🔐", url_path="admin"),
]
nav = st.navigation(pages, position="top")
nav.run()

st.markdown("---")
st.caption(f"The George Washington University · Data Science Program · Advisor: {config.ADVISOR_NAME} "
           f"([{config.ADVISOR_EMAIL}](mailto:{config.ADVISOR_EMAIL})) · "
           f"[Website]({config.SITE_URL}) · [Source on GitHub](https://github.com/{config.GITHUB_REPO})")
