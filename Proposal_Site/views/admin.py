"""Admin: password login, review queue, approve / request changes / reject, comment, promote, delete."""
import hmac
from collections import Counter

import streamlit as st

from core import config, data, submissions, ui

ADMIN_PW = config.secret("admin", "password") or config._top("password") or config._top("ADMIN_PASSWORD")
ADMIN_NAME = config.secret("admin", "name") or config._top("name") or config.ADVISOR_NAME

st.markdown("## 🔐 Reviewer dashboard")

if not ADMIN_PW:
    st.error("No admin password configured. Add `[admin] password = \"...\"` to your Streamlit secrets, "
             "click Save, then reboot the app.")
    found = config.secret_sections()
    st.info("Secrets the app can see (names only): " + (", ".join(f"`{s}`" for s in found) if found else
            "**none** — the secrets are empty or not valid TOML (check for curly “smart quotes” "
            "and the `[admin]` / `[github]` header lines)."))
    st.code('[admin]\npassword = "your-strong-password"\nname = "Amir Jafari"\n\n'
            '[github]\ntoken = "github_pat_..."\nrepo = "amir-jafari/Capstone"\nbranch = "main"', language="toml")
    st.stop()

if not st.session_state.get("is_admin"):
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if hmac.compare_digest(pw.encode(), str(ADMIN_PW).encode()):
                st.session_state["is_admin"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

store = data.store()
top_l, top_r = st.columns([4, 1])
top_l.caption(f"Signed in as **{ADMIN_NAME}** · storage: **{store.mode}**"
              + (f" → `{config.GITHUB_REPO}@{config.GITHUB_BRANCH}`" if store.mode == "github" else
                 " (local disk — commit & push changes yourself)"))
if top_r.button("Log out", width="stretch"):
    st.session_state["is_admin"] = False
    st.rerun()

if msg := st.session_state.pop("admin_flash", None):
    st.success(msg)

ext = [p for p in data.proposals() if p.kind == "external"]
counts = Counter(p.status for p in ext)
cols = st.columns(4)
for c, s in zip(cols, ["pending", "changes_requested", "approved", "rejected"]):
    label = ui.STATUS_STYLE[s][0]
    c.markdown(f'<div class="stat"><div class="n">{counts.get(s, 0)}</div><div class="l">{label}</div></div>',
               unsafe_allow_html=True)
st.write("")

DECISIONS = {"approved": "✅ Approve & publish", "changes_requested": "✏️ Request changes",
             "rejected": "⛔ Reject", "pending": "⏳ Keep pending"}


def _delete(p):
    try:
        submissions.delete(store, p)
        st.session_state["admin_flash"] = f"Deleted {p.code}."
        st.rerun()
    except Exception as e:
        st.error(f"Delete failed: {e}")


def review_panel(p, tab: str):
    head = (f"{ui.STATUS_STYLE.get(p.status, (p.status,))[0]}  ·  {p.code}  ·  "
            f"{p.title[:80]}{'…' if len(p.title) > 80 else ''}  ·  {p.proposer}")
    with st.expander(head, expanded=False):
        # --- action bar, always visible at the top of the panel ---
        a1, a2, a3 = st.columns([2, 1.3, 1.1])
        a1.markdown(f"Status: {ui.status_chip(p.status)}", unsafe_allow_html=True)
        promoted = (p.review or {}).get("promoted_to")
        if promoted:
            a2.caption(f"Promoted to `{promoted}`")
        elif p.status == "approved":
            if a2.button("⭐ Promote to official", key=f"pr_{tab}_{p.slug}", width="stretch",
                         help=f"Copies it into {config.OFFICIAL}/{p.year}/{p.semester}/<next number>/"):
                try:
                    folder = submissions.promote(store, p, ADMIN_NAME)
                    st.session_state["admin_flash"] = f"Promoted {p.code} → {folder}"
                    st.rerun()
                except Exception as e:
                    st.error(f"Promote failed: {e}")
        with a3.popover("🗑 Delete", width="stretch"):
            st.write(f"Permanently delete **{p.code}** and its folder from GitHub?")
            if st.button("Yes, delete", key=f"del_{tab}_{p.slug}", type="primary"):
                _delete(p)

        view, act = st.tabs(["Proposal", "Review & comments"])
        with view:
            ui.proposal_view(p, store, key=f"adm_{tab}_{p.slug}")
        with act:
            ui.comments_view(p.review, public_only=False)
            with st.form(f"rev_{tab}_{p.slug}"):
                dec = st.radio("Decision", list(DECISIONS), format_func=DECISIONS.get, horizontal=True,
                               index=list(DECISIONS).index(p.status) if p.status in DECISIONS else 3)
                comment = st.text_area("Comment", placeholder="Feedback for the submitter…")
                public = st.checkbox("Visible to the submitter", value=True,
                                     help="Uncheck for an internal note only you see here. "
                                          "Note: review.json lives in the GitHub repo.")
                if st.form_submit_button("Save review", type="primary"):
                    if dec == p.status and not comment.strip():
                        st.warning("Nothing to save.")
                    else:
                        try:
                            submissions.review(store, p, dec, comment.strip(), public, ADMIN_NAME)
                            st.session_state["admin_flash"] = f"Saved review for {p.code}: {dec}."
                            st.rerun()
                        except Exception as e:
                            st.error(f"Save failed: {e}")


queue = [p for p in ext if p.status == "pending"]
t1, t2, t3 = st.tabs([f"Review queue ({len(queue)})", f"All community submissions ({len(ext)})", "Help"])
with t1:
    if not queue:
        st.info("Nothing waiting for review. 🎉")
    for p in queue:
        review_panel(p, "q")
with t2:
    flt = st.segmented_control("Status", ["All"] + list(DECISIONS), default="All",
                               format_func=lambda s: "All" if s == "All" else ui.STATUS_STYLE[s][0])
    rejected = [p for p in ext if p.status == "rejected"]
    if rejected:
        with st.popover(f"🗑 Delete all rejected ({len(rejected)})"):
            st.write("Permanently delete these submissions from GitHub (one commit):")
            st.markdown("\n".join(f"- {p.code} · {ui.esc(p.title[:70])}" for p in rejected))
            if st.button("Yes, delete all rejected", type="primary", key="del_all_rejected"):
                try:
                    submissions.delete_many(store, rejected)
                    st.session_state["admin_flash"] = f"Deleted {len(rejected)} rejected submission(s)."
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
    for p in ext:
        if flt in (None, "All") or p.status == flt:
            review_panel(p, "all")
with t3:
    st.markdown(f"""
- Submissions are saved to `{config.EXTERNAL}/<Year>/<Semester>/<N>/` with the same files as your
  Report_Gen proposals (`input.json`, `proposal_N.md`, `proposal_N.html`, figure) plus `review.json`.
- **Approve** publishes the proposal on the public site (Community badge).
- **Request changes** lets the submitter edit and resubmit using their access code.
- **Promote** copies an approved proposal into `{config.OFFICIAL}/` with the next official number.
- Your own proposals keep coming from the Report_Gen pipeline as before — the site picks up every
  folder under `{config.OFFICIAL}/` automatically.
""")
