"""Session-wide access to the storage backend and the proposal catalog."""
import streamlit as st

from core import catalog
from core.storage import get_storage


@st.cache_resource(show_spinner=False)
def store():
    return get_storage()


def proposals():
    return catalog.load_all(store())
