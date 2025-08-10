import asyncio
import streamlit as st
from src.financial_rag.parsers.parser import MultimodalParser
from src.financial_rag.retrieval.engine import RetrievalEngine
import tempfile

from apps.streamlit_app.components.file_uploader import file_uploader_section
from apps.streamlit_app.components.query_interface import query_interface_section
from apps.streamlit_app.components.results_display import results_display_section, show_welcome_info

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Financial RAG System", layout="wide")
st.title("Financial RAG System")

# Initialize session state variables if missing
if "docs" not in st.session_state:
    st.session_state.docs = None
if "engine" not in st.session_state:
    st.session_state.engine = None

# File upload and parsing
file_uploader_section()

# If engine is ready, show query UI and results
if st.session_state.engine:
    st.markdown("---")
    query_interface_section()
else:
    st.markdown("---")
    show_welcome_info()
