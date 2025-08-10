import streamlit as st
import tempfile
from src.financial_rag.parsers.parser import MultimodalParser
from src.financial_rag.retrieval.engine import RetrievalEngine

def file_uploader_section():
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        if st.session_state.docs is None:
            st.info("Parsing PDF, please wait...")
            parser = MultimodalParser()
            # Pass the original filename to the parser
            docs = parser.parse(pdf_path, original_filename=uploaded_file.name)
            st.session_state.docs = docs

            st.info("Building retrieval pipeline...")
            engine = RetrievalEngine()
            engine.build_pipeline(docs)
            st.session_state.engine = engine

            st.success("PDF processed successfully!")
