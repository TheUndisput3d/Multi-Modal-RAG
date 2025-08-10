import streamlit as st

def results_display_section(selected_query):
    st.markdown("---")
    
    st.markdown("### Processing Query")
    st.info(f"**Question:** {selected_query}")
    
    with st.spinner("Searching through documents..."):
        result = st.session_state.engine.query(selected_query)
    
    st.markdown("---")
    
    st.markdown("### Answer")
    answer = result.get("result", "No answer generated.")
    
    st.success(answer)
    
    st.markdown("---")
    st.markdown("### Source Documents")
    
    source_docs = result.get("source_documents", [])
    
    if source_docs:
        st.markdown(f"*Found {len(source_docs)} relevant documents*")
        
        doc_types = {}
        for doc in source_docs:
            doc_type = doc.metadata.get('type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        for doc_type, docs in doc_types.items():
            st.markdown(f"#### {doc_type.title()} Documents ({len(docs)})")
            
            for i, doc in enumerate(docs, 1):
                with st.expander(f"{doc_type.title()} Document {i} - {doc.metadata.get('source', 'Unknown Source')}"):
                    st.markdown("**Content:**")
                    st.text_area(
                        "Document content",
                        doc.page_content,
                        height=200,
                        key=f"doc_content_{doc_type}_{i}",
                        label_visibility="collapsed"
                    )
                    
                    st.markdown("**Metadata:**")
                    metadata_str = "\n".join([f"- **{k}:** {v}" for k, v in doc.metadata.items()])
                    st.markdown(metadata_str)
            
            st.markdown("")
    else:
        st.warning("No source documents found.")

def show_welcome_info():
    st.info("Please upload a PDF document to start querying.")
    
    st.markdown("### How to use:")
    st.markdown("""
    1. **Upload a PDF** using the file uploader above
    2. **Wait for processing** - the system will parse text, tables, and images
    3. **Choose your query method:**
       - **Custom Query**: Ask your own questions
       - **Predefined Examples**: Try sample questions
    4. **Review the results** - see the answer and source documents used
    """)
    
    st.markdown("### System Features:")
    st.markdown("""
    - **Multi-modal parsing**: Extracts text, tables, and image descriptions
    - **Hybrid retrieval**: Combines semantic and keyword search
    - **Financial focus**: Optimized for financial document analysis
    - **Source tracking**: Shows which documents were used for each answer
    """)
