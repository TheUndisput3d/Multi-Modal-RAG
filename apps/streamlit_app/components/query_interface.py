import streamlit as st
from apps.streamlit_app.components.results_display import results_display_section

def query_interface_section():
    st.subheader("Query the Document")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### Choose your query method:")
        
        query_type = st.radio(
            "Select query type:",
            ["Custom Query", "Predefined Examples"],
            horizontal=True
        )
        
        selected_query = None
        if query_type == "Custom Query":
            user_query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What was Apple's total revenue in Q3 2022?"
            )
            query_button = st.button("Ask Question", type="primary")
            selected_query = user_query if query_button and user_query else None
            
        else:
            st.markdown("**Select from example questions:**")
            predefined_queries = [
                "What was the primary reason for the increase in iPhone net sales during the third quarter of 2022 compared to the same quarter in 2021?",
                "Which two new Mac models powered by the M2 chip were introduced at the end of the third quarter of 2022?",
                "What operating system updates did Apple announce in the third quarter of 2022 that were expected to be available in fall 2022?",
            ]
            
            selected_example = st.selectbox(
                "Choose a predefined query:",
                ["Select a question..."] + predefined_queries
            )
            
            query_button = st.button("Run Selected Query", type="primary")
            selected_query = selected_example if query_button and selected_example != "Select a question..." else None

    with col2:
        st.markdown("#### Quick Stats")
        if st.session_state.docs:
            total_docs = len(st.session_state.docs)
            table_docs = len([doc for doc in st.session_state.docs if 'table' in doc.metadata.get('type', '')])
            text_docs = len([doc for doc in st.session_state.docs if 'text' in doc.metadata.get('type', '')])
            
            st.metric("Total Documents", total_docs)
            st.metric("Table Documents", table_docs)
            st.metric("Text Documents", text_docs)

    if selected_query:
        results_display_section(selected_query)
