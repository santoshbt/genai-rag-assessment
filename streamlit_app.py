import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

API_BASE = "http://localhost:8000/api/v1"

st.title("📚 RAG Document Q&A System")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if st.button("Ingest Documents", type="primary"):
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    response = requests.post(f"{API_BASE}/ingest", files=files)
                    
                    if response.status_code == 201:
                        result = response.json()
                        st.success(f"✅ {file.name}: {result['total_chunks']} chunks")
                    else:
                        st.error(f"❌ {file.name}: {response.json().get('detail', 'Error')}")
        else:
            st.warning("Please upload files first")

# Main chat interface
st.header("💬 Ask Questions")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"**{i}. {source['document_name']}** (Page {source['page_number']})")
                    st.text(source['chunk_excerpt'])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(f"{API_BASE}/query", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.markdown(result["answer"])
                
                # Update session ID
                st.session_state.session_id = result["session_id"]
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                })
                
                # Show sources
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.caption(f"**{i}. {source['document_name']}** (Page {source['page_number']})")
                            st.text(source['chunk_excerpt'])
            else:
                error = response.json().get("detail", "Unknown error")
                st.error(f"Error: {error}")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.session_id = None
    st.rerun()
