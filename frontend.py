import streamlit as st
import requests
import json
import os

BASE_URL = "http://localhost:8000"

def index_document(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        with st.spinner("Indexing document..."):
            response = requests.post(f"{BASE_URL}/index", files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error during indexing: {str(e)}")
        st.error(f"Response content: {e.response.content if e.response else 'No response'}")
        return None

def query_document(query, original_answer=None):
    try:
        data = {"query": query, "original_answer": original_answer}
        response = requests.post(f"{BASE_URL}/query", json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error during querying: {str(e)}")
        st.error(f"Response content: {e.response.content if e.response else 'No response'}")
        return None

def delete_all_embeddings():
    try:
        response = requests.delete(f"{BASE_URL}/delete_all")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error during deletion: {str(e)}")
        st.error(f"Response content: {e.response.content if e.response else 'No response'}")
        return None

def get_indexed_documents():
    try:
        response = requests.get(f"{BASE_URL}/indexed_documents")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error retrieving indexed documents: {str(e)}")
        return []

def load_config():
    try:
        response = requests.get(f"{BASE_URL}/config")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error retrieving configuration: {str(e)}")
        return {}

st.title("Plugin.io - Chat your documents!")

# Sidebar for configuration display
st.sidebar.header("Configuration")
config = load_config()
st.sidebar.subheader("Model Configuration")
st.sidebar.json(config)

st.sidebar.subheader("System Information")
st.sidebar.text(f"Backend URL: {BASE_URL}")
try:
    response = requests.get(f"{BASE_URL}/health")
    st.sidebar.text(f"Backend status: {'Online' if response.status_code == 200 else 'Offline'}")
except requests.RequestException:
    st.sidebar.text("Backend status: Offline or unreachable")

# Main content
st.header("Upload and Index Documents")
uploaded_file = st.file_uploader("Choose a file to index", type=["pdf", "txt", "csv"])
if uploaded_file is not None:
    if st.button("Index Document"):
        result = index_document(uploaded_file)
        if result:
            st.success(result["message"])

# Display indexed documents in a collapsible view
with st.expander("View Indexed Documents", expanded=False):
    st.subheader("Indexed Documents")
    indexed_docs = get_indexed_documents()
    if indexed_docs:
        for doc in indexed_docs:
            st.text(f"- {doc}")
    else:
        st.text("No documents indexed yet.")

st.header("Query Documents")
query = st.text_input("Enter your query")
use_original_answer = st.checkbox("Compare with original answer")
original_answer = st.text_area("Enter the original answer (optional)") if use_original_answer else None

if st.button("Submit Query"):
    if query:
        with st.spinner("Processing query..."):
            result = query_document(query, original_answer)
        if result:
            st.subheader("Generated Answer")
            st.write(result["generated_answer"])
            
            if use_original_answer and "similarity_score" in result:
                st.subheader("Similarity Score")
                st.write(result["similarity_score"])
            
            st.subheader("Relevant Documents")
            for doc in result["relevant_documents"]:
                st.write(f"Content: {doc['content'][:100]}...")
                st.write(f"Score: {doc['score']}")
                st.write("---")
    else:
        st.warning("Please enter a query.")

if st.button("Delete All Embeddings"):
    if st.button("Confirm Deletion"):
        with st.spinner("Deleting all embeddings..."):
            result = delete_all_embeddings()
        if result:
            st.success(result["message"])