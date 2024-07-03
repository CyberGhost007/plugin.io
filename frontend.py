import streamlit as st
import requests
import json
import os
import plotly.express as px
import pandas as pd

# Constants
API_URL = "http://localhost:8000"

# Function to query the backend
def query_backend(endpoint, method="GET", data=None, files=None):
    url = f"{API_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, headers=headers, data=json.dumps(data))
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

# App title and description
st.set_page_config(page_title="FloatStream.io", layout="wide", page_icon="⛵")
st.title("⛵ FloatStream.io")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Document", "Query Document", "Manage Embeddings"])

if page == "Home":
    st.markdown("""
        This application offers a comprehensive solution for managing and querying your documents. Here’s what you can do:
        - **Upload Documents**: Upload documents in various formats (PDF, TXT, CSV).
        - **Index and Embed**: Automatically index and embed documents for efficient querying.
        - **Search and Query**: Query the indexed documents to find relevant information.
        - **Manage Embeddings**: Manage the stored embeddings, including the ability to delete all embeddings.

        Use the navigation menu on the left to get started.
    """)

    st.markdown("### Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("📤 **Upload Documents**")
        st.markdown("""
            - PDF, TXT, CSV
            - Automatic Text Extraction
        """)

    with col2:
        st.markdown("🔎 **Query Documents**")
        st.markdown("""
            - Efficient Search
            - Contextual Results
        """)

    with col3:
        st.markdown("🎯 **Manage Embeddings**")
        st.markdown("""
            - Persistent Storage
            - Easy Management
        """)

    # st.markdown("### How It Works")
    # st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

elif page == "Upload Document":
    st.header("Upload Document")
    st.markdown("Upload a document to be indexed and embedded.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "md"])
    
    if uploaded_file is not None:
        with st.spinner("Indexing document..."):
            files = {"file": uploaded_file.getvalue()}
            response = query_backend("index", method="POST", files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)})
            if response:
                st.success(f"Document '{uploaded_file.name}' indexed successfully!")

elif page == "Query Document":
    st.header("Query Document")
    st.markdown("Enter a query to search the indexed documents.")

    query_text = st.text_area("Enter your query:")
    original_answer = st.text_area("Optional: Provide your original answer for similarity comparison (Leave empty if not needed):")

    if st.button("Submit Query"):
        if query_text:
            with st.spinner("Searching documents..."):
                data = {"query": query_text, "original_answer": original_answer}
                response = query_backend("query", method="POST", data=data)
                if response:
                    st.success("Query processed successfully!")
                    st.markdown("### Generated Answer")
                    st.write(response.get("generated_answer", "No answer generated."))

                    st.markdown("### Relevant Documents")
                    for i, doc in enumerate(response.get("relevant_documents", [])):
                        with st.expander(f"Document {i+1} (Score: {doc['score']})"):
                            st.write(doc['content'])

                    if "similarity_score" in response:
                        st.write(f"**Similarity Score:** {response['similarity_score']}")
        else:
            st.error("Please enter a query.")

elif page == "Manage Embeddings":
    st.header("Manage Embeddings")
    st.markdown("Manage the document embeddings.")
    
    if st.button("Delete All Embeddings"):
        with st.spinner("Deleting all embeddings..."):
            response = query_backend("delete_all", method="DELETE")
            if response:
                st.success("All embeddings have been deleted successfully!")

st.sidebar.markdown("### Configuration")
config = query_backend("config")
if config:
    st.sidebar.json(config)

st.sidebar.markdown("### Indexed Documents")
indexed_documents = query_backend("indexed_documents")
if indexed_documents:
    st.sidebar.write(indexed_documents)
