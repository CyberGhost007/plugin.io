import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import os

API_URL = "http://localhost:8000"

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
                response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    st.set_page_config(page_title="FloatStream.io", layout="wide", page_icon="⛵")
    st.title("⛵ FloatStream.io")

    pages = {
        "Home": home_page,
        "Upload Documents": upload_page,
        "Query Documents": query_page,
        "Manage Embeddings": manage_page
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))

    pages[page]()

    st.sidebar.markdown("### Configuration")
    config = query_backend("config")
    if config:
        st.sidebar.json(config)

    st.sidebar.markdown("### Indexed Documents")
    indexed_documents = query_backend("indexed_documents")
    if indexed_documents:
        st.sidebar.write(indexed_documents)

def home_page():
    st.markdown("""
        Welcome to FloatStream.io, your comprehensive solution for document management and querying.

        ### Key Features:
        - 📤 **Upload Documents**: Support for PDF, TXT, CSV, and MD files.
        - 🔎 **Query Documents**: Efficient search with contextual results.
        - 🎯 **Manage Embeddings**: Easy management of indexed documents.

        Use the navigation menu on the left to get started.
    """)

    st.markdown("### How It Works")
    image_path = os.path.join("assets", "arc.png")
    st.image(image_path, caption="FloatStream.io Workflow", use_column_width=True)

def upload_page():
    st.header("Upload Documents")
    st.markdown("Upload one or more documents to be indexed and embedded.")
    
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "csv", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Index Selected Documents"):
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Indexing {file.name}..."):
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    response = query_backend("index", method="POST", files=files)
                    if response:
                        st.success(f"Document '{file.name}' indexed successfully!")
                    else:
                        st.error(f"Failed to index document '{file.name}'")
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("All documents have been processed!")

def query_page():
    st.header("Query Documents")
    st.markdown("Enter a query to search the indexed documents.")

    query_text = st.text_area("Enter your query:")
    original_answer = st.text_area("Optional: Provide your original answer for similarity comparison")

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
                        with st.expander(f"Document {i+1} (Score: {doc['score']:.4f})"):
                            st.write(doc['content'])

                    if "similarity_score" in response:
                        st.write(f"**Similarity Score:** {response['similarity_score']:.4f}")

                    # Visualize document relevance
                    if response.get("relevant_documents"):
                        df = pd.DataFrame(response["relevant_documents"])
                        fig = px.bar(df, x=df.index, y="score", labels={"index": "Document", "score": "Relevance Score"})
                        st.plotly_chart(fig)
        else:
            st.error("Please enter a query.")

def manage_page():
    st.header("Manage Embeddings")
    st.markdown("Manage your document embeddings.")
    
    if st.button("Delete All Embeddings"):
        if st.checkbox("I understand this action cannot be undone"):
            with st.spinner("Deleting all embeddings..."):
                response = query_backend("delete_all", method="DELETE")
                if response:
                    st.success("All embeddings have been deleted successfully!")
        else:
            st.warning("Please confirm that you understand the consequences of this action.")

    st.markdown("### Embedding Statistics")
    stats = query_backend("embedding_stats")
    if stats:
        st.write(f"Total documents: {stats['total_documents']}")
        st.write(f"Total embeddings: {stats['total_embeddings']}")
        st.write(f"Average embeddings per document: {stats['avg_embeddings_per_doc']:.2f}")

if __name__ == "__main__":
    main()