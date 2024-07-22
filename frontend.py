import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from streamlit_chat import message
import time

# Load environment variables
load_dotenv()

# Set API URL based on environment
API_URL = os.getenv('API_URL', 'http://localhost:8000')

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

    pages = {
        "Home": home_page,
        "Upload Documents": upload_page,
        "Chat": improved_chat_page,
        "Retrival Scores": query_page,
        "Manage Embeddings": manage_page
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))

    # Conditionally display the title
    if page != "Chat":
        st.title("⛵ FloatStream.io")

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
    if os.path.exists(image_path):
        st.image(image_path, caption="FloatStream.io Workflow", use_column_width=True)
    else:
        st.warning("Workflow image not found. Please add an image to the 'assets' folder.")

def chat_page():
    # st.header("Chat with AI")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Create a placeholder for the AI's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with a loading indicator
            with st.spinner("FloatStream is thinking..."):
                response = query_backend("query", method="POST", data={"query": prompt})
                full_response = response.get("generated_answer", "I'm sorry, I couldn't generate an answer.")

            # Add the full response to the message placeholder
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.caption("FloatStream can make mistakes. Please verify important information.")

def improved_chat_page():
    # st.header("Chat with ⛵ FloatStream.io")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["role"] == "user", key=str(i))

    # Check if chat is empty
    if not st.session_state.messages:
        # st.info("Please enter your question below to start the conversation.")
        st.info("""
        **How to use FloatStream Chat:**
        1. Type your question or query in the input box below and press Enter.
        2. FloatStream will process your query and provide an answer based on the indexed documents.
        3. You can view relevant source documents by expanding the "View Relevant Documents" section after each response.
        """)
    else:
        # Add a button to clear the chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True)

        with st.spinner("FloatStream is thinking..."):
            response = query_backend("query", method="POST", data={"query": prompt})
            if response and "generated_answer" in response:
                ai_response = response["generated_answer"]
            else:
                ai_response = "I'm sorry, I couldn't generate an answer at this time."

        # Display assistant response in chat message container
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        message(ai_response)

        # Display relevant documents
        if response and "relevant_documents" in response:
            with st.expander("View Relevant Documents"):
                for i, doc in enumerate(response["relevant_documents"]):
                    st.markdown(f"**Document {i+1}** (Score: {doc['score']:.4f})")
                    st.write(doc['content'])

        # Rerun the app to show the clear button after the first message
        st.experimental_rerun()

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
    st.markdown("Verifying the accuracy of Retrieval-Augmented Generation (RAG)")

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

    # Visualize embedding statistics
    if stats:
        fig = px.bar(
            x=["Total Documents", "Total Embeddings"],
            y=[stats['total_documents'], stats['total_embeddings']],
            labels={"x": "Metric", "y": "Count"}
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()