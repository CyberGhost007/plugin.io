import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import ollama
import pickle
from PyPDF2 import PdfReader
import json
import csv
import io
import chardet
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load configuration
with open('data/config.json', 'r') as config_file:
    config = json.load(config_file)

class PersistentEmbeddingDatabase:
    def __init__(self, model_name: str = config['model_name'], index_file: str = config['index_file'], docs_file: str = config['docs_file']):
        self.model_name = model_name
        self.index_file = os.path.join('data', index_file)
        self.docs_file = os.path.join('data', docs_file)
        self.documents = []
        self.index = None
        self.dimension = None
        self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            self.load_index()
        else:
            logger.info("No existing index found. A new one will be created when documents are added.")

    def create_new_index(self, dimension):
        M = 16
        ef_construction = 200
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = config['ef_search']
        self.dimension = dimension
        logger.info(f"Created new HNSW index with dimension {dimension}")

    def load_index(self):
        self.index = faiss.read_index(self.index_file)
        self.dimension = self.index.d
        with open(self.docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded HNSW index with {self.index.ntotal} vectors of dimension {self.dimension} and {len(self.documents)} documents")

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Saved HNSW index with {self.index.ntotal} vectors and {len(self.documents)} documents")
        else:
            logger.warning("No index to save")

    def add_documents(self, documents: List[str]):
        all_embeddings = []
        for doc in documents:
            embedding_response = ollama.embeddings(model=self.model_name, prompt=doc)
            embedding = np.array(embedding_response['embedding']).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            all_embeddings.append(embedding)
        
        embeddings = np.array(all_embeddings)
        
        if self.index is None:
            self.create_new_index(embeddings.shape[1])
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.save_index()

    def search(self, query: str, k: int = config['search_k'], ef_search: int = config['ef_search']):
        query_embedding_response = ollama.embeddings(model=self.model_name, prompt=query)
        query_embedding = np.array(query_embedding_response['embedding']).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.dimension}")
        
        self.index.hnsw.efSearch = ef_search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            score = float(scores[0][i])
            results.append((self.documents[idx], score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def delete_all_embeddings(self):
        if self.index is not None:
            self.create_new_index(self.dimension)
            self.documents = []
            self.save_index()
            logger.info("All embeddings have been deleted")
        else:
            logger.warning("No index to delete")

db = PersistentEmbeddingDatabase(index_file=config['index_file'], docs_file=config['docs_file'])

def load_document(file_path: str):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return load_pdf_document(file_path)
    elif file_extension.lower() in ['.md', '.txt']:
        return load_text_document(file_path)
    elif file_extension.lower() == '.csv':
        return load_csv_document(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def load_pdf_document(file_path: str):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return split_text(text)

def load_text_document(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return split_text(content)

def load_csv_document(file_path: str):
    # Detect the file encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    logger.info(f"Detected encoding: {encoding}")

    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode with {encoding}, falling back to ISO-8859-1")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Convert DataFrame to a list of dictionaries
    records = df.to_dict('records')

    # Create a list to store the processed content
    processed_content = []

    # Process each row
    for record in records:
        row_content = ""
        for column, value in record.items():
            row_content += f"{column}: {value}\n"
        processed_content.append(row_content.strip())

    # Join all processed content
    full_content = "\n\n".join(processed_content)
    
    return split_text(full_content)

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    logger.info(f"Split document into {len(chunks)} chunks")
    logger.info(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.2f} characters")
    logger.info(f"Largest chunk size: {max(len(chunk) for chunk in chunks)} characters")
    
    return chunks

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class Query(BaseModel):
    query: str
    original_answer: Optional[str] = None

@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = os.path.join('data', file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        
        chunks = load_document(file_path)
        db.add_documents(chunks)
        os.remove(file_path)
        return {"message": f"Successfully indexed document: {file.filename}"}
    except Exception as e:
        logger.exception("An error occurred during indexing")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(query: Query):
    try:
        results = db.search(query.query, k=config['search_k'], ef_search=config['ef_search'])
        
        if not results:
            return {"message": "No relevant documents found."}

        context = "\n\n".join([doc for doc, _ in results])
        
        prompt = f"""Use the following pieces of context to answer the question at the end. If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question: {query.query}
        Answer:"""

        response = ollama.chat(model=config['model_name'], messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant that answers questions based solely on the provided context. Do not use any external knowledge.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])

        generated_answer = response['message']['content']
        
        result = {
            "generated_answer": generated_answer,
            "relevant_documents": [{"content": doc, "score": score} for doc, score in results]
        }

        if query.original_answer:
            original_embedding = np.array(ollama.embeddings(model=db.model_name, prompt=query.original_answer)['embedding'])
            generated_embedding = np.array(ollama.embeddings(model=db.model_name, prompt=generated_answer)['embedding'])
            similarity_score = calculate_cosine_similarity(original_embedding, generated_embedding)
            result["similarity_score"] = similarity_score
        
        return result
    except Exception as e:
        logger.exception("An error occurred during querying")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_all")
async def delete_all_embeddings():
    try:
        db.delete_all_embeddings()
        return {"message": "All embeddings have been deleted."}
    except Exception as e:
        logger.exception("An error occurred during deletion")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    return config

@app.get("/indexed_documents")
async def get_indexed_documents():
    return [os.path.basename(doc) for doc in db.documents]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)