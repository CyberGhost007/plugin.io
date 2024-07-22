import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import ollama
import pickle
from PyPDF2 import PdfReader
import chardet
import pandas as pd
import json
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration management
class Config:
    def __init__(self, config_path: str = 'config'):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        env = os.getenv('ENVIRONMENT', 'development')
        config_file = f'{self.config_path}/{env}.json'
        
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_file}. Using default configuration.")
            self.config = self.default_config()
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_file}. Using default configuration.")
            self.config = self.default_config()

    def default_config(self):
        return {
            "model_name": "qwen2",
            "embedding_model_name": "mxbai-embed-large",
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "search_k": 20,
            "ef_search": 100,
            "index_file": "faiss_hnsw_index.bin",
            "docs_file": "documents.pkl"
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

config = Config()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

class PersistentEmbeddingDatabase:
    def __init__(self, model_name: str, embedding_model_name: str, index_file: str, docs_file: str):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
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
        M = 16  # Number of connections per layer
        ef_construction = 200  # Controls index quality and build time trade-off
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = config.get('ef_search', 50)
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
            embedding_response = ollama.embeddings(model=self.embedding_model_name, prompt=doc)
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

    def search(self, query: str, k: int = config.get('search_k', 10)):
        query_embedding_response = ollama.embeddings(model=self.embedding_model_name, prompt=query)
        query_embedding = np.array(query_embedding_response['embedding']).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {self.dimension}")
        
        self.index.hnsw.efSearch = config.get('ef_search', 50)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            document_embedding = self.index.reconstruct(int(idx))
            similarity = float(calculate_cosine_similarity(query_embedding, document_embedding))
            results.append((self.documents[idx], similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def delete_all_embeddings(self):
        if self.index is not None:
            self.create_new_index(self.dimension)
            self.documents = []
            self.save_index()
            logger.info("All embeddings have been deleted")
        else:
            logger.warning("No index to delete")

    def get_stats(self):
        return {
            "total_documents": len(self.documents),
            "total_embeddings": self.index.ntotal if self.index else 0,
            "avg_embeddings_per_doc": self.index.ntotal / len(self.documents) if self.documents else 0
        }

db = PersistentEmbeddingDatabase(
    model_name=config.get('model_name'),
    embedding_model_name=config.get('embedding_model_name'),
    index_file=config.get('index_file'),
    docs_file=config.get('docs_file')
)

def load_document(file_path: str) -> List[str]:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return load_pdf_document(file_path)
    elif file_extension.lower() in ['.md', '.txt']:
        return load_text_document(file_path)
    elif file_extension.lower() == '.csv':
        return load_csv_document(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def load_pdf_document(file_path: str) -> List[str]:
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return split_text(text)

def load_text_document(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return split_text(content)

def load_csv_document(file_path: str) -> List[str]:
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    logger.info(f"Detected encoding: {encoding}")

    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode with {encoding}, falling back to ISO-8859-1")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    records = df.to_dict('records')
    processed_content = []

    for record in records:
        row_content = "\n".join([f"{column}: {value}" for column, value in record.items()])
        processed_content.append(row_content)

    return split_text("\n\n".join(processed_content), chunk_size=config.get('chunk_size', 2000))

def split_text(text: str, chunk_size: int = config.get('chunk_size', 2000)) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=config.get('chunk_overlap', 200),
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    logger.info(f"Split document into {len(chunks)} chunks")
    logger.info(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.2f} characters")
    logger.info(f"Largest chunk size: {max(len(chunk) for chunk in chunks)} characters")
    
    return chunks

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
        results = db.search(query.query, k=config.get('search_k', 30))
        
        if not results:
            return {"message": "No relevant documents found."}

        max_context_length = config.get('max_context_length', 8000)
        context = ""
        for doc, score in results:
            if len(context) + len(doc) + 2 <= max_context_length:
                context += doc + "\n\n"
            else:
                break

        prompt = f"""Use the following pieces of context to answer the question at the end. If you cannot answer the question based on the context, say "I don't have enough information to answer that question." Provide an extremely detailed, comprehensive, and long answer. Include as much relevant information as possible.

        Context:
        {context}

        Question: {query.query}
        Answer:"""

        response = ollama.chat(
            model=config.get('model_name'),
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions based solely on the provided context. Do not use any external knowledge. Provide extremely detailed, comprehensive, and long answers. Include as much relevant information as possible.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                "num_predict": config.get('max_response_tokens', 4096),
            }
        )

        generated_answer = response['message']['content']
        
        result = {
            "generated_answer": generated_answer,
            "relevant_documents": [{"content": doc[:1000] + "..." if len(doc) > 1000 else doc, "score": float(score) if not math.isnan(score) else None} for doc, score in results]
        }

        if query.original_answer:
            original_embedding = np.array(ollama.embeddings(model=db.embedding_model_name, prompt=query.original_answer)['embedding'])
            generated_embedding = np.array(ollama.embeddings(model=db.embedding_model_name, prompt=generated_answer)['embedding'])
            similarity_score = float(calculate_cosine_similarity(original_embedding, generated_embedding))
            result["similarity_score"] = similarity_score if not math.isnan(similarity_score) else None
        
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
    return config.config

@app.get("/indexed_documents")
async def get_indexed_documents():
    return {"documents": [doc[:100] + "..." if len(doc) > 100 else doc for doc in db.documents]}

@app.get("/embedding_stats")
async def get_embedding_stats():
    return db.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)