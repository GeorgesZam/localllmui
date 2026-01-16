"""
LLM Engine with RAG (embeddings) and conversation history.
"""

import os
import sys
import json
import numpy as np
from typing import Iterator, Optional, Callable, List
import config


def get_resource_path(relative_path: str) -> str:
    """Gets path compatible with PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


class EmbeddingModel:
    """Simple embedding model using sentence-transformers."""
    
    def __init__(self):
        self.model = None
    
    def load(self):
        """Loads the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Small multilingual model (~100MB)
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return True
        except Exception as e:
            print(f"Embedding model error: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts to vectors."""
        if self.model is None:
            return np.array([])
        return self.model.encode(texts, normalize_embeddings=True)


class RAG:
    """RAG with semantic search using embeddings."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = EmbeddingModel()
        self.index_file = get_resource_path("data/index.json")
        self.embeddings_file = get_resource_path("data/embeddings.npy")
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Initializes RAG system."""
        def log(msg):
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        log("Loading embedding model...")
        if not self.embedding_model.load():
            log("Embedding model failed, using keyword search")
            self._load_documents_simple()
            return True
        
        # Check if index exists
        if os.path.exists(self.index_file) and os.path.exists(self.embeddings_file):
            log("Loading existing index...")
            self._load_index()
        else:
            log("Building index from documents...")
            self._build_index(log)
        
        log(f"RAG ready: {len(self.documents)} chunks indexed")
        return True
    
    def _load_index(self):
        """Loads pre-built index."""
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.embeddings = np.load(self.embeddings_file)
        except Exception as e:
            print(f"Error loading index: {e}")
            self.documents = []
            self.embeddings = None
    
    def _build_index(self, log: Callable[[str], None]):
        """Builds index from documents."""
        folder = get_resource_path(config.RAG_FOLDER)
        
        if not os.path.exists(folder):
            return
        
        # Load all documents
        all_chunks = []
        for filename in os.listdir(folder):
            if filename.endswith((".txt", ".md")):
                path = os.path.join(folder, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                        for i, chunk in enumerate(chunks):
                            all_chunks.append({
                                "source": filename,
                                "chunk_id": i,
                                "content": chunk
                            })
                except:
                    pass
        
        if not all_chunks:
            return
        
        log(f"Encoding {len(all_chunks)} chunks...")
        
        # Create embeddings
        texts = [c["content"] for c in all_chunks]
        self.embeddings = self.embedding_model.encode(texts)
        self.documents = all_chunks
        
        # Save index
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False)
            np.save(self.embeddings_file, self.embeddings)
            log("Index saved")
        except:
            pass
    
    def _load_documents_simple(self):
        """Fallback: loads documents without embeddings."""
        folder = get_resource_path(config.RAG_FOLDER)
        
        if not os.path.exists(folder):
            return
        
        for filename in os.listdir(folder):
            if filename.endswith((".txt", ".md")):
                path = os.path.join(folder, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                        for chunk in chunks:
                            self.documents.append({
                                "source": filename,
                                "content": chunk
                            })
                except:
                    pass
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Splits text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        # Overlap of 20%
        step = int(chunk_size * 0.8)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 50:  # Min 50 words
                chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def search(self, query: str, top_k: int = None) -> str:
        """Searches documents using semantic similarity."""
        if not self.documents:
            return ""
        
        top_k = top_k or config.RAG_TOP_K
        
        # Semantic search with embeddings
        if self.embeddings is not None and self.embedding_model.model is not None:
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Cosine similarity
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold
                    results.append(self.documents[idx])
        else:
            # Fallback: keyword search
            query_words = set(query.lower().split())
            scored = []
            
            for doc in self.documents:
                content_words = set(doc["content"].lower().split())
                score = len(query_words & content_words)
                if score > 0:
                    scored.append((score, doc))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [doc for _, doc in scored[:top_k]]
        
        if not results:
            return ""
        
        context = "\n\n---\n\n".join([
            f"[Source: {doc['source']}]\n{doc['content']}" 
            for doc in results
        ])
        
        return f"\n\nRelevant documents:\n{context}\n"


class LLMEngine:
    """LLM Engine with history and RAG."""
    
    def __init__(self):
        self.llm = None
        self.history = []
        self.rag = RAG()
        self.is_ready = False
        self.error = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Loads the LLM model and RAG."""
        def log(msg):
            if on_progress:
                on_progress(msg)
        
        try:
            # Initialize RAG first
            self.rag.initialize(log)
            
            log("Importing llama_cpp...")
            from llama_cpp import Llama
            
            model_path = get_resource_path(config.MODEL_FILE)
            log(f"Model: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            log("Loading LLM...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=config.CONTEXT_SIZE,
                n_threads=config.THREADS,
                verbose=False
            )
            
            self.is_ready = True
            log("Ready!")
            return True
            
        except Exception as e:
            self.error = str(e)
            log(f"Error: {e}")
            return False
    
    def generate(self, message: str) -> Iterator[str]:
        """Generates response with RAG and history."""
        if not self.is_ready:
            return
        
        rag_context = self.rag.search(message) if config.RAG_ENABLED else ""
        
        history_str = ""
        for h in self.history[-6:]:
            history_str += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            history_str += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"
        
        system = config.SYSTEM_PROMPT + rag_context
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += history_str
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        full_response = ""
        for chunk in self.llm(
            prompt,
            max_tokens=config.MAX_TOKENS,
            stop=config.STOP_TOKENS,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            full_response += token
            yield token
        
        self.history.append({
            "user": message,
            "assistant": full_response.strip()
        })
    
    def clear_history(self):
        """Clears conversation history."""
        self.history = []
