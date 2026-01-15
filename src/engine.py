"""
LLM Engine with RAG and conversation history.
"""

import os
from typing import Iterator, Optional, Callable
import config


class RAG:
    """Simple RAG: loads .txt files and searches by keyword."""
    
    def __init__(self):
        self.documents = []
        self._load_documents()
    
    def _load_documents(self):
        """Loads all .txt files from RAG_FOLDER."""
        if not config.RAG_ENABLED:
            return
        
        folder = config.RAG_FOLDER
        if not os.path.exists(folder):
            os.makedirs(folder)
            return
        
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    # Split into chunks
                    chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                    for chunk in chunks:
                        self.documents.append({
                            "source": filename,
                            "content": chunk
                        })
    
    def _split_text(self, text: str, chunk_size: int) -> list:
        """Splits text into chunks."""
        words = text.split()
        chunks = []
        current = []
        count = 0
        
        for word in words:
            current.append(word)
            count += len(word) + 1
            if count >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                count = 0
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def search(self, query: str) -> str:
        """Searches documents for relevant chunks."""
        if not self.documents:
            return ""
        
        query_words = set(query.lower().split())
        scored = []
        
        for doc in self.documents:
            content_words = set(doc["content"].lower().split())
            score = len(query_words & content_words)
            if score > 0:
                scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:config.RAG_TOP_K]
        
        if not top:
            return ""
        
        context = "\n\n".join([
            f"[{doc['source']}]: {doc['content']}" 
            for _, doc in top
        ])
        
        return f"\n\nRelevant context:\n{context}\n"


class LLMEngine:
    """LLM Engine with history and RAG."""
    
    def __init__(self):
        self.llm = None
        self.history = []
        self.rag = RAG()
        self.is_ready = False
        self.error = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Loads the LLM model."""
        def log(msg):
            if on_progress:
                on_progress(msg)
        
        try:
            log("Importing llama_cpp...")
            from llama_cpp import Llama
            
            model_path = self._get_model_path()
            log(f"Model: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            log("Loading model...")
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
    
    def _get_model_path(self) -> str:
        """Gets model path (PyInstaller compatible)."""
        import sys
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, config.MODEL_FILE)
        return config.MODEL_FILE
    
    def generate(self, message: str) -> Iterator[str]:
        """Generates response with RAG and history."""
        if not self.is_ready:
            return
        
        # RAG context
        rag_context = self.rag.search(message) if config.RAG_ENABLED else ""
        
        # Build history string
        history_str = ""
        for h in self.history[-6:]:  # Last 3 exchanges
            history_str += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            history_str += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"
        
        # Full prompt
        system = config.SYSTEM_PROMPT + rag_context
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += history_str
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        # Generate
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
        
        # Save to history
        self.history.append({
            "user": message,
            "assistant": full_response.strip()
        })
    
    def clear_history(self):
        """Clears conversation history."""
        self.history = []