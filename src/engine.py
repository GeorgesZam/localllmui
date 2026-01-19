"""
LLM Engine with RAG (embeddings) and conversation history.
FIXED: Added .py support, writable paths for PyInstaller, better RAG injection
"""

import os
import sys
import json
import numpy as np
from typing import Iterator, Optional, Callable, List
import config

# Document parsers
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import openpyxl
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


def get_resource_path(relative_path: str) -> str:
    """Gets path compatible with PyInstaller (read-only bundled files)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


def get_writable_path(filename: str) -> str:
    """Gets writable path for user data - works with PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        # When packaged, use user's home directory
        app_data = os.path.join(os.path.expanduser("~"), ".localchat")
    else:
        # Development mode - use current directory
        app_data = os.path.join(os.path.dirname(__file__), "..", ".localchat")
    
    os.makedirs(app_data, exist_ok=True)
    return os.path.join(app_data, filename)


class EmbeddingModel:
    """Simple embedding model using sentence-transformers."""
    
    def __init__(self):
        self.model = None
    
    def load(self):
        """Loads the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            return True
        except Exception as e:
            print(f"[Embedding] Error: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts to vectors."""
        if self.model is None:
            return np.array([])
        return self.model.encode(texts, normalize_embeddings=True)


class RAG:
    """RAG with semantic search using embeddings."""
    
    # FIXED: Support many more file types including code files
    SUPPORTED_EXTENSIONS = (
        # Documents
        '.txt', '.md', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.csv',
        # Code files
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        # Config/Data files  
        '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.toml',
        # Scripts
        '.sh', '.bash', '.zsh', '.ps1', '.bat', '.sql',
        # Web
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
    )
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = EmbeddingModel()
        
        # FIXED: All user data goes to writable location
        self.index_file = get_writable_path("index.json")
        self.embeddings_file = get_writable_path("embeddings.npy")
        self.user_docs_folder = get_writable_path("documents")
        
        # Bundled data folder (read-only when packaged)
        self.bundled_data_folder = get_resource_path(config.RAG_FOLDER)
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Initializes RAG system."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        # Create user docs folder
        os.makedirs(self.user_docs_folder, exist_ok=True)
        log(f"User docs folder: {self.user_docs_folder}")
        
        log("Loading embedding model...")
        if not self.embedding_model.load():
            log("Embedding model failed, using keyword search")
            self._load_documents_simple()
            return True
        
        # Check if index exists and is valid
        if os.path.exists(self.index_file) and os.path.exists(self.embeddings_file):
            log("Loading existing index...")
            self._load_index()
            if self.documents:
                log(f"Loaded {len(self.documents)} chunks from index")
            else:
                log("Index empty, rebuilding...")
                self._build_index(log)
        else:
            log("No index found, building...")
            self._build_index(log)
        
        log(f"RAG ready: {len(self.documents)} chunks indexed")
        return True
    
    def _load_index(self):
        """Loads pre-built index."""
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.embeddings = np.load(self.embeddings_file)
            print(f"[RAG] Index loaded: {len(self.documents)} docs")
        except Exception as e:
            print(f"[RAG] Error loading index: {e}")
            self.documents = []
            self.embeddings = None
    
    def _save_index(self, log: Callable[[str], None]):
        """Saves index to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            np.save(self.embeddings_file, self.embeddings)
            log(f"Index saved: {len(self.documents)} chunks")
        except Exception as e:
            log(f"Failed to save index: {e}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        if not HAS_PDF:
            return ""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"[RAG] PDF error {file_path}: {e}")
            return ""
    
    def _extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel."""
        if not HAS_EXCEL:
            return ""
        try:
            text = ""
            wb = openpyxl.load_workbook(file_path, data_only=True)
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text += f"\n=== Sheet: {sheet_name} ===\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    if row_text.strip():
                        text += row_text + "\n"
            wb.close()
            return text.strip()
        except Exception as e:
            print(f"[RAG] Excel error {file_path}: {e}")
            return ""
    
    def _extract_text_from_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint."""
        if not HAS_PPTX:
            return ""
        try:
            text = ""
            prs = Presentation(file_path)
            for i, slide in enumerate(prs.slides, 1):
                text += f"\n=== Slide {i} ===\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text += shape.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"[RAG] PPTX error {file_path}: {e}")
            return ""
    
    def _read_document(self, file_path: str) -> str:
        """Read document based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        print(f"[RAG] Reading: {file_path} (ext: {ext})")
        
        if ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif ext in ('.xlsx', '.xls'):
            return self._extract_text_from_excel(file_path)
        elif ext in ('.pptx', '.ppt'):
            return self._extract_text_from_pptx(file_path)
        else:
            # Read as text file (works for .py, .js, .json, .txt, etc.)
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"[RAG] Read {len(content)} chars from {file_path}")
                    return content
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"[RAG] Error reading {file_path}: {e}")
                    return ""
            return ""
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Splits text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        if len(words) < 20:
            # Very short text - keep as single chunk if meaningful
            if len(words) > 5:
                return [text]
            return []
        
        # Overlap of 20%
        step = max(1, int(chunk_size * 0.8))
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) >= 20:
                chunks.append(" ".join(chunk_words))
        
        # Make sure we don't miss the end
        if not chunks:
            chunks.append(text)
        
        return chunks
    
    def _build_index(self, log: Callable[[str], None]):
        """Builds index from all document folders."""
        all_chunks = []
        
        # Scan both bundled and user folders
        folders = []
        if os.path.exists(self.bundled_data_folder):
            folders.append(("bundled", self.bundled_data_folder))
        if os.path.exists(self.user_docs_folder):
            folders.append(("user", self.user_docs_folder))
        
        for folder_type, folder in folders:
            log(f"Scanning {folder_type} folder: {folder}")
            
            if not os.path.isdir(folder):
                continue
                
            for filename in os.listdir(folder):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in self.SUPPORTED_EXTENSIONS:
                    log(f"⏭ Skip {filename} (unsupported: {ext})")
                    continue
                
                file_path = os.path.join(folder, filename)
                
                try:
                    text = self._read_document(file_path)
                    
                    if not text or len(text.strip()) < 10:
                        log(f"⚠ {filename}: empty or too short")
                        continue
                    
                    chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            "source": filename,
                            "chunk_id": i,
                            "content": chunk
                        })
                    
                    log(f"✓ {filename}: {len(chunks)} chunks ({len(text)} chars)")
                    
                except Exception as e:
                    log(f"✗ {filename}: {e}")
        
        if not all_chunks:
            log("No documents found to index")
            self.documents = []
            self.embeddings = None
            return
        
        log(f"Encoding {len(all_chunks)} chunks...")
        
        # Create embeddings
        texts = [c["content"] for c in all_chunks]
        self.embeddings = self.embedding_model.encode(texts)
        self.documents = all_chunks
        
        self._save_index(log)
    
    def _load_documents_simple(self):
        """Fallback: loads documents without embeddings (keyword search)."""
        folders = [self.bundled_data_folder, self.user_docs_folder]
        
        for folder in folders:
            if not os.path.exists(folder):
                continue
            
            for filename in os.listdir(folder):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(folder, filename)
                    try:
                        text = self._read_document(file_path)
                        if text:
                            chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                            for i, chunk in enumerate(chunks):
                                self.documents.append({
                                    "source": filename,
                                    "chunk_id": i,
                                    "content": chunk
                                })
                    except:
                        pass
    
    def search(self, query: str, top_k: int = None) -> str:
        """Searches documents using semantic similarity."""
        if not self.documents:
            print("[RAG] No documents indexed!")
            return ""
        
        top_k = top_k or config.RAG_TOP_K
        print(f"[RAG] Searching '{query}' in {len(self.documents)} chunks")
        
        results = []
        
        if self.embeddings is not None and self.embedding_model.model is not None:
            # Semantic search
            query_embedding = self.embedding_model.encode([query])[0]
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                score = similarities[idx]
                if score > 0.15:  # Lower threshold
                    results.append((self.documents[idx], score))
                    print(f"[RAG] Found: {self.documents[idx]['source']} (score: {score:.3f})")
        else:
            # Keyword fallback
            query_words = set(query.lower().split())
            scored = []
            for doc in self.documents:
                content_words = set(doc["content"].lower().split())
                score = len(query_words & content_words)
                if score > 0:
                    scored.append((doc, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = scored[:top_k]
        
        if not results:
            print("[RAG] No matches found")
            return ""
        
        # Format context for LLM
        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Source: {doc['source']}]\n{doc['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        return f"\n\n### RELEVANT DOCUMENTS ###\n{context}\n### END DOCUMENTS ###\n"
    
    def add_documents(self, file_paths: list, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Adds new documents and rebuilds index."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            os.makedirs(self.user_docs_folder, exist_ok=True)
            
            import shutil
            added_count = 0
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in self.SUPPORTED_EXTENSIONS:
                    log(f"⚠ {filename}: unsupported ({ext})")
                    log(f"   Supported: {', '.join(self.SUPPORTED_EXTENSIONS[:10])}...")
                    continue
                
                dest = os.path.join(self.user_docs_folder, filename)
                shutil.copy2(file_path, dest)
                log(f"✓ Copied: {filename}")
                added_count += 1
            
            if added_count == 0:
                log("No supported files to add")
                return False
            
            # Rebuild entire index
            log("Rebuilding index...")
            self._build_index(log)
            
            return len(self.documents) > 0
            
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


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
            print(f"[Engine] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            # Initialize RAG
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
                n_gpu_layers=-1,  # -1 = ALL layers on GPU (Metal on Mac, CUDA on Windows if available)
                verbose=True      # Show Metal/CUDA logs at startup
            )
            
            self.is_ready = True
            log("Ready!")
            return True
            
        except Exception as e:
            self.error = str(e)
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(self, message: str) -> Iterator[str]:
        """Generates response with RAG and history."""
        if not self.is_ready:
            yield "Error: Model not ready"
            return
        
        # Get RAG context - ALWAYS try to search if RAG enabled
        rag_context = ""
        if config.RAG_ENABLED:
            print(f"[Engine] RAG has {len(self.rag.documents)} documents")
            if self.rag.documents:
                rag_context = self.rag.search(message)
                print(f"[Engine] RAG context: {len(rag_context)} chars")
            else:
                print("[Engine] RAG enabled but no documents loaded")
        
        # Build conversation history
        history_str = ""
        for h in self.history[-4:]:  # Réduit à 4 derniers échanges pour vitesse
            history_str += f"<start_of_turn>user\n{h['user']}<end_of_turn>\n"
            history_str += f"<start_of_turn>model\n{h['assistant']}<end_of_turn>\n"
        
        # Build system prompt
        system = config.SYSTEM_PROMPT
        if rag_context:
            system += rag_context
            system += "\n\nUse the documents above to answer. Be specific and quote relevant parts."
        
        # Build full prompt (Gemma format)
        prompt = f"<start_of_turn>user\n{system}\n\n{message}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        
        # If we have history, use multi-turn format
        if history_str:
            prompt = f"<start_of_turn>user\n{system}<end_of_turn>\n"
            prompt += history_str
            prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
        
        print(f"[Engine] Prompt: {len(prompt)} chars, History: {len(self.history)} msgs")
        print(f"[Engine] RAG in prompt: {'### RELEVANT DOCUMENTS ###' in prompt}")
        
        # DEBUG: Print first 500 chars of prompt to verify RAG context
        print(f"[Engine] Prompt preview:\n{prompt[:1000]}...")
        
        # Generate
        full_response = ""
        try:
            for chunk in self.llm(
                prompt,
                max_tokens=config.MAX_TOKENS,
                stop=config.STOP_TOKENS,
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                full_response += token
                yield token
        except Exception as e:
            print(f"[Engine] Generation error: {e}")
            yield f"\n[Error: {e}]"
        
        # Save to history
        if full_response.strip():
            self.history.append({
                "user": message,
                "assistant": full_response.strip()
            })
            print(f"[Engine] Saved to history, now {len(self.history)} messages")
    
    def clear_history(self):
        """Clears conversation history."""
        self.history = []
        print("[Engine] History cleared")
