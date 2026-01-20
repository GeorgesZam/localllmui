"""
LLM Engine with RAG (embeddings) and conversation history.
IMPROVED: Better chunking, better embeddings, better retrieval, SOURCE DISPLAY
"""

import os
import sys
import json
import numpy as np
from typing import Iterator, Optional, Callable, List, Tuple
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
        app_data = os.path.join(os.path.expanduser("~"), ".localchat")
    else:
        app_data = os.path.join(os.path.dirname(__file__), "..", ".localchat")
    
    os.makedirs(app_data, exist_ok=True)
    return os.path.join(app_data, filename)


class EmbeddingModel:
    """Embedding model using sentence-transformers with better model."""
    
    def __init__(self):
        self.model = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None):
        """Loads the embedding model."""
        def log(msg):
            print(f"[Embedding] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = getattr(config, 'EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
            log(f"Loading embedding model: {model_name}")
            
            self.model = SentenceTransformer(model_name)
            log("Embedding model loaded!")
            return True
        except Exception as e:
            log(f"Error loading embedding model: {e}")
            return False
    
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encodes texts to vectors."""
        if self.model is None:
            return np.array([])
        
        if is_query and 'bge' in getattr(config, 'EMBEDDING_MODEL', '').lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class RAG:
    """RAG with improved semantic search and source tracking."""
    
    SUPPORTED_EXTENSIONS = (
        '.txt', '.md', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.csv',
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.toml',
        '.sh', '.bash', '.zsh', '.ps1', '.bat', '.sql',
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
    )
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = EmbeddingModel()
        self.last_sources = []  # NEW: Store last used sources
        
        self.index_file = get_writable_path("index.json")
        self.embeddings_file = get_writable_path("embeddings.npy")
        self.user_docs_folder = get_writable_path("documents")
        self.bundled_data_folder = get_resource_path(config.RAG_FOLDER)
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Initializes RAG system."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        os.makedirs(self.user_docs_folder, exist_ok=True)
        log(f"User docs folder: {self.user_docs_folder}")
        
        log("Loading embedding model...")
        if not self.embedding_model.load(on_progress):
            log("Embedding model failed, using keyword search")
            self._load_documents_simple()
            return True
        
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
    
    def _split_text(self, text: str, chunk_size: int, overlap: int = None) -> List[str]:
        """Splits text into overlapping chunks with better sentence handling."""
        if overlap is None:
            overlap = getattr(config, 'RAG_CHUNK_OVERLAP', 100)
        
        text = text.strip()
        if len(text) < 50:
            return [text] if len(text) > 20 else []
        
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length <= chunk_size:
                current_chunk.extend(sentence_words)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                if overlap > 0 and current_chunk:
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words + sentence_words
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence_words
                    current_length = sentence_length
        
        if current_chunk and len(current_chunk) >= 20:
            chunks.append(" ".join(current_chunk))
        
        if not chunks and len(text.split()) >= 20:
            words = text.split()
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) >= 20:
                    chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def _build_index(self, log: Callable[[str], None]):
        """Builds index from all document folders."""
        all_chunks = []
        
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
        
        texts = [c["content"] for c in all_chunks]
        self.embeddings = self.embedding_model.encode(texts, is_query=False)
        self.documents = all_chunks
        
        self._save_index(log)
    
    def _load_documents_simple(self):
        """Fallback: loads documents without embeddings."""
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
    
    def search(self, query: str, top_k: int = None) -> Tuple[str, List[dict]]:
        """Searches documents using semantic similarity.
        
        Returns:
            Tuple of (context_string, list_of_sources)
        """
        if not self.documents:
            print("[RAG] No documents indexed!")
            self.last_sources = []
            return "", []
        
        top_k = top_k or config.RAG_TOP_K
        min_score = getattr(config, 'RAG_MIN_SCORE', 0.25)
        
        print(f"[RAG] Searching '{query}' in {len(self.documents)} chunks (top_k={top_k}, min_score={min_score})")
        
        results = []
        
        if self.embeddings is not None and self.embedding_model.model is not None:
            query_embedding = self.embedding_model.encode([query], is_query=True)[0]
            similarities = np.dot(self.embeddings, query_embedding)
            
            top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append((self.documents[idx], score))
                    print(f"[RAG] ✓ {self.documents[idx]['source']} chunk {self.documents[idx]['chunk_id']} (score: {score:.3f})")
                else:
                    print(f"[RAG] ✗ {self.documents[idx]['source']} chunk {self.documents[idx]['chunk_id']} (score: {score:.3f} < {min_score})")
            
            results = results[:top_k]
        else:
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
            print("[RAG] No matches found above threshold")
            self.last_sources = []
            return "", []
        
        # Store sources for display
        self.last_sources = []
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            # Store source info
            self.last_sources.append({
                "index": i,
                "source": doc["source"],
                "chunk_id": doc["chunk_id"],
                "score": score,
                "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            })
            context_parts.append(f"[Document {i} - {doc['source']}]\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        print(f"[RAG] Returning {len(results)} chunks, {len(context)} chars")
        
        return context, self.last_sources
    
    def get_last_sources(self) -> List[dict]:
        """Returns the sources used in the last search."""
        return self.last_sources
    
    def format_sources_for_display(self) -> str:
        """Formats sources for UI display."""
        if not self.last_sources:
            return ""
        
        lines = ["", "📚 Sources used:"]
        for src in self.last_sources:
            lines.append(f"  [{src['index']}] {src['source']} (chunk {src['chunk_id']}, score: {src['score']:.2f})")
            # Show preview (first 150 chars)
            preview = src['preview'][:150].repla
