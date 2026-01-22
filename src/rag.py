"""
RAG system - Optimized with lazy loading and caching.
"""

import os
import json
import re
import shutil
import hashlib
import numpy as np
from typing import Optional, Callable, List, Tuple

import config
from utils import get_resource_path, get_writable_path, get_file_hash
from ocr import OCRProcessor

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

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class EmbeddingModel:
    """Lazy-loaded embedding model for memory efficiency."""
    
    def __init__(self):
        self._model = None
        self._loaded = False
    
    @property
    def model(self):
        return self._model
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        if self._loaded:
            return True
        
        def log(msg):
            print(f"[Embedding] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            from sentence_transformers import SentenceTransformer
            bundled_path = get_resource_path(config.EMBEDDING_MODEL_FOLDER)
            
            if os.path.exists(bundled_path):
                log(f"Loading: {bundled_path}")
                self._model = SentenceTransformer(bundled_path)
                self._loaded = True
                log("Embedding model loaded!")
                return True
            else:
                log(f"Model not found: {bundled_path}")
                return False
        except Exception as e:
            log(f"Error: {e}")
            return False
    
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self._model is None:
            return np.array([])
        
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        batch_size = getattr(config, 'BATCH_SIZE', 512)
        return self._model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            batch_size=batch_size
        )
    
    def unload(self):
        """Free memory by unloading model."""
        self._model = None
        self._loaded = False
        import gc
        gc.collect()


class DocumentParser:
    """Optimized document parser with format detection."""
    
    SUPPORTED_EXTENSIONS = (
        '.txt', '.md', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.csv',
        '.docx', '.doc', '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php',
        '.json', '.xml', '.yaml', '.yml', '.html', '.htm', '.css',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
    )
    
    TEXT_EXTENSIONS = (
        '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php',
        '.json', '.xml', '.yaml', '.yml', '.html', '.htm', '.css', '.csv',
    )
    
    def __init__(self, ocr_processor: OCRProcessor):
        self.ocr = ocr_processor
        self._parsers = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx, '.doc': self._parse_docx,
            '.xlsx': self._parse_excel, '.xls': self._parse_excel,
            '.pptx': self._parse_pptx, '.ppt': self._parse_pptx,
            '.png': self._parse_image, '.jpg': self._parse_image,
            '.jpeg': self._parse_image, '.tiff': self._parse_image,
            '.bmp': self._parse_image,
        }
    
    def parse(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[Parser] Reading: {os.path.basename(file_path)} ({ext})")
        
        parser = self._parsers.get(ext, self._parse_text)
        return parser(file_path)
    
    def _parse_pdf(self, file_path: str) -> str:
        if not HAS_PDF:
            return ""
        
        try:
            text_parts = []
            scanned_pages = []
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = (page.extract_text() or "").strip()
                        
                        if len(page_text) > 50:
                            text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                        else:
                            scanned_pages.append(page_num + 1)
                    except Exception:
                        scanned_pages.append(page_num + 1)
            
            text = "\n\n".join(text_parts)
            
            # OCR only if needed and available
            if not text.strip() and scanned_pages and self.ocr.available and self.ocr.pdf_support:
                text = self.ocr.ocr_pdf(file_path, dpi=200)  # Lower DPI for speed
            
            return text.strip()
        except Exception as e:
            print(f"[Parser] PDF error: {e}")
            return ""
    
    def _parse_docx(self, file_path: str) -> str:
        if not HAS_DOCX:
            return ""
        
        try:
            doc = Document(file_path)
            text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[Parser] DOCX error: {e}")
            return ""
    
    def _parse_excel(self, file_path: str) -> str:
        if not HAS_EXCEL:
            return ""
        
        try:
            text_parts = []
            wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) if c else "" for c in row)
                    if row_text.strip():
                        text_parts.append(row_text)
            
            wb.close()
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[Parser] Excel error: {e}")
            return ""
    
    def _parse_pptx(self, file_path: str) -> str:
        if not HAS_PPTX:
            return ""
        
        try:
            prs = Presentation(file_path)
            text_parts = []
            
            for i, slide in enumerate(prs.slides, 1):
                slide_text = [s.text for s in slide.shapes if hasattr(s, "text") and s.text]
                if slide_text:
                    text_parts.append(f"=== Slide {i} ===\n" + "\n".join(slide_text))
            
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"[Parser] PPTX error: {e}")
            return ""
    
    def _parse_image(self, file_path: str) -> str:
        if not self.ocr.available:
            return ""
        return self.ocr.ocr_image(file_path)
    
    def _parse_text(self, file_path: str) -> str:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception:
                return ""
        return ""


class RAG:
    """Optimized RAG with caching and lazy loading."""
    
    def __init__(self):
        self.documents = []
        self._embeddings = None
        self.embedding_model = EmbeddingModel()
        self.ocr_processor = OCRProcessor()
        self.parser = DocumentParser(self.ocr_processor)
        self.last_sources = []
        self._index_hash = None
        
        self.index_file = get_writable_path("index.json")
        self.embeddings_file = get_writable_path("embeddings.npy")
        self.hash_file = get_writable_path("index_hash.txt")
        self.user_docs_folder = get_writable_path("documents")
        self.bundled_data_folder = get_resource_path(config.RAG_FOLDER)
    
    @property
    def embeddings(self):
        if self._embeddings is None and os.path.exists(self.embeddings_file):
            try:
                self._embeddings = np.load(self.embeddings_file)
            except Exception:
                pass
        return self._embeddings
    
    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        os.makedirs(self.user_docs_folder, exist_ok=True)
        
        # Check OCR status
        ocr_status = self.ocr_processor.get_status()
        log("✓ OCR available" if ocr_status["ocr_available"] else "⚠ OCR not available")
        
        # Load embedding model
        log("Loading embedding model...")
        if not self.embedding_model.load(on_progress):
            log("⚠ Using keyword search (no embedding)")
        
        # Check if we can use cached index
        if self._is_cache_valid():
            log("Loading cached index...")
            self._load_index()
        else:
            log("Building new index...")
            self._build_index(log)
        
        log(f"RAG ready: {len(self.documents)} chunks")
        return True
    
    def _compute_docs_hash(self) -> str:
        """Compute hash of all documents for cache invalidation."""
        hashes = []
        
        for folder in [self.bundled_data_folder, self.user_docs_folder]:
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in DocumentParser.SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(folder, filename)
                    hashes.append(get_file_hash(filepath))
        
        return hashlib.md5("|".join(hashes).encode()).hexdigest()
    
    def _is_cache_valid(self) -> bool:
        """Check if cached index is still valid."""
        if not config.INDEX_CACHE_ENABLED:
            return False
        
        if not os.path.exists(self.index_file):
            return False
        
        if not os.path.exists(self.hash_file):
            return False
        
        try:
            with open(self.hash_file, 'r') as f:
                cached_hash = f.read().strip()
            return cached_hash == self._compute_docs_hash()
        except Exception:
            return False
    
    def _load_index(self):
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            # Lazy load embeddings - only when needed
            self._embeddings = None
        except Exception as e:
            print(f"[RAG] Load error: {e}")
            self.documents = []
            self._embeddings = None
    
    def _save_index(self, log):
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False)
            
            if self._embeddings is not None:
                np.save(self.embeddings_file, self._embeddings)
            
            # Save hash for cache validation
            with open(self.hash_file, 'w') as f:
                f.write(self._compute_docs_hash())
            
            log(f"Index saved: {len(self.documents)} chunks")
        except Exception as e:
            log(f"Save error: {e}")
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        overlap = getattr(config, 'RAG_CHUNK_OVERLAP', 50)
        text = text.strip()
        
        if len(text) < 50:
            return [text] if len(text) > 20 else []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk, current_length = [], [], 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) <= chunk_size:
                current_chunk.extend(words)
                current_length += len(words)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Overlap
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
        
        if current_chunk and len(current_chunk) >= 15:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _build_index(self, log):
        all_chunks = []
        
        folders = []
        if os.path.exists(self.bundled_data_folder):
            folders.append(("bundled", self.bundled_data_folder))
        if os.path.exists(self.user_docs_folder):
            folders.append(("user", self.user_docs_folder))
        
        for folder_type, folder in folders:
            log(f"Scanning {folder_type}: {folder}")
            
            if not os.path.isdir(folder):
                continue
            
            for filename in os.listdir(folder):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
                    continue
                
                file_path = os.path.join(folder, filename)
                
                try:
                    log(f"Processing: {filename}")
                    text = self.parser.parse(file_path)
                    
                    if not text or len(text.strip()) < 10:
                        log(f"⚠ {filename}: empty")
                        continue
                    
                    chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            "source": filename,
                            "chunk_id": i,
                            "content": chunk
                        })
                    
                    log(f"✓ {filename}: {len(chunks)} chunks")
                except Exception as e:
                    log(f"✗ {filename}: {e}")
        
        if not all_chunks:
            log("No documents found")
            self.documents = []
            self._embeddings = None
            return
        
        # Encode embeddings
        if self.embedding_model.is_loaded:
            log(f"Encoding {len(all_chunks)} chunks...")
            texts = [c["content"] for c in all_chunks]
            self._embeddings = self.embedding_model.encode(texts, is_query=False)
        else:
            self._embeddings = None
        
        self.documents = all_chunks
        self._save_index(log)
    
    def search(self, query: str, top_k: int = None) -> Tuple[str, List[dict]]:
        if not self.documents:
            self.last_sources = []
            return "", []
        
        top_k = top_k or config.RAG_TOP_K
        min_score = getattr(config, 'RAG_MIN_SCORE', 0.3)
        results = []
        
        # Semantic search if embeddings available
        if self.embeddings is not None and self.embedding_model.is_loaded:
            query_emb = self.embedding_model.encode([query], is_query=True)[0]
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append((self.documents[idx], score))
            results = results[:top_k]
        else:
            # Fallback: keyword search
            query_words = set(query.lower().split())
            scored = []
            for doc in self.documents:
                content_lower = doc["content"].lower()
                content_words = set(content_lower.split())
                matches = query_words & content_words
                score = len(matches) + sum(0.5 for w in query_words if w in content_lower)
                if score > 0:
                    scored.append((doc, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = scored[:top_k]
        
        if not results:
            self.last_sources = []
            return "", []
        
        self.last_sources = []
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            self.last_sources.append({
                "index": i,
                "source": doc["source"],
                "chunk_id": doc["chunk_id"],
                "score": score,
                "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            })
            context_parts.append(f"[Document {i} - {doc['source']}]\n{doc['content']}")
        
        return "\n\n".join(context_parts), self.last_sources
    
    def format_sources_for_display(self) -> str:
        if not self.last_sources:
            return ""
        
        lines = ["", "📚 Sources:"]
        for src in self.last_sources:
            lines.append(f"  [{src['index']}] {src['source']} (score: {src['score']:.2f})")
            preview = src['preview'][:80].replace('\n', ' ')
            lines.append(f"      \"{preview}...\"")
        
        return "\n".join(lines)
    
    def add_documents(self, file_paths: list, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            os.makedirs(self.user_docs_folder, exist_ok=True)
            added = 0
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
                    log(f"⚠ {filename}: unsupported")
                    continue
                
                dest = os.path.join(self.user_docs_folder, filename)
                shutil.copy2(file_path, dest)
                log(f"✓ Copied: {filename}")
                added += 1
            
            if added == 0:
                return False
            
            log("Rebuilding index...")
            self._build_index(log)
            
            return len(self.documents) > 0
        except Exception as e:
            log(f"Error: {e}")
            return False
    
    def clear_cache(self):
        """Clear all cached data."""
        for f in [self.index_file, self.embeddings_file, self.hash_file]:
            if os.path.exists(f):
                os.remove(f)
        self.documents = []
        self._embeddings = None
