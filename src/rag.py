"""
RAG (Retrieval-Augmented Generation) system with document parsing and embeddings.
"""

import os
import json
import re
import numpy as np
from typing import Optional, Callable, List, Tuple

import config
from utils import get_resource_path, get_writable_path
from ocr import OCRProcessor

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

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class EmbeddingModel:
    """Embedding model using sentence-transformers."""
    
    def __init__(self):
        self.model = None
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Loads the embedding model from bundled files."""
        def log(msg):
            print(f"[Embedding] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            bundled_model_path = get_resource_path(config.EMBEDDING_MODEL_FOLDER)
            
            if os.path.exists(bundled_model_path):
                log(f"Loading: {bundled_model_path}")
                self.model = SentenceTransformer(bundled_model_path)
                log("Embedding model loaded!")
                return True
            else:
                log(f"Model not found: {bundled_model_path}")
                return False
                
        except Exception as e:
            log(f"Error: {e}")
            return False
    
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encodes texts to vectors."""
        if self.model is None:
            return np.array([])
        
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class DocumentParser:
    """Handles parsing of various document formats."""
    
    SUPPORTED_EXTENSIONS = (
        '.txt', '.md', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.csv',
        '.docx', '.doc',
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.toml',
        '.sh', '.bash', '.zsh', '.ps1', '.bat', '.sql',
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp',  # Images (OCR)
    )
    
    def __init__(self, ocr_processor: OCRProcessor):
        self.ocr = ocr_processor
    
    def parse(self, file_path: str) -> str:
        """Parse document and return extracted text."""
        ext = os.path.splitext(file_path)[1].lower()
        
        print(f"[Parser] Reading: {file_path} ({ext})")
        
        parsers = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.doc': self._parse_docx,
            '.xlsx': self._parse_excel,
            '.xls': self._parse_excel,
            '.pptx': self._parse_pptx,
            '.ppt': self._parse_pptx,
            '.png': self._parse_image,
            '.jpg': self._parse_image,
            '.jpeg': self._parse_image,
            '.tiff': self._parse_image,
            '.bmp': self._parse_image,
        }
        
        parser = parsers.get(ext, self._parse_text)
        return parser(file_path)
    
    def _parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF with OCR fallback."""
        if not HAS_PDF:
            return ""
        
        try:
            text_parts = []
            scanned_pages = []
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    
                    if len(page_text.strip()) > 50:
                        text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                    else:
                        scanned_pages.append(page_num + 1)
            
            text = "\n\n".join(text_parts)
            
            # OCR for scanned pages
            if scanned_pages and self.ocr.available and self.ocr.pdf_support:
                print(f"[Parser] {len(scanned_pages)} scanned pages detected, running OCR...")
                ocr_text = self.ocr.ocr_pdf(file_path)
                if ocr_text:
                    text += f"\n\n=== OCR Content ===\n{ocr_text}"
            
            return text.strip()
        except Exception as e:
            print(f"[Parser] PDF error: {e}")
            return ""
    
    def _parse_docx(self, file_path: str) -> str:
        """Extract text from Word documents."""
        if not HAS_DOCX:
            return ""
        
        try:
            doc = Document(file_path)
            text_parts = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)
            
            text = "\n".join(text_parts)
            
            # OCR images in document
            if self.ocr.available:
                ocr_text = self.ocr.ocr_docx_images(file_path)
                if ocr_text:
                    text += f"\n\n=== Image Content (OCR) ===\n{ocr_text}"
            
            return text
        except Exception as e:
            print(f"[Parser] DOCX error: {e}")
            return ""
    
    def _parse_excel(self, file_path: str) -> str:
        """Extract text from Excel files."""
        if not HAS_EXCEL:
            return ""
        
        try:
            text_parts = []
            wb = openpyxl.load_workbook(file_path, data_only=True)
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                    if row_text.strip():
                        text_parts.append(row_text)
            
            wb.close()
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[Parser] Excel error: {e}")
            return ""
    
    def _parse_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint files."""
        if not HAS_PPTX:
            return ""
        
        try:
            prs = Presentation(file_path)
            text_parts = []
            
            for i, slide in enumerate(prs.slides, 1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_text.append(shape.text)
                
                if slide_text:
                    text_parts.append(f"=== Slide {i} ===\n" + "\n".join(slide_text))
            
            text = "\n\n".join(text_parts)
            
            # OCR images in slides
            if self.ocr.available:
                ocr_text = self.ocr.ocr_pptx_images(file_path)
                if ocr_text:
                    text += f"\n\n=== Image Content (OCR) ===\n{ocr_text}"
            
            return text
        except Exception as e:
            print(f"[Parser] PPTX error: {e}")
            return ""
    
    def _parse_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        if not self.ocr.available:
            print(f"[Parser] OCR not available for: {file_path}")
            return ""
        
        return self.ocr.ocr_image(file_path)
    
    def _parse_text(self, file_path: str) -> str:
        """Parse plain text files."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"[Parser] Read {len(content)} chars")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[Parser] Error: {e}")
                return ""
        
        return ""


class RAG:
    """RAG system with semantic search and source tracking."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = EmbeddingModel()
        self.ocr_processor = OCRProcessor()
        self.parser = DocumentParser(self.ocr_processor)
        self.last_sources = []
        
        # Paths
        self.index_file = get_writable_path("index.json")
        self.embeddings_file = get_writable_path("embeddings.npy")
        self.user_docs_folder = get_writable_path("documents")
        self.bundled_data_folder = get_resource_path(config.RAG_FOLDER)
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Initialize RAG system."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        os.makedirs(self.user_docs_folder, exist_ok=True)
        log(f"Documents folder: {self.user_docs_folder}")
        
        # Report OCR status
        ocr_status = self.ocr_processor.get_status()
        if ocr_status["ocr_available"]:
            log("✓ OCR available (Tesseract)")
        else:
            log("⚠ OCR not available")
        
        # Load embedding model
        log("Loading embedding model...")
        if not self.embedding_model.load(on_progress):
            log("Using keyword search (no embeddings)")
        
        # Load or build index
        if os.path.exists(self.index_file):
            log("Loading existing index...")
            self._load_index()
            if not self.documents:
                log("Index empty, rebuilding...")
                self._build_index(log)
        else:
            log("Building index...")
            self._build_index(log)
        
        log(f"RAG ready: {len(self.documents)} chunks")
        return True
    
    def _load_index(self):
        """Load pre-built index."""
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            if os.path.exists(self.embeddings_file) and self.embedding_model.model:
                self.embeddings = np.load(self.embeddings_file)
            else:
                self.embeddings = None
        except Exception as e:
            print(f"[RAG] Load error: {e}")
            self.documents = []
            self.embeddings = None
    
    def _save_index(self, log: Callable[[str], None]):
        """Save index to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            log(f"Index saved: {len(self.documents)} chunks")
        except Exception as e:
            log(f"Save error: {e}")
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into overlapping chunks."""
        overlap = getattr(config, 'RAG_CHUNK_OVERLAP', 100)
        text = text.strip()
        
        if len(text) < 50:
            return [text] if len(text) > 20 else []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            
            if current_length + len(words) <= chunk_size:
                current_chunk.extend(words)
                current_length += len(words)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                if overlap > 0 and current_chunk:
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words + words
                else:
                    current_chunk = words
                current_length = len(current_chunk)
        
        if current_chunk and len(current_chunk) >= 20:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _build_index(self, log: Callable[[str], None]):
        """Build index from all documents."""
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
            self.embeddings = None
            return
        
        # Encode if embeddings available
        if self.embedding_model.model:
            log(f"Encoding {len(all_chunks)} chunks...")
            texts = [c["content"] for c in all_chunks]
            self.embeddings = self.embedding_model.encode(texts, is_query=False)
        else:
            self.embeddings = None
        
        self.documents = all_chunks
        self._save_index(log)
    
    def search(self, query: str, top_k: int = None) -> Tuple[str, List[dict]]:
        """Search documents."""
        if not self.documents:
            self.last_sources = []
            return "", []
        
        top_k = top_k or config.RAG_TOP_K
        min_score = getattr(config, 'RAG_MIN_SCORE', 0.25)
        
        results = []
        
        if self.embeddings is not None and self.embedding_model.model:
            # Semantic search
            query_emb = self.embedding_model.encode([query], is_query=True)[0]
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append((self.documents[idx], score))
            
            results = results[:top_k]
        else:
            # Keyword search
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
        
        # Format results
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
    
    def get_last_sources(self) -> List[dict]:
        """Get sources from last search."""
        return self.last_sources
    
    def format_sources_for_display(self) -> str:
        """Format sources for UI."""
        if not self.last_sources:
            return ""
        
        lines = ["", "📚 Sources:"]
        for src in self.last_sources:
            lines.append(f"  [{src['index']}] {src['source']} (score: {src['score']:.2f})")
            preview = src['preview'][:100].replace('\n', ' ')
            lines.append(f"      \"{preview}...\"")
        
        return "\n".join(lines)
    
    def add_documents(self, file_paths: list, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Add new documents and rebuild index."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            os.makedirs(self.user_docs_folder, exist_ok=True)
            
            import shutil
            added = 0
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
                    log(f"⚠ {filename}: unsupported")
                    continue
                
                dest = os.path.join(self.user_docs_folder, filename)
                shutil.copy2(file_path, dest)
                log(f"✓ Added: {filename}")
                added += 1
            
            if added == 0:
                return False
            
            log("Rebuilding index...")
            self._build_index(log)
            
            return len(self.documents) > 0
        except Exception as e:
            log(f"Error: {e}")
            return False
