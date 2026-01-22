#!/usr/bin/env python3
"""
Local Chat - All-in-one optimized version.
Privacy-focused AI document assistant that runs entirely offline.
"""

import sys
import os
import json
import re
import shutil
import hashlib
import glob
import multiprocessing
from datetime import datetime
from typing import Optional, List, Dict, Callable, Iterator, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache
from io import BytesIO

# Platform-specific setup
if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
if hasattr(sys, '_MEIPASS'):
    multiprocessing.freeze_support()

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration."""
    
    # App
    APP_NAME = "Local Chat"
    WINDOW_SIZE = "1100x700"
    
    # Model
    MODEL_FILE = "models/model.gguf"
    EMBEDDING_MODEL_FOLDER = "embedding_model"
    
    # Auto-detect optimal settings
    _CPU_COUNT = multiprocessing.cpu_count()
    CONTEXT_SIZE = 4096
    MAX_TOKENS = 512
    THREADS = max(4, _CPU_COUNT - 2)
    GPU_LAYERS = -1
    
    # Prompt
    SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have this information in the provided documents."
Be concise and specific. Quote relevant parts when possible.
Answer in the same language as the user."""
    
    STOP_TOKENS = ["<|im_end|>", "<end_of_turn>", "<|endoftext|>"]
    
    # RAG
    RAG_ENABLED = True
    RAG_FOLDER = "data"
    RAG_CHUNK_SIZE = 400
    RAG_CHUNK_OVERLAP = 50
    RAG_TOP_K = 3
    RAG_MIN_SCORE = 0.3
    RAG_SHOW_SOURCES = True
    
    # Sampling
    TEMPERATURE = 0.2
    TOP_P = 0.9
    REPEAT_PENALTY = 1.1
    
    # Performance
    BATCH_SIZE = 512
    INDEX_CACHE_ENABLED = True


config = Config()

# ============================================================================
# UTILITIES
# ============================================================================

@lru_cache(maxsize=32)
def get_resource_path(relative_path: str) -> str:
    """Gets path compatible with PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


_APP_DATA_DIR = None

def get_writable_path(filename: str) -> str:
    """Gets writable path for user data."""
    global _APP_DATA_DIR
    
    if _APP_DATA_DIR is None:
        if hasattr(sys, '_MEIPASS'):
            _APP_DATA_DIR = os.path.join(os.path.expanduser("~"), ".localchat")
        else:
            _APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", ".localchat")
        os.makedirs(_APP_DATA_DIR, exist_ok=True)
    
    return os.path.join(_APP_DATA_DIR, filename)


def get_file_hash(filepath: str) -> str:
    """Get a quick hash of file for cache invalidation."""
    stat = os.stat(filepath)
    content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================================
# OCR PROCESSOR
# ============================================================================

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class OCRProcessor:
    """Handles OCR for scanned documents and images."""
    
    def __init__(self):
        self.available = HAS_OCR
        self.pdf_support = HAS_PDF2IMAGE
        self.poppler_path = None
        self._lang_cache = None
        self._configure_tesseract()
        self._configure_poppler()
    
    def _configure_tesseract(self):
        if not HAS_OCR:
            return
        
        if sys.platform == 'win32':
            if hasattr(sys, '_MEIPASS'):
                tesseract_path = os.path.join(sys._MEIPASS, 'tesseract', 'tesseract.exe')
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    return
            
            standard_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            self.available = False
    
    def _configure_poppler(self):
        if not HAS_PDF2IMAGE or sys.platform != 'win32':
            return
        
        poppler_search_paths = [
            r"C:\Program Files\poppler-*\Library\bin",
            r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin",
            r"C:\ProgramData\chocolatey\bin",
        ]
        
        for pattern in poppler_search_paths:
            if '*' in pattern:
                matches = glob.glob(pattern)
                for match in matches:
                    if os.path.isdir(match):
                        self.poppler_path = match
                        return
            elif os.path.isdir(pattern):
                self.poppler_path = pattern
                return
    
    def get_status(self) -> dict:
        return {
            "ocr_available": self.available,
            "pdf_ocr_available": self.available and self.pdf_support,
        }
    
    def _get_best_language(self, preferred: str = 'eng') -> str:
        if not self.available:
            return 'eng'
        
        if self._lang_cache is None:
            try:
                self._lang_cache = [l for l in pytesseract.get_languages() if l != 'osd']
            except Exception:
                self._lang_cache = ['eng']
        
        preferred_list = preferred.split('+')
        valid_langs = [lang for lang in preferred_list if lang in self._lang_cache]
        
        if valid_langs:
            return '+'.join(valid_langs)
        return 'eng' if 'eng' in self._lang_cache else (self._lang_cache[0] if self._lang_cache else 'eng')
    
    def ocr_image(self, image_path: str, lang: str = 'eng') -> str:
        if not self.available:
            return ""
        
        try:
            image = Image.open(image_path)
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            max_dim = 2000
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            use_lang = self._get_best_language(lang)
            
            try:
                text = pytesseract.image_to_string(image, lang=use_lang)
            except Exception:
                text = pytesseract.image_to_string(image)
            
            return text.strip()
        except Exception as e:
            print(f"[OCR] Error: {e}")
            return ""
    
    def ocr_image_from_bytes(self, image_bytes: bytes, lang: str = 'eng') -> str:
        if not self.available:
            return ""
        
        try:
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            use_lang = self._get_best_language(lang)
            
            try:
                return pytesseract.image_to_string(image, lang=use_lang).strip()
            except Exception:
                return pytesseract.image_to_string(image).strip()
        except Exception:
            return ""
    
    def ocr_pdf(self, pdf_path: str, lang: str = 'eng', dpi: int = 200,
                on_progress: Optional[Callable[[str], None]] = None) -> str:
        if not self.available or not self.pdf_support:
            return ""
        
        def log(msg):
            print(f"[OCR] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            convert_kwargs = {'dpi': dpi}
            if sys.platform == 'win32' and self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path
            
            try:
                images = convert_from_path(pdf_path, **convert_kwargs)
            except Exception:
                convert_kwargs['dpi'] = 150
                try:
                    images = convert_from_path(pdf_path, **convert_kwargs)
                except Exception:
                    return ""
            
            if not images:
                return ""
            
            all_text = []
            use_lang = self._get_best_language(lang)
            
            for i, image in enumerate(images):
                log(f"OCR page {i + 1}/{len(images)}")
                
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                try:
                    text = pytesseract.image_to_string(image, lang=use_lang)
                except Exception:
                    text = pytesseract.image_to_string(image)
                
                if text.strip():
                    all_text.append(f"=== Page {i + 1} ===\n{text.strip()}")
            
            return "\n\n".join(all_text)
        except Exception as e:
            log(f"Error: {e}")
            return ""


# ============================================================================
# DOCUMENT PARSER
# ============================================================================

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


class DocumentParser:
    """Optimized document parser."""
    
    SUPPORTED_EXTENSIONS = (
        '.txt', '.md', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.csv',
        '.docx', '.doc', '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php',
        '.json', '.xml', '.yaml', '.yml', '.html', '.htm', '.css',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
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
            
            if not text.strip() and scanned_pages and self.ocr.available and self.ocr.pdf_support:
                text = self.ocr.ocr_pdf(file_path, dpi=200)
            
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


# ============================================================================
# EMBEDDING MODEL
# ============================================================================

class EmbeddingModel:
    """Lazy-loaded embedding model."""
    
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
        
        return self._model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            batch_size=config.BATCH_SIZE
        )


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAG:
    """RAG system with per-conversation document isolation."""
    
    def __init__(self):
        self.documents = []
        self._embeddings = None
        self.embedding_model = EmbeddingModel()
        self.ocr_processor = OCRProcessor()
        self.parser = DocumentParser(self.ocr_processor)
        self.last_sources = []
        self._current_conv_id = None
        
        # Global paths
        self.user_docs_folder = get_writable_path("documents")
        self.bundled_data_folder = get_resource_path(config.RAG_FOLDER)
    
    def _get_conv_index_file(self, conv_id: str) -> str:
        return get_writable_path(f"index_{conv_id}.json")
    
    def _get_conv_embeddings_file(self, conv_id: str) -> str:
        return get_writable_path(f"embeddings_{conv_id}.npy")
    
    def _get_conv_docs_folder(self, conv_id: str) -> str:
        folder = os.path.join(self.user_docs_folder, conv_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @property
    def embeddings(self):
        return self._embeddings
    
    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value
    
    def initialize(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Initialize RAG system (load embedding model only)."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not config.RAG_ENABLED:
            return True
        
        os.makedirs(self.user_docs_folder, exist_ok=True)
        
        ocr_status = self.ocr_processor.get_status()
        log("✓ OCR available" if ocr_status["ocr_available"] else "⚠ OCR not available")
        
        log("Loading embedding model...")
        if not self.embedding_model.load(on_progress):
            log("⚠ Using keyword search (no embedding)")
        
        # Don't load any documents at startup - wait for conversation selection
        self.documents = []
        self._embeddings = None
        
        log("RAG ready (no documents loaded)")
        return True
    
    def load_conversation_documents(self, conv_id: str, on_progress: Optional[Callable[[str], None]] = None):
        """Load documents for a specific conversation."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        self._current_conv_id = conv_id
        
        # Check if we have cached index for this conversation
        index_file = self._get_conv_index_file(conv_id)
        embeddings_file = self._get_conv_embeddings_file(conv_id)
        
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                
                if os.path.exists(embeddings_file) and self.embedding_model.is_loaded:
                    self._embeddings = np.load(embeddings_file)
                else:
                    self._embeddings = None
                
                log(f"Loaded {len(self.documents)} chunks for conversation")
                return
            except Exception as e:
                log(f"Error loading index: {e}")
        
        # No cached index - start fresh
        self.documents = []
        self._embeddings = None
        log("No documents for this conversation")
    
    def _save_index(self, log):
        """Save index for current conversation."""
        if not self._current_conv_id:
            return
        
        try:
            index_file = self._get_conv_index_file(self._current_conv_id)
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False)
            
            if self._embeddings is not None:
                embeddings_file = self._get_conv_embeddings_file(self._current_conv_id)
                np.save(embeddings_file, self._embeddings)
            
            log(f"Index saved: {len(self.documents)} chunks")
        except Exception as e:
            log(f"Save error: {e}")
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        overlap = config.RAG_CHUNK_OVERLAP
        text = text.strip()
        
        if len(text) < 50:
            return [text] if len(text) > 20 else []
        
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
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
        
        if current_chunk and len(current_chunk) >= 15:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def search(self, query: str, top_k: int = None) -> Tuple[str, List[dict]]:
        if not self.documents:
            self.last_sources = []
            return "", []
        
        top_k = top_k or config.RAG_TOP_K
        min_score = config.RAG_MIN_SCORE
        results = []
        
        if self._embeddings is not None and self.embedding_model.is_loaded:
            query_emb = self.embedding_model.encode([query], is_query=True)[0]
            similarities = np.dot(self._embeddings, query_emb)
            top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append((self.documents[idx], score))
            results = results[:top_k]
        else:
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
        """Add documents to current conversation."""
        def log(msg):
            print(f"[RAG] {msg}")
            if on_progress:
                on_progress(msg)
        
        if not self._current_conv_id:
            log("No conversation selected")
            return False
        
        try:
            docs_folder = self._get_conv_docs_folder(self._current_conv_id)
            added = 0
            new_chunks = []
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
                    log(f"⚠ {filename}: unsupported")
                    continue
                
                # Copy file to conversation folder
                dest = os.path.join(docs_folder, filename)
                shutil.copy2(file_path, dest)
                log(f"✓ Copied: {filename}")
                
                # Parse and chunk
                text = self.parser.parse(dest)
                if not text or len(text.strip()) < 10:
                    log(f"⚠ {filename}: empty")
                    continue
                
                chunks = self._split_text(text, config.RAG_CHUNK_SIZE)
                
                for i, chunk in enumerate(chunks):
                    new_chunks.append({
                        "source": filename,
                        "chunk_id": i,
                        "content": chunk
                    })
                
                log(f"✓ {filename}: {len(chunks)} chunks")
                added += 1
            
            if added == 0:
                return False
            
            # Add to existing documents
            self.documents.extend(new_chunks)
            
            # Re-encode all embeddings
            if self.embedding_model.is_loaded:
                log(f"Encoding {len(self.documents)} chunks...")
                texts = [c["content"] for c in self.documents]
                self._embeddings = self.embedding_model.encode(texts, is_query=False)
            
            self._save_index(log)
            
            return len(self.documents) > 0
        except Exception as e:
            log(f"Error: {e}")
            return False
    
    def clear_conversation_documents(self, conv_id: str):
        """Clear all documents for a conversation."""
        # Delete index files
        index_file = self._get_conv_index_file(conv_id)
        embeddings_file = self._get_conv_embeddings_file(conv_id)
        
        for f in [index_file, embeddings_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        
        # Delete documents folder
        docs_folder = os.path.join(self.user_docs_folder, conv_id)
        if os.path.exists(docs_folder):
            try:
                shutil.rmtree(docs_folder)
            except Exception:
                pass
        
        # Clear current state if this is the current conversation
        if self._current_conv_id == conv_id:
            self.documents = []
            self._embeddings = None


# ============================================================================
# LLM ENGINE
# ============================================================================

class LLMEngine:
    """LLM Engine for text generation."""
    
    def __init__(self):
        self.llm = None
        self.history = []
        self.rag = RAG()
        self.is_ready = False
        self.error = None
        self._max_history = 3
    
    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        def log(msg):
            print(f"[LLM] {msg}")
            if on_progress:
                on_progress(msg)
        
        try:
            self.rag.initialize(log)
            
            log("Importing llama_cpp...")
            from llama_cpp import Llama
            
            model_path = get_resource_path(config.MODEL_FILE)
            log(f"Model: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            log("Loading model...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=config.CONTEXT_SIZE,
                n_threads=config.THREADS,
                n_gpu_layers=config.GPU_LAYERS,
                n_batch=512,
                use_mmap=True,
                use_mlock=False,
                verbose=False
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
    
    def _build_prompt(self, message: str, rag_context: str = "") -> str:
        if rag_context:
            system = f"""{config.SYSTEM_PROMPT}

=== CONTEXT ===
{rag_context}
=== END ===

Answer based on context. If not found, say so."""
        else:
            system = config.SYSTEM_PROMPT
        
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        for h in self.history[-self._max_history:]:
            prompt += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def generate(self, message: str) -> Iterator[str]:
        if not self.is_ready:
            yield "Error: Model not ready"
            return
        
        rag_context = ""
        sources = []
        
        if config.RAG_ENABLED and self.rag.documents:
            rag_context, sources = self.rag.search(message)
        
        prompt = self._build_prompt(message, rag_context)
        
        full_response = ""
        
        try:
            for chunk in self.llm(
                prompt,
                max_tokens=config.MAX_TOKENS,
                stop=config.STOP_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repeat_penalty=config.REPEAT_PENALTY,
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                full_response += token
                yield token
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            yield f"\n[Error: {e}]"
        
        if config.RAG_SHOW_SOURCES and sources:
            sources_text = self.rag.format_sources_for_display()
            yield sources_text
            full_response += sources_text
        
        if full_response.strip():
            clean = full_response.split("📚 Sources:")[0].strip()
            self.history.append({"user": message, "assistant": clean})
            
            if len(self.history) > self._max_history * 2:
                self.history = self.history[-self._max_history:]
    
    def clear_history(self):
        self.history = []
        print("[LLM] History cleared")


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

@dataclass
class Conversation:
    """Represents a single conversation."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, str]]
    document_ids: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Conversation':
        return cls(**data)


class ConversationManager:
    """Manages multiple conversations."""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.current_id: Optional[str] = None
        
        self.data_dir = get_writable_path("conversations")
        self.index_file = os.path.join(self.data_dir, "index.json")
        self.docs_dir = get_writable_path("documents")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        
        self._load_index()
    
    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.conversations = {
                    cid: Conversation.from_dict(conv) 
                    for cid, conv in data.get("conversations", {}).items()
                }
                self.current_id = data.get("current_id")
                
                if self.current_id and self.current_id not in self.conversations:
                    self.current_id = None
                    
            except Exception as e:
                print(f"[ConvManager] Error loading index: {e}")
                self.conversations = {}
                self.current_id = None
    
    def _save_index(self):
        try:
            data = {
                "conversations": {cid: conv.to_dict() for cid, conv in self.conversations.items()},
                "current_id": self.current_id
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConvManager] Error saving index: {e}")
    
    def _generate_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _generate_title(self, first_message: str = "") -> str:
        if first_message:
            title = first_message[:40].strip()
            if len(first_message) > 40:
                title += "..."
            return title
        return f"New Chat {datetime.now().strftime('%d/%m %H:%M')}"
    
    def create_conversation(self, title: str = "") -> Conversation:
        conv_id = self._generate_id()
        now = datetime.now().isoformat()
        
        conv = Conversation(
            id=conv_id,
            title=title or self._generate_title(),
            created_at=now,
            updated_at=now,
            messages=[],
            document_ids=[]
        )
        
        self.conversations[conv_id] = conv
        self.current_id = conv_id
        self._save_index()
        
        print(f"[ConvManager] Created conversation: {conv_id}")
        return conv
    
    def get_current(self) -> Optional[Conversation]:
        if self.current_id and self.current_id in self.conversations:
            return self.conversations[self.current_id]
        return None
    
    def set_current(self, conv_id: str) -> Optional[Conversation]:
        if conv_id in self.conversations:
            self.current_id = conv_id
            self._save_index()
            return self.conversations[conv_id]
        return None
    
    def get_all(self) -> List[Conversation]:
        return sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
    
    def add_message(self, role: str, content: str):
        conv = self.get_current()
        if not conv:
            conv = self.create_conversation()
        
        conv.messages.append({"role": role, "content": content})
        conv.updated_at = datetime.now().isoformat()
        
        if role == "user" and len(conv.messages) == 1:
            conv.title = self._generate_title(content)
        
        self._save_index()
    
    def add_document(self, filename: str):
        conv = self.get_current()
        if not conv:
            conv = self.create_conversation()
        
        if filename not in conv.document_ids:
            conv.document_ids.append(filename)
            conv.updated_at = datetime.now().isoformat()
            self._save_index()
    
    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id not in self.conversations:
            return False
        
        del self.conversations[conv_id]
        
        if self.current_id == conv_id:
            remaining = self.get_all()
            self.current_id = remaining[0].id if remaining else None
        
        self._save_index()
        print(f"[ConvManager] Deleted conversation: {conv_id}")
        return True
    
    def clear_history(self):
        conv = self.get_current()
        if conv:
            conv.messages = []
            conv.updated_at = datetime.now().isoformat()
            self._save_index()


# ============================================================================
# USER INTERFACE
# ============================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ConversationItem(ctk.CTkFrame):
    def __init__(self, parent, conv_id: str, title: str, doc_count: int,
                 is_active: bool, on_select: Callable, on_delete: Callable):
        super().__init__(parent, corner_radius=8, height=50)
        
        self.conv_id = conv_id
        self.on_select = on_select
        self.on_delete = on_delete
        
        if is_active:
            self.configure(fg_color=("#3b7ac7", "#2d5f9e"))
        else:
            self.configure(fg_color=("gray75", "gray25"))
        
        self.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        title_label = ctk.CTkLabel(
            self,
            text=title[:25] + "..." if len(title) > 25 else title,
            font=ctk.CTkFont(size=12, weight="bold" if is_active else "normal"),
            anchor="w"
        )
        title_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)
        title_label.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        if doc_count > 0:
            doc_badge = ctk.CTkLabel(
                self, text=f"📄{doc_count}",
                font=ctk.CTkFont(size=10),
                text_color=("#50fa7b", "#40c969")
            )
            doc_badge.pack(side="left", padx=(0, 5))
            doc_badge.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        delete_btn = ctk.CTkButton(
            self, text="✕", width=24, height=24, corner_radius=4,
            fg_color="transparent", hover_color=("#ff5555", "#cc4444"),
            font=ctk.CTkFont(size=12),
            command=lambda: self.on_delete(self.conv_id)
        )
        delete_btn.pack(side="right", padx=5)


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, on_new: Callable, on_select: Callable, on_delete: Callable):
        super().__init__(parent, width=250, corner_radius=0)
        
        self.on_new = on_new
        self.on_select = on_select
        self.on_delete = on_delete
        
        self.pack_propagate(False)
        self._create_widgets()
    
    def _create_widgets(self):
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="💬 Chats",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", pady=10)
        
        ctk.CTkButton(
            header, text="+ New", width=70, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#50fa7b", "#40c969"), hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000"), command=self.on_new
        ).pack(side="right", pady=10)
        
        ctk.CTkFrame(self, height=2, fg_color=("gray70", "gray30")).pack(fill="x", padx=10, pady=(0, 10))
        
        self.conv_list = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=("#4a9eff", "#3b7ac7")
        )
        self.conv_list.pack(fill="both", expand=True, padx=5, pady=5)
    
    def update_conversations(self, conversations: list, current_id: str):
        for widget in self.conv_list.winfo_children():
            widget.destroy()
        
        if not conversations:
            ctk.CTkLabel(
                self.conv_list, text="No conversations yet.\nClick '+ New' to start!",
                font=ctk.CTkFont(size=12), text_color=("gray50", "gray50")
            ).pack(pady=20)
            return
        
        for conv in conversations:
            item = ConversationItem(
                self.conv_list, conv_id=conv.id, title=conv.title,
                doc_count=len(conv.document_ids), is_active=(conv.id == current_id),
                on_select=self.on_select, on_delete=self._confirm_delete
            )
            item.pack(fill="x", pady=2)
    
    def _confirm_delete(self, conv_id: str):
        if messagebox.askyesno("Delete Chat", "Delete this conversation?"):
            self.on_delete(conv_id)


class ChatUI:
    def __init__(self, root: ctk.CTk, on_send, on_clear, on_load_files,
                 on_new_chat, on_select_chat, on_delete_chat):
        self.root = root
        self.on_send = on_send
        self.on_clear = on_clear
        self.on_load_files = on_load_files
        self.on_new_chat = on_new_chat
        self.on_select_chat = on_select_chat
        self.on_delete_chat = on_delete_chat
        
        self._setup_window()
        self._create_widgets()
    
    def _setup_window(self):
        self.root.title(f"🤖 {config.APP_NAME}")
        self.root.geometry(config.WINDOW_SIZE)
        self.root.minsize(900, 600)
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
    
    def _create_widgets(self):
        self.sidebar = Sidebar(
            self.root, on_new=self.on_new_chat,
            on_select=self.on_select_chat, on_delete=self.on_delete_chat
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        header = ctk.CTkFrame(main_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(header, text=f"🤖 {config.APP_NAME}",
                     font=ctk.CTkFont(family="Arial", size=24, weight="bold")).pack(side="left")
        
        self.status = ctk.CTkLabel(header, text="⏳ Loading...",
                                   font=ctk.CTkFont(size=12), text_color=("#888888", "#888888"))
        self.status.pack(side="right", padx=10)
        
        toolbar = ctk.CTkFrame(main_frame, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        ctk.CTkButton(
            toolbar, text="📁 Load Files", command=self._load_files,
            width=120, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#4a9eff", "#3b7ac7"), hover_color=("#3b7ac7", "#2d5f9e")
        ).pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(
            toolbar, text="🗑️ Clear", command=self.on_clear,
            width=100, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#ff5555", "#cc4444"), hover_color=("#ff7777", "#dd5555")
        ).pack(side="left")
        
        self.doc_info = ctk.CTkLabel(toolbar, text="📚 No documents",
                                     font=ctk.CTkFont(size=11), text_color=("#888888", "#888888"))
        self.doc_info.pack(side="right", padx=10)
        
        chat_container = ctk.CTkFrame(main_frame, corner_radius=10)
        chat_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)
        
        self.chat = ctk.CTkTextbox(
            chat_container, wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12), corner_radius=10,
            fg_color=("#1e1e2e", "#16213e"),
            scrollbar_button_color=("#4a9eff", "#3b7ac7")
        )
        self.chat.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input = ctk.CTkTextbox(
            input_frame, height=60,
            font=ctk.CTkFont(family="Consolas", size=12), corner_radius=10,
            border_width=2, border_color=("#4a9eff", "#3b7ac7"),
            fg_color=("#1e1e2e", "#16213e")
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)
        
        ctk.CTkButton(
            input_frame, text="Send ➤", command=self._send,
            width=90, height=60, corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#50fa7b", "#40c969"), hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000")
        ).grid(row=0, column=1)
    
    def _on_enter(self, event):
        if not (event.state & 0x1):
            self._send()
            return "break"
    
    def _send(self):
        text = self.input.get("0.0", "end").strip()
        if text:
            self.input.delete("0.0", "end")
            self.on_send(text)
    
    def _load_files(self):
        files = filedialog.askopenfilenames(
            parent=self.root, title="Select documents",
            filetypes=[
                ("All supported", "*.txt *.md *.pdf *.xlsx *.xls *.pptx *.ppt *.docx *.doc "
                 "*.py *.js *.json *.csv *.xml *.yaml *.yml *.html *.css "
                 "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("Documents", "*.txt *.md *.pdf *.docx *.doc"),
                ("Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Presentations", "*.pptx *.ppt"),
                ("Images (OCR)", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.on_load_files(files)
    
    def set_status(self, text: str, is_error: bool = False):
        color = ("#ff5555", "#ff5555") if is_error else ("#50fa7b", "#50fa7b")
        self.status.configure(text=text, text_color=color)
    
    def update_doc_count(self, count: int):
        if count == 0:
            text, color = "📚 No documents", ("#888888", "#888888")
        else:
            text, color = f"📚 {count} doc{'s' if count > 1 else ''}", ("#50fa7b", "#50fa7b")
        self.doc_info.configure(text=text, text_color=color)
    
    def add_message(self, sender: str, text: str, tag: str = ""):
        if self.chat.get("0.0", "end").strip():
            self.chat.insert("end", "\n")
        self.chat.insert("end", f"{sender}:\n{text}\n")
        self.chat.see("end")
    
    def stream(self, text: str):
        self.chat.insert("end", text)
        self.chat.see("end")
    
    def clear_chat(self):
        self.chat.delete("0.0", "end")
    
    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.input.configure(state=state)
    
    def focus_input(self):
        self.input.focus_set()
    
    def update_sidebar(self, conversations: list, current_id: str):
        self.sidebar.update_conversations(conversations, current_id)
    
    def load_messages(self, messages: list):
        self.clear_chat()
        for msg in messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            self.add_message(role, msg["content"], msg["role"])


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class App:
    def __init__(self):
        self.root = ctk.CTk()
        self.engine = LLMEngine()
        self.conv_manager = ConversationManager()
        self.generating = False
        
        self.ui = ChatUI(
            self.root,
            on_send=self.send,
            on_clear=self.clear,
            on_load_files=self.load_files,
            on_new_chat=self.new_chat,
            on_select_chat=self.select_chat,
            on_delete_chat=self.delete_chat
        )
        
        self.ui.add_message("System", "🚀 Starting...", "system")
        self._load_model()
    
    def _load_model(self):
        def task():
            success = self.engine.load(
                on_progress=lambda m: self.root.after(0, lambda: self.ui.add_message("Debug", m, "system"))
            )
            if success:
                self.root.after(0, self._on_ready)
            else:
                self.root.after(0, lambda: self.ui.set_status("❌ Failed", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _on_ready(self):
        self.ui.set_status("✅ Ready!")
        
        if not self.conv_manager.get_all():
            self.conv_manager.create_conversation("Welcome Chat")
        
        self._load_current_conversation()
        self._update_sidebar()
        
        self.ui.add_message("System", "✨ Ready! Start chatting or load documents.", "system")
    
    def _update_sidebar(self):
        convs = self.conv_manager.get_all()
        current = self.conv_manager.get_current()
        current_id = current.id if current else None
        self.ui.update_sidebar(convs, current_id)
    
    def _load_current_conversation(self):
        conv = self.conv_manager.get_current()
        
        if not conv:
            self.ui.clear_chat()
            self.ui.update_doc_count(0)
            self.engine.rag.documents = []
            self.engine.rag._embeddings = None
            return
        
        # Load conversation's documents into RAG
        self.engine.rag.load_conversation_documents(conv.id)
        
        self.ui.load_messages(conv.messages)
        self.ui.update_doc_count(len(conv.document_ids))
        
        # Clear LLM history when switching conversations
        self.engine.clear_history()
    
    def new_chat(self):
        conv = self.conv_manager.create_conversation()
        self._load_current_conversation()
        self._update_sidebar()
        self.ui.add_message("System", "💬 New conversation!", "system")
    
    def select_chat(self, conv_id: str):
        self.conv_manager.set_current(conv_id)
        self._load_current_conversation()
        self._update_sidebar()
    
    def delete_chat(self, conv_id: str):
        # Clear RAG documents for this conversation
        self.engine.rag.clear_conversation_documents(conv_id)
        
        self.conv_manager.delete_conversation(conv_id)
        self._load_current_conversation()
        self._update_sidebar()
    
    def send(self, message: str):
        if self.generating or not self.engine.is_ready:
            return
        
        if not self.conv_manager.get_current():
            self.conv_manager.create_conversation()
            self._update_sidebar()
        
        self.ui.add_message("You", message, "user")
        self.conv_manager.add_message("user", message)
        
        self.generating = True
        self.ui.set_enabled(False)
        self.ui.set_status("🤔 Thinking...")
        
        threading.Thread(target=self._generate, args=(message,), daemon=True).start()
    
    def _generate(self, message: str):
        self.root.after(0, lambda: self.ui.add_message("Assistant", "", "bot"))
        
        full_response = ""
        try:
            for token in self.engine.generate(message):
                full_response += token
                self.root.after(0, lambda t=token: self.ui.stream(t))
        except Exception as e:
            self.root.after(0, lambda: self.ui.add_message("Error", str(e), "error"))
        finally:
            clean_response = full_response.split("📚 Sources:")[0].strip()
            if clean_response:
                self.conv_manager.add_message("assistant", clean_response)
            
            self.root.after(0, self._done)
    
    def _done(self):
        self.generating = False
        self.ui.set_enabled(True)
        self.ui.set_status("✅ Ready!")
        self.ui.focus_input()
        self._update_sidebar()
    
    def clear(self):
        self.conv_manager.clear_history()
        self.engine.clear_history()
        self.ui.clear_chat()
        self.ui.add_message("System", "💬 Messages cleared.", "system")
    
    def load_files(self, files: tuple):
        conv = self.conv_manager.get_current()
        if not conv:
            self.conv_manager.create_conversation()
            conv = self.conv_manager.get_current()
            self._update_sidebar()
        
        # Set current conversation ID in RAG
        self.engine.rag._current_conv_id = conv.id
        
        self.ui.add_message("System", f"📥 Loading {len(files)} file(s)...", "system")
        self.ui.set_status("⏳ Loading...")
        
        def task():
            success = self.engine.rag.add_documents(
                files,
                on_progress=lambda m: self.root.after(0, lambda msg=m: self.ui.add_message("Debug", msg, "system"))
            )
            
            if success:
                for f in files:
                    self.conv_manager.add_document(os.path.basename(f))
                
                conv = self.conv_manager.get_current()
                self.root.after(0, lambda: self.ui.update_doc_count(len(conv.document_ids) if conv else 0))
                self.root.after(0, lambda: self.ui.add_message("System", f"✅ Loaded!", "system"))
                self.root.after(0, lambda: self.ui.set_status("✅ Ready!"))
                self.root.after(0, self._update_sidebar)
            else:
                self.root.after(0, lambda: self.ui.add_message("Error", "❌ Failed", "error"))
                self.root.after(0, lambda: self.ui.set_status("❌ Failed", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def run(self):
        self.root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    App().run()
