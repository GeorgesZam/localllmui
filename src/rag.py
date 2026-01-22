"""
RAG (Retrieval-Augmented Generation) system with document parsing and embeddings.
"""

import os
import json
import re
import shutil
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
            import traceback
            traceback.print_exc()
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
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
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
        """Extract text from PDF with OCR fallback for scanned pages."""
        if not HAS_PDF:
            print("[Parser] PyPDF2 not available")
            return ""
        
        try:
            text_parts = []
            scanned_pages = []
            total_pages = 0
            
            print(f"[Parser] Opening PDF: {file_path}")
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                print(f"[Parser] PDF has {total_pages} pages")
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        page_text = page_text.strip()
                        
                        print(f"[Parser] Page {page_num + 1}: {len(page_text)} chars extracted")
                        
                        if len(page_text) > 50:
                            text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                        else:
                            scanned_pages.append(page_num + 1)
                            print(f"[Parser] Page {page_num + 1} appears to be scanned")
                    except Exception as e:
                        print(f"[Parser] Error on page {page_num + 1}: {e}")
                        scanned_pages.append(page_num + 1)
            
            text = "\n\n".join(text_parts)
            print(f"[Parser] Extracted {len(text)} chars from {len(text_parts)} text pages")
            
            # OCR for scanned pages
            if scanned_pages:
                print(f"[Parser] {len(scanned_pages)} pages appear scanned: {scanned_pages}")
                
                if self.ocr.available and self.ocr.pdf_support:
                    print("[Parser] Running OCR on PDF...")
                    ocr_text = self.ocr.ocr_pdf(file_path)
                    if ocr_text:
                        print(f"[Parser] OCR extracted {len(ocr_text)} chars")
                        if text:
                            text += f"\n\n=== OCR Content ===\n{ocr_text}"
                        else:
                            text = ocr_text
                    else:
                        print("[Parser] OCR returned no text")
                else:
                    print(f"[Parser] OCR not available (available={self.ocr.available}, pdf_support={self.ocr.pdf_support})")
            
            # Full OCR if no text at all
            if not text.strip() and total_pages > 0:
                print("[Parser] No text extracted, attempting full OCR...")
                if self.ocr.available and self.ocr.pdf_support:
                    text = self.ocr.ocr_pdf(file_path)
            
            return text.strip()
            
        except Exception as e:
            print(f"[Parser] PDF error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _parse_docx(self, file_path: str) -> str:
        """Extract text from Word documents."""
        if not HAS_DOCX:
            print("[Parser] python-docx not available")
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
            print("[Parser] openpyxl not available")
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
            print("[Parser] python-pptx not available")
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
        
        print(f"[Parser] Running OCR on image: {file_path}")
        return self.ocr.ocr_image(file_path)
    
    def _parse_text(self, file_path: str) -> str:
        """Parse plain text files."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"[Parser] Read {len(content)} chars with {encoding}")
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
            if ocr_status.get("pdf_ocr_available"):
                log("✓ PDF OCR available (Poppler)")
            else:
                log("⚠ PDF OCR not available (Poppler missing)")
        else:
            log("⚠ OCR not available (Tesseract missing)")
        
        # Load embedding model
        log("Loading embedding model...")
        if not self.embedding_model.load(on_progress):
            log("Using keyword search (no embeddings)")
        
        # Load or build index
        if os.path.exists(self.index_file):
            log("Loading existing index...")
            self._load_index()
