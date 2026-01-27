"""
Local RAG Application with Qwen2.5:0.5b and BGE-small-en
A beautiful CustomTkinter interface for document Q&A
"""

import os
import sys
import json
import threading
import queue
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

# Document processing
from docx import Document as DocxDocument
import fitz  # PyMuPDF
from openpyxl import load_workbook
import csv

# ML/AI
from llama_cpp import Llama
import numpy as np
from sentence_transformers import SentenceTransformer

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return str(base_path / relative_path)


class DocumentProcessor:
    """Handles document loading and text extraction"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.csv', '.xlsx', '.md'}
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from various document formats"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.pdf':
            return DocumentProcessor._extract_pdf(file_path)
        elif ext in {'.docx', '.doc'}:
            return DocumentProcessor._extract_docx(file_path)
        elif ext == '.txt' or ext == '.md':
            return DocumentProcessor._extract_txt(file_path)
        elif ext == '.csv':
            return DocumentProcessor._extract_csv(file_path)
        elif ext == '.xlsx':
            return DocumentProcessor._extract_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        doc = fitz.open(file_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        doc.close()
        return "\n".join(text)
    
    @staticmethod
    def _extract_docx(file_path: str) -> str:
        """Extract text from Word document"""
        doc = DocxDocument(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    @staticmethod
    def _extract_txt(file_path: str) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def _extract_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        rows = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        return "\n".join(rows)
    
    @staticmethod
    def _extract_xlsx(file_path: str) -> str:
        """Extract text from Excel file"""
        wb = load_workbook(file_path, read_only=True)
        text = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                if row_text.strip():
                    text.append(row_text)
        wb.close()
        return "\n".join(text)


class VectorStore:
    """Simple in-memory vector store for RAG"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.sources: List[str] = []
    
    def add_documents(self, chunks: List[str], source: str):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        new_embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search for most similar chunks"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], self.sources[idx], float(similarities[idx])))
        
        return results
    
    def clear(self):
        """Clear all stored documents"""
        self.chunks = []
        self.embeddings = None
        self.sources = []


class RAGEngine:
    """Main RAG engine combining embeddings and LLM"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda x: None)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.llm: Optional[Llama] = None
        self.vector_store: Optional[VectorStore] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.loaded_documents: List[str] = []
        
    def initialize(self):
        """Initialize models"""
        self.status_callback("Loading embedding model...")
        
        # Try to load local model first, fallback to download
        model_path = get_resource_path("models/bge-small-en")
        
        if os.path.exists(model_path):
            self.embedding_model = SentenceTransformer(model_path)
        else:
            self.status_callback("Downloading embedding model...")
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        self.vector_store = VectorStore(self.embedding_model)
        
        self.status_callback("Loading LLM...")
        llm_path = get_resource_path("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
        
        if os.path.exists(llm_path):
            self.llm = Llama(
                model_path=llm_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
        else:
            raise FileNotFoundError(f"LLM model not found at {llm_path}")
        
        self.status_callback("Models loaded successfully!")
    
    def add_document(self, file_path: str) -> int:
        """Add a document to the knowledge base"""
        self.status_callback(f"Processing: {Path(file_path).name}")
        
        # Extract text
        text = DocumentProcessor.extract_text(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add to vector store
        source = Path(file_path).name
        self.vector_store.add_documents(chunks, source)
        self.loaded_documents.append(source)
        
        self.status_callback(f"Added {len(chunks)} chunks from {source}")
        return len(chunks)
    
    def query(self, question: str, stream_callback=None) -> str:
        """Query the RAG system"""
        if not self.vector_store or len(self.vector_store.chunks) == 0:
            return "Please load some documents first."
        
        # Retrieve relevant chunks
        results = self.vector_store.search(question, top_k=3)
        
        # Build context
        context_parts = []
        for chunk, source, score in results:
            context_parts.append(f"[Source: {source}]\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions based on the provided context. 
Be concise and accurate. If the answer is not in the context, say so.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        # Generate response
        if stream_callback:
            response_text = ""
            for token in self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["<|im_end|>", "<|im_start|>"],
                stream=True
            ):
                text = token['choices'][0]['text']
                response_text += text
                stream_callback(text)
            return response_text
        else:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            return response['choices'][0]['text'].strip()
    
    def clear_documents(self):
        """Clear all loaded documents"""
        if self.vector_store:
            self.vector_store.clear()
        self.loaded_documents = []


class ModernRAGApp(ctk.CTk):
    """Modern CustomTkinter RAG Application"""
    
    # Color scheme - Warm amber/gold theme with dark mode
    COLORS = {
        'bg_dark': '#0D0D0D',
        'bg_secondary': '#1A1A1A',
        'bg_tertiary': '#252525',
        'accent': '#E6A23C',
        'accent_hover': '#F5BA5C',
        'accent_dim': '#8B6914',
        'text_primary': '#FAFAFA',
        'text_secondary': '#A0A0A0',
        'text_muted': '#606060',
        'success': '#67C23A',
        'error': '#F56C6C',
        'border': '#333333',
    }
    
    def __init__(self):
        super().__init__()
        
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Window setup
        self.title("⚡ Local RAG Assistant")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.configure(fg_color=self.COLORS['bg_dark'])
        
        # State
        self.rag_engine: Optional[RAGEngine] = None
        self.is_initialized = False
        self.is_processing = False
        self.message_queue = queue.Queue()
        
        # Setup UI
        self._setup_fonts()
        self._create_layout()
        self._start_queue_processor()
        
        # Auto-initialize on start
        self.after(500, self._initialize_models)
    
    def _setup_fonts(self):
        """Setup custom fonts"""
        self.font_title = ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
        self.font_heading = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        self.font_body = ctk.CTkFont(family="Segoe UI", size=14)
        self.font_small = ctk.CTkFont(family="Segoe UI", size=12)
        self.font_mono = ctk.CTkFont(family="Consolas", size=13)
    
    def _create_layout(self):
        """Create the main layout"""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create sidebar
        self._create_sidebar()
        
        # Create main content
        self._create_main_content()
    
    def _create_sidebar(self):
        """Create the left sidebar"""
        sidebar = ctk.CTkFrame(
            self,
            width=300,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=0
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        
        # Logo/Title area
        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(30, 20))
        
        logo_label = ctk.CTkLabel(
            title_frame,
            text="⚡",
            font=ctk.CTkFont(size=40)
        )
        logo_label.pack()
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="Local RAG",
            font=self.font_title,
            text_color=self.COLORS['text_primary']
        )
        title_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Powered by Qwen2.5",
            font=self.font_small,
            text_color=self.COLORS['text_muted']
        )
        subtitle_label.pack()
        
        # Divider
        divider = ctk.CTkFrame(sidebar, height=1, fg_color=self.COLORS['border'])
        divider.pack(fill="x", padx=20, pady=20)
        
        # Documents section
        docs_header = ctk.CTkFrame(sidebar, fg_color="transparent")
        docs_header.pack(fill="x", padx=20, pady=(0, 10))
        
        docs_label = ctk.CTkLabel(
            docs_header,
            text="📁 Documents",
            font=self.font_heading,
            text_color=self.COLORS['text_primary'],
            anchor="w"
        )
        docs_label.pack(side="left")
        
        # Add document button
        self.add_doc_btn = ctk.CTkButton(
            sidebar,
            text="+ Add Document",
            font=self.font_body,
            fg_color=self.COLORS['accent'],
            hover_color=self.COLORS['accent_hover'],
            text_color=self.COLORS['bg_dark'],
            height=45,
            corner_radius=10,
            command=self._add_document
        )
        self.add_doc_btn.pack(fill="x", padx=20, pady=(0, 10))
        
        # Document list frame
        self.doc_list_frame = ctk.CTkScrollableFrame(
            sidebar,
            fg_color=self.COLORS['bg_tertiary'],
            corner_radius=10,
            height=200
        )
        self.doc_list_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Empty state label
        self.empty_docs_label = ctk.CTkLabel(
            self.doc_list_frame,
            text="No documents loaded\nAdd PDF, DOCX, TXT, CSV, or XLSX",
            font=self.font_small,
            text_color=self.COLORS['text_muted'],
            justify="center"
        )
        self.empty_docs_label.pack(pady=30)
        
        # Clear documents button
        self.clear_docs_btn = ctk.CTkButton(
            sidebar,
            text="🗑 Clear All",
            font=self.font_small,
            fg_color=self.COLORS['bg_tertiary'],
            hover_color=self.COLORS['border'],
            text_color=self.COLORS['text_secondary'],
            height=35,
            corner_radius=8,
            command=self._clear_documents
        )
        self.clear_docs_btn.pack(fill="x", padx=20, pady=(0, 20))
        
        # Status section at bottom
        status_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        
        # Model status
        self.status_indicator = ctk.CTkFrame(
            status_frame,
            fg_color=self.COLORS['bg_tertiary'],
            corner_radius=10
        )
        self.status_indicator.pack(fill="x")
        
        status_inner = ctk.CTkFrame(self.status_indicator, fg_color="transparent")
        status_inner.pack(fill="x", padx=15, pady=12)
        
        self.status_dot = ctk.CTkLabel(
            status_inner,
            text="●",
            font=ctk.CTkFont(size=10),
            text_color=self.COLORS['text_muted']
        )
        self.status_dot.pack(side="left")
        
        self.status_text = ctk.CTkLabel(
            status_inner,
            text="Initializing...",
            font=self.font_small,
            text_color=self.COLORS['text_secondary']
        )
        self.status_text.pack(side="left", padx=(8, 0))
    
    def _create_main_content(self):
        """Create the main chat content area"""
        main = ctk.CTkFrame(self, fg_color=self.COLORS['bg_dark'], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)
        
        # Chat container
        chat_container = ctk.CTkFrame(main, fg_color="transparent")
        chat_container.grid(row=0, column=0, sticky="nsew", padx=30, pady=20)
        chat_container.grid_columnconfigure(0, weight=1)
        chat_container.grid_rowconfigure(0, weight=1)
        
        # Chat display area
        self.chat_frame = ctk.CTkScrollableFrame(
            chat_container,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=15
        )
        self.chat_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Welcome message
        self._add_welcome_message()
        
        # Input area
        input_frame = ctk.CTkFrame(
            chat_container,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=15,
            height=80
        )
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_propagate(False)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Input container for padding
        input_inner = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_inner.pack(fill="both", expand=True, padx=15, pady=15)
        input_inner.grid_columnconfigure(0, weight=1)
        
        # Text input
        self.input_entry = ctk.CTkEntry(
            input_inner,
            placeholder_text="Ask a question about your documents...",
            font=self.font_body,
            fg_color=self.COLORS['bg_tertiary'],
            border_color=self.COLORS['border'],
            text_color=self.COLORS['text_primary'],
            placeholder_text_color=self.COLORS['text_muted'],
            height=50,
            corner_radius=12
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", lambda e: self._send_message())
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_inner,
            text="Send ➤",
            font=self.font_body,
            fg_color=self.COLORS['accent'],
            hover_color=self.COLORS['accent_hover'],
            text_color=self.COLORS['bg_dark'],
            width=100,
            height=50,
            corner_radius=12,
            command=self._send_message
        )
        self.send_btn.pack(side="right")
    
    def _add_welcome_message(self):
        """Add welcome message to chat"""
        welcome_frame = ctk.CTkFrame(
            self.chat_frame,
            fg_color="transparent"
        )
        welcome_frame.pack(fill="x", padx=30, pady=40)
        
        welcome_icon = ctk.CTkLabel(
            welcome_frame,
            text="🤖",
            font=ctk.CTkFont(size=50)
        )
        welcome_icon.pack(pady=(0, 15))
        
        welcome_title = ctk.CTkLabel(
            welcome_frame,
            text="Welcome to Local RAG Assistant",
            font=self.font_heading,
            text_color=self.COLORS['text_primary']
        )
        welcome_title.pack()
        
        welcome_text = ctk.CTkLabel(
            welcome_frame,
            text="Add documents using the sidebar, then ask questions about them.\nAll processing happens locally - your data never leaves your computer.",
            font=self.font_body,
            text_color=self.COLORS['text_secondary'],
            justify="center"
        )
        welcome_text.pack(pady=(10, 0))
    
    def _add_message(self, content: str, is_user: bool = False, is_streaming: bool = False):
        """Add a message to the chat"""
        # Message container
        msg_container = ctk.CTkFrame(
            self.chat_frame,
            fg_color="transparent"
        )
        msg_container.pack(fill="x", padx=20, pady=10)
        
        # Alignment frame
        align_frame = ctk.CTkFrame(msg_container, fg_color="transparent")
        if is_user:
            align_frame.pack(side="right")
        else:
            align_frame.pack(side="left")
        
        # Message bubble
        bubble_color = self.COLORS['accent'] if is_user else self.COLORS['bg_tertiary']
        text_color = self.COLORS['bg_dark'] if is_user else self.COLORS['text_primary']
        
        bubble = ctk.CTkFrame(
            align_frame,
            fg_color=bubble_color,
            corner_radius=15
        )
        bubble.pack()
        
        # Message label
        msg_label = ctk.CTkLabel(
            bubble,
            text=content,
            font=self.font_body,
            text_color=text_color,
            wraplength=500,
            justify="left" if not is_user else "right"
        )
        msg_label.pack(padx=18, pady=12)
        
        # Store reference for streaming updates
        if is_streaming:
            self.current_response_label = msg_label
            self.current_response_bubble = bubble
        
        # Scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
        return msg_label
    
    def _update_streaming_message(self, text: str):
        """Update the current streaming message"""
        if hasattr(self, 'current_response_label'):
            current_text = self.current_response_label.cget("text")
            self.current_response_label.configure(text=current_text + text)
            self.chat_frame._parent_canvas.yview_moveto(1.0)
    
    def _add_document(self):
        """Add a document to the knowledge base"""
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Please wait for models to initialize.")
            return
        
        if self.is_processing:
            return
        
        filetypes = [
            ("All Supported", "*.pdf *.docx *.doc *.txt *.csv *.xlsx *.md"),
            ("PDF", "*.pdf"),
            ("Word", "*.docx *.doc"),
            ("Text", "*.txt *.md"),
            ("Excel", "*.xlsx"),
            ("CSV", "*.csv"),
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=filetypes
        )
        
        if file_path:
            self.is_processing = True
            self.add_doc_btn.configure(state="disabled")
            
            def process():
                try:
                    chunks = self.rag_engine.add_document(file_path)
                    self.message_queue.put(("doc_added", (file_path, chunks)))
                except Exception as e:
                    self.message_queue.put(("error", str(e)))
                finally:
                    self.message_queue.put(("processing_done", None))
            
            threading.Thread(target=process, daemon=True).start()
    
    def _add_document_to_list(self, file_path: str, chunks: int):
        """Add a document entry to the sidebar list"""
        # Hide empty state
        self.empty_docs_label.pack_forget()
        
        # Document item frame
        doc_frame = ctk.CTkFrame(
            self.doc_list_frame,
            fg_color=self.COLORS['bg_secondary'],
            corner_radius=8
        )
        doc_frame.pack(fill="x", padx=5, pady=3)
        
        # Icon based on file type
        ext = Path(file_path).suffix.lower()
        icons = {
            '.pdf': '📕', '.docx': '📘', '.doc': '📘',
            '.txt': '📄', '.md': '📝', '.csv': '📊', '.xlsx': '📊'
        }
        icon = icons.get(ext, '📄')
        
        icon_label = ctk.CTkLabel(
            doc_frame,
            text=icon,
            font=ctk.CTkFont(size=16)
        )
        icon_label.pack(side="left", padx=(10, 5), pady=8)
        
        # File info
        info_frame = ctk.CTkFrame(doc_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, pady=8)
        
        name_label = ctk.CTkLabel(
            info_frame,
            text=Path(file_path).name[:25] + "..." if len(Path(file_path).name) > 25 else Path(file_path).name,
            font=self.font_small,
            text_color=self.COLORS['text_primary'],
            anchor="w"
        )
        name_label.pack(anchor="w")
        
        chunks_label = ctk.CTkLabel(
            info_frame,
            text=f"{chunks} chunks",
            font=ctk.CTkFont(size=10),
            text_color=self.COLORS['text_muted'],
            anchor="w"
        )
        chunks_label.pack(anchor="w")
    
    def _clear_documents(self):
        """Clear all loaded documents"""
        if not self.is_initialized:
            return
        
        if self.rag_engine and self.rag_engine.loaded_documents:
            if messagebox.askyesno("Clear Documents", "Remove all loaded documents?"):
                self.rag_engine.clear_documents()
                
                # Clear document list UI
                for widget in self.doc_list_frame.winfo_children():
                    widget.destroy()
                
                # Show empty state
                self.empty_docs_label = ctk.CTkLabel(
                    self.doc_list_frame,
                    text="No documents loaded\nAdd PDF, DOCX, TXT, CSV, or XLSX",
                    font=self.font_small,
                    text_color=self.COLORS['text_muted'],
                    justify="center"
                )
                self.empty_docs_label.pack(pady=30)
                
                self._update_status("Documents cleared", "success")
    
    def _send_message(self):
        """Send a message and get response"""
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Please wait for models to initialize.")
            return
        
        if self.is_processing:
            return
        
        question = self.input_entry.get().strip()
        if not question:
            return
        
        # Clear input
        self.input_entry.delete(0, 'end')
        
        # Add user message
        self._add_message(question, is_user=True)
        
        # Add placeholder for response
        self._add_message("", is_user=False, is_streaming=True)
        
        # Process in thread
        self.is_processing = True
        self.send_btn.configure(state="disabled")
        
        def process():
            try:
                def stream_callback(text):
                    self.message_queue.put(("stream", text))
                
                self.rag_engine.query(question, stream_callback=stream_callback)
            except Exception as e:
                self.message_queue.put(("error", str(e)))
            finally:
                self.message_queue.put(("processing_done", None))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _initialize_models(self):
        """Initialize the RAG engine"""
        def init():
            try:
                self.rag_engine = RAGEngine(
                    status_callback=lambda msg: self.message_queue.put(("status", msg))
                )
                self.rag_engine.initialize()
                self.message_queue.put(("initialized", None))
            except Exception as e:
                self.message_queue.put(("init_error", str(e)))
        
        threading.Thread(target=init, daemon=True).start()
    
    def _update_status(self, text: str, status_type: str = "normal"):
        """Update the status indicator"""
        colors = {
            "normal": self.COLORS['text_muted'],
            "success": self.COLORS['success'],
            "error": self.COLORS['error'],
            "loading": self.COLORS['accent']
        }
        
        self.status_dot.configure(text_color=colors.get(status_type, colors["normal"]))
        self.status_text.configure(text=text)
    
    def _start_queue_processor(self):
        """Process messages from the queue"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "status":
                    self._update_status(data, "loading")
                
                elif msg_type == "initialized":
                    self.is_initialized = True
                    self._update_status("Ready", "success")
                    self.add_doc_btn.configure(state="normal")
                
                elif msg_type == "init_error":
                    self._update_status(f"Error: {data}", "error")
                    messagebox.showerror("Initialization Error", data)
                
                elif msg_type == "doc_added":
                    file_path, chunks = data
                    self._add_document_to_list(file_path, chunks)
                    self._update_status(f"Added: {Path(file_path).name}", "success")
                
                elif msg_type == "stream":
                    self._update_streaming_message(data)
                
                elif msg_type == "error":
                    if hasattr(self, 'current_response_label'):
                        self.current_response_label.configure(text=f"Error: {data}")
                    self._update_status("Error occurred", "error")
                
                elif msg_type == "processing_done":
                    self.is_processing = False
                    self.add_doc_btn.configure(state="normal")
                    self.send_btn.configure(state="normal")
                    
        except queue.Empty:
            pass
        
        self.after(50, self._start_queue_processor)


def main():
    app = ModernRAGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
