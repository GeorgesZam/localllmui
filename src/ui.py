"""
Modern User Interface with CustomTkinter and Sidebar.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Callable, Optional
import config

# Configuration du thème
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ConversationItem(ctk.CTkFrame):
    """Single conversation item in sidebar."""
    
    def __init__(self, parent, conv_id: str, title: str, doc_count: int,
                 is_active: bool, on_select: Callable, on_delete: Callable):
        super().__init__(parent, corner_radius=8, height=50)
        
        self.conv_id = conv_id
        self.on_select = on_select
        self.on_delete = on_delete
        
        # Colors based on active state
        if is_active:
            self.configure(fg_color=("#3b7ac7", "#2d5f9e"))
        else:
            self.configure(fg_color=("gray75", "gray25"))
        
        # Make clickable
        self.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        # Title
        title_label = ctk.CTkLabel(
            self,
            text=title[:25] + "..." if len(title) > 25 else title,
            font=ctk.CTkFont(size=12, weight="bold" if is_active else "normal"),
            anchor="w"
        )
        title_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)
        title_label.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        # Doc count badge
        if doc_count > 0:
            doc_badge = ctk.CTkLabel(
                self,
                text=f"📄{doc_count}",
                font=ctk.CTkFont(size=10),
                text_color=("#50fa7b", "#40c969")
            )
            doc_badge.pack(side="left", padx=(0, 5))
            doc_badge.bind("<Button-1>", lambda e: self.on_select(self.conv_id))
        
        # Delete button
        delete_btn = ctk.CTkButton(
            self,
            text="✕",
            width=24,
            height=24,
            corner_radius=4,
            fg_color="transparent",
            hover_color=("#ff5555", "#cc4444"),
            font=ctk.CTkFont(size=12),
            command=lambda: self.on_delete(self.conv_id)
        )
        delete_btn.pack(side="right", padx=5)


class Sidebar(ctk.CTkFrame):
    """Sidebar with conversation list."""
    
    def __init__(self, parent, on_new: Callable, on_select: Callable, on_delete: Callable):
        super().__init__(parent, width=250, corner_radius=0)
        
        self.on_new = on_new
        self.on_select = on_select
        self.on_delete = on_delete
        
        self.pack_propagate(False)
        self._create_widgets()
    
    def _create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)
        
        title = ctk.CTkLabel(
            header,
            text="💬 Chats",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(side="left", pady=10)
        
        # New chat button
        new_btn = ctk.CTkButton(
            header,
            text="+ New",
            width=70,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#50fa7b", "#40c969"),
            hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000"),
            command=self.on_new
        )
        new_btn.pack(side="right", pady=10)
        
        # Separator
        sep = ctk.CTkFrame(self, height=2, fg_color=("gray70", "gray30"))
        sep.pack(fill="x", padx=10, pady=(0, 10))
        
        # Scrollable conversation list
        self.conv_list = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=("#4a9eff", "#3b7ac7")
        )
        self.conv_list.pack(fill="both", expand=True, padx=5, pady=5)
    
    def update_conversations(self, conversations: list, current_id: str):
        """Update conversation list."""
        # Clear existing
        for widget in self.conv_list.winfo_children():
            widget.destroy()
        
        if not conversations:
            empty_label = ctk.CTkLabel(
                self.conv_list,
                text="No conversations yet.\nClick '+ New' to start!",
                font=ctk.CTkFont(size=12),
                text_color=("gray50", "gray50")
            )
            empty_label.pack(pady=20)
            return
        
        # Add conversation items
        for conv in conversations:
            item = ConversationItem(
                self.conv_list,
                conv_id=conv.id,
                title=conv.title,
                doc_count=len(conv.document_ids),
                is_active=(conv.id == current_id),
                on_select=self.on_select,
                on_delete=self._confirm_delete
            )
            item.pack(fill="x", pady=2)
    
    def _confirm_delete(self, conv_id: str):
        """Confirm before deleting."""
        if messagebox.askyesno("Delete Chat", "Delete this conversation and its documents?"):
            self.on_delete(conv_id)


class ChatUI:
    """Modern chat UI with sidebar."""
    
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
        """Configure main window."""
        self.root.title(f"🤖 {config.APP_NAME}")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        
        # Grid config
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
    
    def _create_widgets(self):
        """Create all widgets."""
        
        # === SIDEBAR ===
        self.sidebar = Sidebar(
            self.root,
            on_new=self.on_new_chat,
            on_select=self.on_select_chat,
            on_delete=self.on_delete_chat
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # === MAIN AREA ===
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # --- Header ---
        header = ctk.CTkFrame(main_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            header,
            text=f"🤖 {config.APP_NAME}",
            font=ctk.CTkFont(family="Arial", size=24, weight="bold")
        )
        title_label.pack(side="left")
        
        self.status = ctk.CTkLabel(
            header,
            text="⏳ Loading...",
            font=ctk.CTkFont(size=12),
            text_color=("#888888", "#888888")
        )
        self.status.pack(side="right", padx=10)
        
        # --- Toolbar ---
        toolbar = ctk.CTkFrame(main_frame, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.load_btn = ctk.CTkButton(
            toolbar,
            text="📁 Load Files",
            command=self._load_files,
            width=120,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#4a9eff", "#3b7ac7"),
            hover_color=("#3b7ac7", "#2d5f9e")
        )
        self.load_btn.pack(side="left", padx=(0, 10))
        
        self.clear_btn = ctk.CTkButton(
            toolbar,
            text="🗑️ Clear",
            command=self.on_clear,
            width=100,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#ff5555", "#cc4444"),
            hover_color=("#ff7777", "#dd5555")
        )
        self.clear_btn.pack(side="left")
        
        self.doc_info = ctk.CTkLabel(
            toolbar,
            text="📚 No documents",
            font=ctk.CTkFont(size=11),
            text_color=("#888888", "#888888")
        )
        self.doc_info.pack(side="right", padx=10)
        
        # --- Chat Area ---
        chat_container = ctk.CTkFrame(main_frame, corner_radius=10)
        chat_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)
        
        self.chat = ctk.CTkTextbox(
            chat_container,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10,
            fg_color=("#1e1e2e", "#16213e"),
            scrollbar_button_color=("#4a9eff", "#3b7ac7")
        )
        self.chat.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        # --- Input Area ---
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input = ctk.CTkTextbox(
            input_frame,
            height=60,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10,
            border_width=2,
            border_color=("#4a9eff", "#3b7ac7"),
            fg_color=("#1e1e2e", "#16213e")
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)
        
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send ➤",
            command=self._send,
            width=90,
            height=60,
            corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#50fa7b", "#40c969"),
            hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000")
        )
        self.send_btn.grid(row=0, column=1)
    
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
            parent=self.root,
            title="Select documents",
            filetypes=[
                ("All supported", 
                 "*.txt *.md *.pdf *.xlsx *.xls *.pptx *.ppt *.docx *.doc "
                 "*.py *.js *.json *.csv *.xml *.yaml *.yml *.html *.css "
                 "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("Documents", "*.txt *.md *.pdf *.docx *.doc"),
                ("Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Presentations", "*.pptx *.ppt"),
                ("Code", "*.py *.js *.ts *.json *.html *.css *.xml *.yaml"),
                ("Images (OCR)", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.on_load_files(files)
    
    # === Public methods ===
    
    def set_status(self, text: str, is_error: bool = False):
        color = ("#ff5555", "#ff5555") if is_error else ("#50fa7b", "#50fa7b")
        self.status.configure(text=text, text_color=color)
    
    def update_doc_count(self, count: int):
        if count == 0:
            text, color = "📚 No documents", ("#888888", "#888888")
        else:
            text = f"📚 {count} doc{'s' if count > 1 else ''}"
            color = ("#50fa7b", "#50fa7b")
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
        self.send_btn.configure(state=state)
        self.input.configure(state=state)
    
    def focus_input(self):
        self.input.focus_set()
    
    def update_sidebar(self, conversations: list, current_id: str):
        """Update sidebar with conversations."""
        self.sidebar.update_conversations(conversations, current_id)
    
    def load_messages(self, messages: list):
        """Load messages into chat area."""
        self.clear_chat()
        for msg in messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            self.add_message(role, msg["content"], msg["role"])
