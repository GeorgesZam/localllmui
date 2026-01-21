"""
Modern User Interface with CustomTkinter.
"""

import customtkinter as ctk
from tkinter import filedialog
import config

# Configuration du thème
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatUI:
    """Modern chat user interface with CustomTkinter."""
    
    def __init__(self, root: ctk.CTk, on_send, on_clear, on_load_files):
        self.root = root
        self.on_send = on_send
        self.on_clear = on_clear
        self.on_load_files = on_load_files
        
        self._setup_window()
        self._create_widgets()
    
    def _setup_window(self):
        """Configure la fenêtre principale."""
        self.root.title(f"🤖 {config.APP_NAME}")
        self.root.geometry(config.WINDOW_SIZE)
        
        # Configuration grid pour responsive
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def _create_widgets(self):
        """Crée tous les widgets de l'interface."""
        
        # === HEADER ===
        header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        # Titre
        title_label = ctk.CTkLabel(
            header_frame,
            text=f"🤖 {config.APP_NAME}",
            font=ctk.CTkFont(family="Arial", size=28, weight="bold")
        )
        title_label.pack(side="left")
        
        # Status avec couleur dynamique
        self.status = ctk.CTkLabel(
            header_frame,
            text="⏳ Loading...",
            font=ctk.CTkFont(size=13),
            text_color=("#888888", "#888888")
        )
        self.status.pack(side="right", padx=10)
        
        # === TOOLBAR (Boutons) ===
        toolbar = ctk.CTkFrame(self.root, fg_color="transparent")
        toolbar.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Bouton Load Files
        self.load_btn = ctk.CTkButton(
            toolbar,
            text="📁 Load Files",
            command=self._load_files,
            width=140,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#4a9eff", "#3b7ac7"),
            hover_color=("#3b7ac7", "#2d5f9e")
        )
        self.load_btn.pack(side="left", padx=(0, 10))
        
        # Bouton Clear
        self.clear_btn = ctk.CTkButton(
            toolbar,
            text="🗑️ Clear Chat",
            command=self.on_clear,
            width=130,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#ff5555", "#cc4444"),
            hover_color=("#ff7777", "#dd5555")
        )
        self.clear_btn.pack(side="left")
        
        # Info documents
        self.doc_info = ctk.CTkLabel(
            toolbar,
            text="📚 No documents loaded",
            font=ctk.CTkFont(size=12),
            text_color=("#888888", "#888888")
        )
        self.doc_info.pack(side="right", padx=10)
        
        # === CHAT AREA ===
        chat_container = ctk.CTkFrame(self.root, corner_radius=10)
        chat_container.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="nsew")
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)
        
        self.chat = ctk.CTkTextbox(
            chat_container,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10,
            border_width=0,
            activate_scrollbars=True,
            fg_color=("#1e1e2e", "#16213e"),
            scrollbar_button_color=("#4a9eff", "#3b7ac7"),
            scrollbar_button_hover_color=("#3b7ac7", "#2d5f9e")
        )
        self.chat.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        # === INPUT AREA ===
        input_container = ctk.CTkFrame(self.root, fg_color="transparent")
        input_container.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        input_container.grid_columnconfigure(0, weight=1)
        
        # Input text
        self.input = ctk.CTkTextbox(
            input_container,
            height=70,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10,
            border_width=2,
            border_color=("#4a9eff", "#3b7ac7"),
            fg_color=("#1e1e2e", "#16213e")
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", self._on_shift_enter)
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_container,
            text="Send ➤",
            command=self._send,
            width=100,
            height=70,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#50fa7b", "#40c969"),
            hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000")
        )
        self.send_btn.grid(row=0, column=1)
    
    def _on_enter(self, event):
        """Gère l'appui sur Entrée."""
        if not (event.state & 0x1):  # Shift non pressé
            self._send()
            return "break"
    
    def _on_shift_enter(self, event):
        """Permet le saut de ligne avec Shift+Entrée."""
        return None
    
    def _send(self):
        """Envoie le message."""
        text = self.input.get("0.0", "end").strip()
        if text:
            self.input.delete("0.0", "end")
            self.on_send(text)
    
    def set_status(self, text: str, is_error: bool = False):
        """Met à jour le status."""
        if is_error:
            color = ("#ff5555", "#ff5555")
        else:
            color = ("#50fa7b", "#50fa7b")
        
        self.status.configure(text=text, text_color=color)
    
    def update_doc_count(self, count: int):
        """Met à jour le compteur de documents."""
        if count == 0:
            text = "📚 No documents loaded"
            color = ("#888888", "#888888")
        else:
            text = f"📚 {count} document{'s' if count > 1 else ''} loaded"
            color = ("#50fa7b", "#50fa7b")
        
        self.doc_info.configure(text=text, text_color=color)
    
    def add_message(self, sender: str, text: str, tag: str = ""):
        """Ajoute un message au chat avec formatage couleur."""
        current_text = self.chat.get("0.0", "end")
        
        if current_text.strip():
            self.chat.insert("end", "\n")
        
        self.chat.insert("end", f"{sender}:\n")
        self.chat.insert("end", f"{text}\n")
        self.chat.see("end")
    
    def stream(self, text: str):
        """Streaming de texte (pour la génération)."""
        self.chat.insert("end", text)
        self.chat.see("end")
    
    def clear_chat(self):
        """Efface le contenu du chat."""
        self.chat.delete("0.0", "end")
    
    def set_enabled(self, enabled: bool):
        """Active/désactive les contrôles."""
        state = "normal" if enabled else "disabled"
        self.send_btn.configure(state=state)
        self.input.configure(state=state)
    
    def focus_input(self):
        """Focus sur l'input."""
        self.input.focus_set()
    
    def _load_files(self):
        """Ouvre le dialogue de sélection de fichiers."""
        files = filedialog.askopenfilenames(
            parent=self.root,
            title="Select documents to load",
            filetypes=[
                ("All supported files", 
                 "*.txt *.md *.pdf *.xlsx *.xls *.pptx *.ppt *.docx *.doc "
                 "*.py *.js *.ts *.java *.c *.cpp *.json *.csv *.xml *.yaml *.yml *.html *.css "
                 "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("Documents", "*.txt *.md *.pdf *.docx *.doc"),
                ("Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Presentations", "*.pptx *.ppt"),
                ("Code files", "*.py *.js *.ts *.java *.c *.cpp *.json *.html *.css *.xml *.yaml *.yml"),
                ("Images (OCR)", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.on_load_files(files)
