"""
User Interface.
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog
import config


class ChatUI:
    """Chat user interface."""
    
    def __init__(self, root: tk.Tk, on_send, on_clear, on_load_files):
        self.root = root
        self.on_send = on_send
        self.on_clear = on_clear
        self.on_load_files = on_load_files
        
        self._setup_window()
        self._create_widgets()
    
    def _setup_window(self):
        """Configures main window."""
        self.root.title(f"🤖 {config.APP_NAME}")
        self.root.geometry(config.WINDOW_SIZE)
        self.root.configure(bg=config.COLORS["bg"])
    
    def _create_widgets(self):
        """Creates all UI widgets."""
        c = config.COLORS
        f = config.FONTS
        
        # Header
        header = tk.Frame(self.root, bg=c["bg"])
        header.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(
            header, 
            text=f"🤖 {config.APP_NAME}",
            font=f["title"],
            bg=c["bg"],
            fg=c["fg"]
        ).pack(side=tk.LEFT)
        
        self.status = tk.Label(
            header,
            text="⏳ Loading...",
            font=("Arial", 11),
            bg=c["bg"],
            fg=c["system"]
        )
        self.status.pack(side=tk.RIGHT)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg=c["bg"])
        btn_frame.pack(fill=tk.X, padx=20)
        
        tk.Button(
            btn_frame,
            text="📁 Load Files",
            font=f["button"],
            bg=c["accent"],
            fg="white",
            relief=tk.FLAT,
            command=self._load_files
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            btn_frame,
            text="🗑️ Clear",
            font=f["button"],
            bg=c["accent"],
            fg="white",
            relief=tk.FLAT,
            command=self.on_clear
        ).pack(side=tk.LEFT)
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg=c["bg"])
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=f["chat"],
            bg=c["bg_chat"],
            fg=c["fg"],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)
        
        # Tags
        self.chat.tag_configure("user", foreground=c["user"], font=(f["chat"][0], f["chat"][1], "bold"))
        self.chat.tag_configure("bot", foreground=c["bot"], font=(f["chat"][0], f["chat"][1], "bold"))
        self.chat.tag_configure("system", foreground=c["system"])
        self.chat.tag_configure("error", foreground=c["error"])
        
        # Input area
        input_frame = tk.Frame(self.root, bg=c["bg"])
        input_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.input = tk.Text(
            input_frame,
            height=2,
            font=f["chat"],
            bg=c["bg_chat"],
            fg=c["fg"],
            relief=tk.FLAT,
            padx=10,
            pady=8
        )
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input.bind("<Return>", self._on_enter)
        
        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            font=f["button"],
            bg=c["accent"],
            fg="white",
            relief=tk.FLAT,
            padx=20,
            command=self._send
        )
        self.send_btn.pack(side=tk.RIGHT)
    
    def _on_enter(self, event):
        if not (event.state & 0x1):  # Shift not pressed
            self._send()
            return "break"
    
    def _send(self):
        text = self.input.get("1.0", tk.END).strip()
        if text:
            self.input.delete("1.0", tk.END)
            self.on_send(text)
    
    def set_status(self, text: str, is_error: bool = False):
        """Updates status label."""
        color = config.COLORS["error"] if is_error else config.COLORS["bot"]
        self.status.config(text=text, fg=color)
    
    def add_message(self, sender: str, text: str, tag: str = ""):
        """Adds a message to chat."""
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"\n{sender}:\n", tag)
        self.chat.insert(tk.END, f"{text}\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
    
    def stream(self, text: str):
        """Streams text to chat."""
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
    
    def clear_chat(self):
        """Clears chat display."""
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.config(state=tk.DISABLED)
    
    def set_enabled(self, enabled: bool):
        """Enables/disables input."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.send_btn.config(state=state)
    
    def focus_input(self):
        """Focuses input field."""
        self.input.focus_set()
    
    def _load_files(self):
        """Opens file dialog to load documents."""
        self.root.attributes('-topmost', True)
        self.root.update()
        
        files = filedialog.askopenfilenames(
            parent=self.root,
            title="Select documents",
            filetypes=[
                ("All supported", "*.txt *.md *.pdf *.xlsx *.xls *.pptx *.ppt *.py *.js *.json *.csv *.xml *.yaml *.yml"),
                ("Text files", "*.txt *.md"),
                ("Code files", "*.py *.js *.ts *.java *.c *.cpp *.go *.rs"),
                ("PDF files", "*.pdf"),
                ("Excel files", "*.xlsx *.xls"),
                ("PowerPoint files", "*.pptx *.ppt"),
                ("Data files", "*.json *.csv *.xml *.yaml *.yml"),
                ("All files", "*.*")
            ]
        )
        
        self.root.attributes('-topmost', False)
        
        if files:
            self.on_load_files(files)
