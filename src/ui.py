import os
import re
import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Callable
import config

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


class FileDownloadManager(ctk.CTkToplevel):
    """File download manager window (like ChatGPT)."""

    def __init__(self, parent, files: list, on_download: Callable, on_close: Callable):
        super().__init__(parent)

        self.title("📁 Download Generated Files")
        self.geometry("600x400")
        self.transient(parent)
        self.grab_set()

        self.on_download = on_download
        self.on_close = on_close

        self._create_widgets(files)

    def _create_widgets(self, files: list):
        # Header
        header = ctk.CTkFrame(self, height=60)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="📁 Files Ready for Download",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", pady=10)

        ctk.CTkButton(
            header, text="✕", width=30, height=30,
            command=self._close,
            fg_color="transparent", hover_color=("#ff5555", "#cc4444")
        ).pack(side="right", padx=10)

        # Instructions
        instructions = ctk.CTkLabel(
            self,
            text="Click a file to choose where to save it. Files will be deleted when you close this window.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray50")
        )
        instructions.pack(pady=(0, 10))

        # File list
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        for file_info in files:
            self._create_file_item(scroll_frame, file_info)

        # Footer with close button
        footer = ctk.CTkFrame(self, height=50)
        footer.pack(fill="x", padx=10, pady=(0, 10))
        footer.pack_propagate(False)

        ctk.CTkButton(
            footer, text="Close (Files will be deleted)",
            command=self._close,
            fg_color=("#ff5555", "#cc4444"), hover_color=("#ff7777", "#dd5555")
        ).pack(side="right", pady=10)

    def _create_file_item(self, parent, file_info: dict):
        """Create a file item with download button."""
        item_frame = ctk.CTkFrame(
            parent,
            fg_color=("gray85", "gray25"),
            corner_radius=8
        )
        item_frame.pack(fill="x", pady=5)

        # File icon based on type
        ext = os.path.splitext(file_info['filename'])[1].lower()
        icons = {
            '.pdf': '📕', '.docx': '📘', '.xlsx': '📗',
            '.pptx': '📙', '.txt': '📄', '.csv': '📊',
            '.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️',
            '.json': '📋', '.html': '🌐'
        }
        icon = icons.get(ext, '📎')

        # Icon and filename
        left_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        left_frame.pack(side="left", padx=10, pady=8)

        ctk.CTkLabel(
            left_frame, text=icon, font=ctk.CTkFont(size=24)
        ).pack(side="left", padx=(0, 10))

        text_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        text_frame.pack(side="left")

        ctk.CTkLabel(
            text_frame,
            text=file_info['filename'],
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        ).pack(fill="x")

        size_mb = file_info['size'] / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB" if size_mb > 1 else f"{file_info['size']} bytes"

        ctk.CTkLabel(
            text_frame,
            text=f"{file_info['mime_type']} • {size_str}",
            font=ctk.CTkFont(size=10),
            text_color=("gray60", "gray40"),
            anchor="w"
        ).pack(fill="x")

        # Download button
        ctk.CTkButton(
            item_frame,
            text="⬇️ Download",
            width=100, height=32,
            command=lambda f=file_info: self._download_file(f),
            fg_color=("#50fa7b", "#40c969"),
            hover_color=("#40c969", "#30b959"),
            text_color=("#000000", "#000000")
        ).pack(side="right", padx=10, pady=8)

    def _download_file(self, file_info: dict):
        """Handle file download."""
        destination = self.on_download(file_info)
        if destination:
            messagebox.showinfo(
                "Download Complete",
                f"File saved to:\n{destination}"
            )
        else:
            messagebox.showwarning(
                "Download Cancelled",
                "File download was cancelled."
            )

    def _close(self):
        """Close the download manager."""
        self.on_close()
        self.destroy()


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

        # Code execution state
        self._code_executor = None
        self._pending_files = []
        self._download_manager_open = False

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

    def _format_code_blocks(self, text: str) -> str:
        """Format Python code blocks with visual styling."""
        def format_block(match):
            code = match.group(1)
            return f"📝 CODE:\n{'─'*40}\n{code}\n{'─'*40}"

        pattern = r'```python\n(.*?)\n```'
        return re.sub(pattern, format_block, text, flags=re.DOTALL)

    def prompt_file_save(self, filename: str, default_name: str) -> str:
        """Prompt user to choose save location for generated file."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            title=f"Save {filename}",
            defaultextension=os.path.splitext(default_name)[1],
            initialfile=default_name,
            filetypes=[
                ("All files", "*.*"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"),
                ("Excel files", "*.xlsx"),
                ("PowerPoint files", "*.pptx"),
                ("PNG images", "*.png"),
                ("Text files", "*.txt"),
            ]
        )
        return filepath

    def show_code_status(self, status: str, icon: str = "⚡"):
        """Show code execution status in chat."""
        status_msg = f"{icon} {status}"
        self.add_message("System", status_msg)

    def handle_code_execution_complete(self, result, executor):
        """
        Handle completion of code execution with file downloads.

        Args:
            result: ExecutionResult from code execution
            executor: The code executor instance
        """
        self.add_message("System", format_code_output(result))

        # If files were created, offer downloads
        if result.files_created:
            self._pending_files = result.files_created
            self._code_executor = executor

            # Add download prompt
            self.add_message("System",
                f"\n💾 {len(result.files_created)} file(s) ready for download!")
            self.add_message("System",
                "Click 'Download Files' to save them to your computer.")

            # Show download button
            self._show_download_button()

    def _show_download_button(self):
        """Show download files button in chat."""
        # Insert a clickable download indicator
        self.chat.insert("end", "\n")
        self.chat.see("end")

    def open_download_manager(self):
        """Open the file download manager window."""
        if not self._pending_files or self._download_manager_open:
            return

        self._download_manager_open = True

        def on_download(file_info):
            """Handle file download."""
            from tkinter import filedialog

            ext = os.path.splitext(file_info['filename'])[1]
            filepath = filedialog.asksaveasfilename(
                title=f"Save {file_info['filename']}",
                defaultextension=ext,
                initialfile=file_info['filename'],
                filetypes=[
                    ("All files", "*.*"),
                    (f"{file_info['filename']} files", f"*{ext}"),
                ]
            )

            if filepath:
                if self._code_executor and self._code_executor.save_file_to(
                    file_info['filename'], filepath
                ):
                    return filepath
            return None

        def on_close():
            """Handle download manager close."""
            self._download_manager_open = False
            # Cleanup temp files
            if self._code_executor:
                self._code_executor.cleanup_temp_files()
            self._pending_files = []
            self.add_message("System", "🗑️ Temporary files cleaned up.")

        FileDownloadManager(
            self.root,
            self._pending_files,
            on_download,
            on_close
        )

    def set_code_executor(self, executor):
        """Set the current code executor for file management."""
        self._code_executor = executor


def format_code_output(result) -> str:
    """Format execution result for display (import from code_executor)."""
    from code_executor import format_code_output as _format
    return _format(result)
