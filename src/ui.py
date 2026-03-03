import os
import re
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox, Tk
from typing import Callable, Optional
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
                 on_new_chat, on_select_chat, on_delete_chat, skills_manager=None,
                 on_skill_toggle=None, on_open_model_catalog=None, model_manager=None):
        self.root = root
        self.on_send = on_send
        self.on_clear = on_clear
        self.on_load_files = on_load_files
        self.on_new_chat = on_new_chat
        self.on_select_chat = on_select_chat
        self.on_delete_chat = on_delete_chat
        self.on_skill_toggle = on_skill_toggle
        self.on_open_model_catalog = on_open_model_catalog
        self.model_manager = model_manager

        # Code execution state
        self._code_executor = None
        self._pending_files = []
        self._download_manager_open = False

        # Skills management
        from skills_manager import SkillsManager
        self.skills_manager = skills_manager if skills_manager else SkillsManager()
        self._skills_window = None

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
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            toolbar, text="🎯 Skills", command=self._open_skills_window,
            width=100, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#9b59b6", "#7d3c98"), hover_color=("#8e44ad", "#6c3483")
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            toolbar, text="🤖 Models", command=self._open_model_catalog,
            width=100, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#e67e22", "#d35400"), hover_color=("#d35400", "#ba4a00")
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

    def _open_skills_window(self):
        """Open the skills management window - single instance."""
        SkillsWindow(
            self.root,
            self.skills_manager,
            on_skill_toggle=self._on_skill_toggle
        )

    def _open_model_catalog(self):
        """Open the model catalog window."""
        if self.on_open_model_catalog:
            self.on_open_model_catalog()

    def _on_skill_toggle(self, skill_id: str, enabled: bool):
        """Handle skill toggle event."""
        status = "enabled" if enabled else "disabled"
        self.add_message("System", f"🎯 Skill '{skill_id}' {status}")
        self.skills_manager.save_config()

        # Notify app about skill toggle
        if self.on_skill_toggle:
            self.on_skill_toggle(skill_id, enabled)

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

    def get_skills_manager(self):
        """Get the skills manager instance."""
        return self.skills_manager

    def get_enabled_skills(self):
        """Get list of enabled skill IDs."""
        return [
            skill_id for skill_id, skill in self.skills_manager.skills.items()
            if skill.enabled
        ]


def format_code_output(result) -> str:
    """Format execution result for display (import from code_executor)."""
    from code_executor import format_code_output as _format
    return _format(result)


class SkillsWindow(ctk.CTkToplevel):
    """Simple Skills window."""

    _instance = None

    def __init__(self, parent, skills_manager, on_skill_toggle=None):
        # Close existing if any
        if SkillsWindow._instance:
            try:
                SkillsWindow._instance.destroy()
            except:
                pass

        super().__init__(parent)
        SkillsWindow._instance = self

        self.title("Skills")
        self.geometry("600x600")
        self.minsize(500, 500)
        self.skills_manager = skills_manager
        self.on_skill_toggle = on_skill_toggle

        self.protocol("WM_DELETE_WINDOW", lambda: self._close())

        self._build_ui()

    def _close(self):
        SkillsWindow._instance = None
        self.destroy()

    def _build_ui(self):
        # Simple column layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Main frame
        main = ctk.CTkFrame(self, fg_color=("#1e1e2e", "#16213e"))
        main.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)  # Skills frame expands
        main.rowconfigure(0, weight=0)  # Title bar fixed height
        main.rowconfigure(2, weight=0)  # Bottom bar fixed height
        # Row 0 (title bar) and row 2 (bottom bar) use their natural heights

        # Title bar
        title_bar = ctk.CTkFrame(main, height=50, fg_color=("#252535", "#1a1a25"))
        title_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        title_bar.columnconfigure(1, weight=1)

        ctk.CTkLabel(title_bar, text="🎯 Skills", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=15, pady=10)

        # Close button
        ctk.CTkButton(title_bar, text="✕", width=40, height=30, command=self._close,
                      fg_color=("#ff5555", "#cc4444")).grid(row=0, column=2, padx=10, pady=8)

        # Skills list
        self.skills_frame = ctk.CTkScrollableFrame(main, fg_color="transparent")
        self.skills_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Add skills
        for skill in self.skills_manager.get_all_skills():
            self._add_skill_row(self.skills_frame, skill)

        # Bottom bar
        bottom = ctk.CTkFrame(main, height=60, fg_color=("#252535", "#1a1a25"))
        bottom.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))

        ctk.CTkButton(bottom, text="➕ Add Skill", width=130, height=40,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      fg_color=("#50fa7b", "#40c969"),
                      hover_color=("#40c969", "#30b959"),
                      text_color=("#000000", "#000000"),
                      corner_radius=8,
                      command=self._add_skill).grid(row=0, column=0, padx=15, pady=10)

        count_label = ctk.CTkLabel(bottom, text=f"{len(self.skills_manager.skills)} skills",
                                   font=ctk.CTkFont(size=11))
        count_label.grid(row=0, column=1, padx=15, pady=10)

    def _add_skill_row(self, parent, skill):
        row = ctk.CTkFrame(parent, fg_color=("#2a2a3a", "#1a1a2a"), corner_radius=8)
        row.pack(fill="x", pady=3)

        # Left side: icon and name
        left_content = ctk.CTkFrame(row, fg_color="transparent")
        left_content.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(left_content, text=skill.icon, font=ctk.CTkFont(size=16)).pack(side="left", padx=10, pady=8)
        ctk.CTkLabel(left_content, text=skill.name, font=ctk.CTkFont(size=12)).pack(side="left", padx=5, pady=8)

        # Right side: buttons and switch
        right_content = ctk.CTkFrame(row, fg_color="transparent")
        right_content.pack(side="right", padx=5)

        # View/Edit button
        ctk.CTkButton(
            right_content,
            text="✏️",
            width=32,
            height=32,
            corner_radius=6,
            fg_color=("#4a9eff", "#3b7ac7"),
            hover_color=("#3b7ac7", "#2d5f9e"),
            font=ctk.CTkFont(size=12),
            command=lambda s=skill: self._view_edit_skill(s)
        ).pack(side="right", padx=(0, 8), pady=8)

        # Enable/Disable switch
        switch = ctk.CTkSwitch(
            right_content,
            text="",
            width=40,
            progress_color=("#50fa7b", "#40c969") if skill.enabled else ("#4a9eff", "#3b7ac7")
        )
        switch.pack(side="right", padx=5, pady=8)
        if skill.enabled:
            switch.select()

        def toggle():
            skill.enabled = switch.get()
            self.skills_manager.save_config()
            if self.on_skill_toggle:
                self.on_skill_toggle(skill.id, skill.enabled)

        switch.configure(command=toggle)

    def _add_skill(self):
        """Open the create skill dialog."""
        from skills_manager import CreateSkillDialog

        CreateSkillDialog(
            self,
            self.skills_manager,
            on_skill_created=self._refresh_skills
        )

    def _view_edit_skill(self, skill):
        """Open the view/edit skill dialog."""
        ViewSkillDialog(
            self,
            self.skills_manager,
            skill,
            on_skill_updated=self._refresh_skills
        )

    def _refresh_skills(self):
        """Refresh the skills list after changes."""
        # Clear and rebuild the skills list
        for widget in self.skills_frame.winfo_children():
            widget.destroy()

        for skill in self.skills_manager.get_all_skills():
            self._add_skill_row(self.skills_frame, skill)

        # Update count label
        for widget in self.winfo_children():
            self._update_count_label(widget)

    def _update_count_label(self, widget):
        """Recursively find and update the count label."""
        try:
            if hasattr(widget, 'cget') and 'skills' in str(widget.cget('text')):
                widget.configure(text=f"{len(self.skills_manager.skills)} skills")
                return
        except:
            pass

        for child in widget.winfo_children():
            self._update_count_label(child)


class ViewSkillDialog(ctk.CTkToplevel):
    """Dialog for viewing and editing an existing skill."""

    def __init__(self, parent, skills_manager, skill, on_skill_updated=None):
        super().__init__(parent)

        self.skills_manager = skills_manager
        self.skill = skill
        self.on_skill_updated = on_skill_updated
        self.selected_image_path = skill.image_path

        self.title(f"✏️ Edit Skill - {skill.name}")
        self.geometry("600x800")
        self.configure(fg_color="#1a1a2e")
        self.protocol("WM_DELETE_WINDOW", self._close_window)

        # Make window resizable
        self.minsize(550, 700)
        self.resizable(True, True)

        # Load existing skill data
        skill_instructions = self.skills_manager.get_skill_instructions(skill.id) or ""

        self._create_widgets(skill_instructions)
        self._center_window()

        try:
            self.transient(parent)
            self.grab_set()
        except Exception:
            pass

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.update_idletasks()
        parent = self.master
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _close_window(self):
        """Properly close the dialog and release grab."""
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _create_widgets(self, instructions_content):
        """Create the dialog widgets."""
        # Configure grid layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(30, 10))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text=f"✏️ Edit Skill: {self.skill.name}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#4a9eff",
        ).pack(pady=10)

        # Scrollable container for form
        scroll_container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color="#4a9eff",
            scrollbar_button_hover_color="#3b7ac7",
            height=450
        )
        scroll_container.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 5))

        form = ctk.CTkFrame(scroll_container, fg_color="#252535", corner_radius=15)
        form.pack(fill="x", pady=(0, 20))

        self._create_form_field(form, "Skill Name *", "e.g., Image Generator", 20, self.skill.name)
        self.name_entry = self.last_entry

        self._create_form_field(
            form, "Description *", "Brief description of what this skill does", 15, self.skill.description
        )
        self.desc_entry = self.last_entry

        self._create_form_field(form, "Category *", "", 15, self.skill.category)
        self.category_combo = self.last_entry
        self.category_combo.configure(
            values=[
                "General",
                "AI",
                "Documents",
                "Development",
                "Visualization",
                "Tools",
            ],
            font=ctk.CTkFont(size=12),
            fg_color="#1e1e2e",
            button_color="#4a9eff",
            hover_color="#3b7ac7",
            border_color="#3a3a4a",
            dropdown_fg_color="#252535",
            text_color="#ffffff",
            height=40,
        )

        self._create_form_field(form, "Icon Emoji", "🔧", 15, self.skill.icon)
        self.icon_entry = self.last_entry

        icon_frame = ctk.CTkFrame(form, fg_color="#1e1e2e")
        icon_frame.pack(fill="x", padx=20, pady=(5, 0))

        for icon in ["🔧", "🎯", "📝", "🧠", "💻", "🎨", "📊", "🔍", "⚡", "🚀"]:
            ctk.CTkButton(
                icon_frame,
                text=icon,
                width=35,
                height=35,
                font=ctk.CTkFont(size=14),
                fg_color="#3a3a4a",
                hover_color="#4a9eff",
                text_color="#ffffff",
                corner_radius=8,
                command=lambda i=icon: self._select_icon(i),
            ).pack(side="left", padx=2, pady=5)

        ctk.CTkLabel(
            form,
            text="Skill Icon (Image)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(15, 5))

        image_btn_frame = ctk.CTkFrame(form, fg_color="transparent")
        image_btn_frame.pack(fill="x", padx=20)

        ctk.CTkButton(
            image_btn_frame,
            text="📁 Change Image",
            font=ctk.CTkFont(size=11),
            fg_color="#4a9eff",
            hover_color="#3b7ac7",
            width=120,
            height=35,
            corner_radius=8,
            command=self._upload_image,
        ).pack(side="left", padx=(0, 10))

        self.image_status = ctk.CTkLabel(
            image_btn_frame,
            text="Current: " + (Path(self.skill.image_path).name if self.skill.image_path else "No image"),
            font=ctk.CTkFont(size=10),
            text_color="#666666",
        )
        self.image_status.pack(side="left", pady=5)

        ctk.CTkLabel(
            form,
            text="Instructions / Content *",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(15, 5))

        self.content_text = ctk.CTkTextbox(
            form,
            font=ctk.CTkFont(size=11),
            fg_color="#1e1e2e",
            border_color="#3a3a4a",
            text_color="#ffffff",
            height=150,
        )
        self.content_text.pack(fill="x", padx=20, pady=(0, 20))
        self.content_text.insert("1.0", instructions_content)

        # Buttons at bottom (always visible)
        button_frame = ctk.CTkFrame(self, fg_color="#252535", corner_radius=15)
        button_frame.grid(row=2, column=0, sticky="ew", padx=30, pady=(5, 25))

        # Center the buttons
        button_inner = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_inner.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkButton(
            button_inner,
            text="✕ Cancel",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3a3a4a",
            hover_color="#4a4a5a",
            text_color="#ffffff",
            width=140,
            height=45,
            corner_radius=10,
            command=self._close_window,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_inner,
            text="💾 Save Changes",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#50fa7b",
            hover_color="#40c969",
            text_color="#000000",
            width=150,
            height=45,
            corner_radius=10,
            command=self._update_skill,
        ).pack(side="left", padx=10)

    def _create_form_field(self, parent, label_text, placeholder, top_padding, default_value=""):
        """Helper to create form labels and entries."""
        ctk.CTkLabel(
            parent,
            text=label_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(top_padding, 5))

        self.last_entry = ctk.CTkEntry(
            parent,
            placeholder_text=placeholder,
            font=ctk.CTkFont(size=12),
            fg_color="#1e1e2e",
            border_color="#3a3a4a",
            text_color="#ffffff",
            placeholder_text_color="#666666",
            height=40,
        )
        self.last_entry.pack(fill="x", padx=20)
        if default_value:
            self.last_entry.insert(0, default_value)

    def _select_icon(self, icon: str):
        """Select an icon from the quick picker."""
        self.icon_entry.delete(0, "end")
        self.icon_entry.insert(0, icon)

    def _upload_image(self):
        """Open file dialog to upload an image."""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Skill Icon",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.selected_image_path = file_path
            filename = Path(file_path).name
            self.image_status.configure(
                text=f"✓ {filename[:30]}..." if len(filename) > 30 else f"✓ {filename}",
                text_color="#50fa7b",
            )

    def _update_skill(self):
        """Update the skill."""
        name = self.name_entry.get().strip()
        description = self.desc_entry.get().strip()
        category = self.category_combo.get()
        icon = self.icon_entry.get().strip() or "🔧"
        content = self.content_text.get("1.0", "end").strip()

        if not name:
            messagebox.showerror("Validation Error", "Please enter a skill name.")
            return

        if not description:
            messagebox.showerror("Validation Error", "Please enter a description.")
            return

        if not content:
            messagebox.showerror("Validation Error", "Please enter skill instructions.")
            return

        success = self.skills_manager.update_skill(
            skill_id=self.skill.id,
            name=name,
            description=description,
            category=category,
            icon=icon,
            content=content,
            image_path=self.selected_image_path,
        )

        if success:
            messagebox.showinfo("Success", f"Skill '{name}' updated successfully!")
            if self.on_skill_updated:
                self.on_skill_updated()
            self._close_window()
        else:
            messagebox.showerror("Error", "Failed to update skill.")


class ModelCatalogWindow(ctk.CTkToplevel):
    """Model catalog and download window."""

    _instance = None

    def __init__(self, parent, model_manager, on_model_select=None, on_model_download=None):
        # Close existing if any
        if ModelCatalogWindow._instance:
            try:
                ModelCatalogWindow._instance.destroy()
            except:
                pass

        super().__init__(parent)
        ModelCatalogWindow._instance = self

        self.title("🤖 Model Catalog")
        self.geometry("900x700")
        self.model_manager = model_manager
        self.on_model_select = on_model_select
        self.on_model_download = on_model_download

        self.protocol("WM_DELETE_WINDOW", lambda: self._close())
        self._download_progress_window = None

        self._build_ui()
        self._load_models()

    def _close(self):
        ModelCatalogWindow._instance = None
        self.destroy()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Main frame
        main = ctk.CTkFrame(self, fg_color=("#1e1e2e", "#16213e"))
        main.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # Title bar
        title_bar = ctk.CTkFrame(main, height=50, fg_color=("#252535", "#1a1a25"))
        title_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        title_bar.columnconfigure(1, weight=1)

        ctk.CTkLabel(title_bar, text="🤖 Model Catalog",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=15, pady=10)

        # Refresh button
        ctk.CTkButton(title_bar, text="🔄 Refresh", width=100,
                      command=self._load_models,
                      fg_color=("#4a9eff", "#3b7ac7")).grid(row=0, column=1, padx=5)

        # Close button
        ctk.CTkButton(title_bar, text="✕", width=40, height=30, command=self._close,
                      fg_color=("#ff5555", "#cc4444")).grid(row=0, column=2, padx=10, pady=8)

        # Content frame with tabs
        content = ctk.CTkFrame(main, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        # Tab buttons
        tab_frame = ctk.CTkFrame(content, height=40, fg_color="transparent")
        tab_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.tab_var = ctk.StringVar(value="catalog")

        ctk.CTkRadioButton(tab_frame, text="📚 Catalog", variable=self.tab_var,
                          value="catalog", command=self._switch_tab,
                          font=ctk.CTkFont(size=13)).pack(side="left", padx=10)

        ctk.CTkRadioButton(tab_frame, text="💾 Installed", variable=self.tab_var,
                          value="installed", command=self._switch_tab,
                          font=ctk.CTkFont(size=13)).pack(side="left", padx=10)

        # Model list container
        self.model_container = ctk.CTkScrollableFrame(content, fg_color="transparent")
        self.model_container.grid(row=1, column=0, sticky="nsew")

        # Store model items
        self.model_items = {}

    def _switch_tab(self):
        """Switch between catalog and installed views."""
        tab = self.tab_var.get()
        self._load_models(tab=tab)

    def _load_models(self, tab=None):
        """Load models based on current tab."""
        if tab is None:
            tab = self.tab_var.get()

        # Clear existing items
        for widget in self.model_container.winfo_children():
            widget.destroy()
        self.model_items.clear()

        if tab == "catalog":
            self._load_catalog()
        else:
            self._load_installed()

    def _load_catalog(self):
        """Load model catalog."""
        installed_ids = set(m.model_id for m in self.model_manager.get_installed_models())

        for model in self.model_manager.get_recommended_catalog():
            self._add_catalog_item(model, model.id in installed_ids)

    def _load_installed(self):
        """Load installed models."""
        installed = self.model_manager.get_installed_models()
        active_id = self.model_manager.get_active_model()

        if not installed:
            ctk.CTkLabel(
                self.model_container,
                text="No models installed.\nBrowse the catalog to download models.",
                font=ctk.CTkFont(size=12),
                text_color=("gray50", "gray50")
            ).pack(pady=40)
            return

        for installed_model in installed:
            model_info = self.model_manager.get_model_info(installed_model.model_id)
            self._add_installed_item(installed_model, model_info, active_id)

    def _add_catalog_item(self, model, is_installed):
        """Add a catalog model item."""
        item = ctk.CTkFrame(
            self.model_container,
            fg_color=("#2a2a3a", "#1a1a2a"),
            corner_radius=10
        )
        item.pack(fill="x", pady=8)

        # Left side - Model info
        left_frame = ctk.CTkFrame(item, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

        # Name and size
        name_label = ctk.CTkLabel(
            left_frame,
            text=f"{model.get_display_name()} • {model.get_size_display()}",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        name_label.pack(fill="x")

        # Description
        desc_label = ctk.CTkLabel(
            left_frame,
            text=model.description,
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray50"),
            anchor="w"
        )
        desc_label.pack(fill="x", pady=(3, 5))

        # Specs
        specs = f"RAM: {model.get_ram_display()} • Context: {model.context_size:,} tokens"
        specs_label = ctk.CTkLabel(
            left_frame,
            text=specs,
            font=ctk.CTkFont(size=10),
            text_color=("gray60", "gray40"),
            anchor="w"
        )
        specs_label.pack(fill="x")

        # Tags
        if model.tags:
            tags_text = " • ".join(model.tags[:3])
            tags_label = ctk.CTkLabel(
                left_frame,
                text=tags_text,
                font=ctk.CTkFont(size=9),
                text_color=("#4a9eff", "#3b7ac7"),
                anchor="w"
            )
            tags_label.pack(fill="x", pady=(3, 0))

        # Right side - Actions
        right_frame = ctk.CTkFrame(item, fg_color="transparent")
        right_frame.pack(side="right", padx=15, pady=10)

        # Check if this is the active model
        active_id = self.model_manager.get_active_model() if self.model_manager else None
        is_active = (model.id == active_id)

        if is_installed:
            if is_active:
                # Active model - show star icon
                status_label = ctk.CTkLabel(
                    right_frame,
                    text="⭐ Active",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=("#f1c40f", "#d4ac0d")
                )
                status_label.pack(pady=5)
            else:
                # Installed but not active - show activate and delete buttons
                activate_btn = ctk.CTkButton(
                    right_frame,
                    text="⭐ Activate",
                    width=110,
                    command=lambda: self._activate_model(model.id),
                    fg_color=("#4a9eff", "#3b7ac7"),
                    hover_color=("#3b7ac7", "#2d5f9e")
                )
                activate_btn.pack(pady=3)

                delete_btn = ctk.CTkButton(
                    right_frame,
                    text="🗑️ Delete",
                    width=110,
                    command=lambda: self._delete_model(model.id),
                    fg_color=("#ff5555", "#cc4444"),
                    hover_color=("#cc4444", "#aa3333")
                )
                delete_btn.pack(pady=3)
        else:
            download_btn = ctk.CTkButton(
                right_frame,
                text="⬇️ Download",
                width=110,
                command=lambda: self._download_model(model),
                fg_color=("#50fa7b", "#40c969"),
                hover_color=("#40c969", "#30b959")
            )
            download_btn.pack(pady=5)

    def _add_installed_item(self, installed_model, model_info, active_model):
        """Add an installed model item."""
        item = ctk.CTkFrame(
            self.model_container,
            fg_color=("#2a2a3a", "#1a1a2a"),
            corner_radius=10
        )
        item.pack(fill="x", pady=8)

        # Active indicator
        if installed_model.is_active:
            item.configure(fg_color=("#2d5f9e", "#1e3f5e"))

        # Left side - Model info
        left_frame = ctk.CTkFrame(item, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

        # Name
        if model_info:
            name = model_info.get_display_name()
        else:
            name = installed_model.model_id

        if installed_model.is_active:
            name += " ⭐ Active"

        name_label = ctk.CTkLabel(
            left_frame,
            text=name,
            font=ctk.CTkFont(size=14, weight="bold" if installed_model.is_active else "normal"),
            anchor="w"
        )
        name_label.pack(fill="x")

        # File info
        size_mb = installed_model.file_size_bytes / (1024 * 1024)
        size_text = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{installed_model.file_size_bytes} bytes"

        info_label = ctk.CTkLabel(
            left_frame,
            text=f"📁 {installed_model.filepath} • {size_text}",
            font=ctk.CTkFont(size=10),
            text_color=("gray60", "gray40"),
            anchor="w"
        )
        info_label.pack(fill="x")

        # Right side - Actions
        right_frame = ctk.CTkFrame(item, fg_color="transparent")
        right_frame.pack(side="right", padx=15, pady=10)

        if not installed_model.is_active:
            activate_btn = ctk.CTkButton(
                right_frame,
                text="⭐ Activate",
                width=100,
                command=lambda: self._activate_model(installed_model.model_id),
                fg_color=("#50fa7b", "#40c969")
            )
            activate_btn.pack(pady=3)

            delete_btn = ctk.CTkButton(
                right_frame,
                text="🗑️",
                width=40,
                command=lambda: self._delete_model(installed_model.model_id),
                fg_color=("#ff5555", "#cc4444")
            )
            delete_btn.pack(pady=3)

    def _download_model(self, model):
        """Download a model."""
        # Open progress window
        self._download_progress_window = DownloadProgressDialog(
            self,
            model,
            on_close=lambda: self._load_models()
        )

        # Start download
        self.model_manager.download_model(
            model.id,
            on_progress=lambda p: self._download_progress_window.update_progress(p),
            on_complete=lambda success, msg: self._download_progress_window.complete(success, msg)
        )

    def _activate_model(self, model_id):
        """Activate a model."""
        if self.on_model_select:
            self.on_model_select(model_id)

    def _delete_model(self, model_id):
        """Delete a model after confirmation."""
        # Check if this is the active model
        active_model = self.model_manager.get_active_model()
        if active_model and active_model.model_id == model_id:
            messagebox.showwarning(
                "Cannot Delete Active Model",
                "You cannot delete the currently active model.\n\n"
                "Please activate another model first, then delete this one."
            )
            return

        if messagebox.askyesno(
            "Delete Model",
            "Are you sure you want to delete this model? This cannot be undone."
        ):
            if self.model_manager.delete_model(model_id):
                messagebox.showinfo("Success", "Model deleted successfully.")
                self._load_models()
            else:
                messagebox.showerror("Error", "Failed to delete model.")


class DownloadProgressDialog(ctk.CTkToplevel):
    """Progress dialog for model downloads."""

    def __init__(self, parent, model, on_close=None):
        super().__init__(parent)

        self.title("⬇️ Downloading Model")
        self.geometry("500x300")
        self.model = model
        self.on_close = on_close

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ctk.CTkFrame(self, fg_color=("#1e1e2e", "#16213e"))
        main.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Model info
        info_frame = ctk.CTkFrame(main, fg_color=("#252535", "#1a1a25"))
        info_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            info_frame,
            text=f"Downloading {self.model.get_display_name()}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        ctk.CTkLabel(
            info_frame,
            text=f"Size: {self.model.get_size_display()}",
            font=ctk.CTkFont(size=12),
            text_color=("gray70", "gray50")
        ).pack()

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main,
            width=400,
            height=20,
            progress_color=("#50fa7b", "#40c969")
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20)

        # Status label
        self.status_label = ctk.CTkLabel(
            main,
            text="Initializing...",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack()

        # Details label
        self.details_label = ctk.CTkLabel(
            main,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray50")
        )
        self.details_label.pack(pady=5)

        # Close button (disabled initially)
        self.close_btn = ctk.CTkButton(
            main,
            text="Close",
            width=100,
            command=self._close,
            state="disabled"
        )
        self.close_btn.pack(pady=20)

    def update_progress(self, progress):
        """Update download progress."""
        self.progress_bar.set(progress.percentage / 100)

        downloaded_mb = progress.downloaded_bytes / (1024 * 1024)
        total_mb = progress.total_bytes / (1024 * 1024)

        self.status_label.configure(
            text=f"Downloading... {progress.percentage:.1f}%"
        )
        self.details_label.configure(
            text=f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB"
        )

    def complete(self, success: bool, message: str):
        """Handle download completion."""
        if success:
            self.progress_bar.set(1)
            self.status_label.configure(
                text="✅ Download Complete!",
                text_color=("#50fa7b", "#40c969")
            )
        else:
            self.progress_bar.set(0)
            self.status_label.configure(
                text="❌ Download Failed",
                text_color=("#ff5555", "#cc4444")
            )
            self.details_label.configure(text=message)

        self.close_btn.configure(state="normal")

    def _close(self):
        """Close the dialog."""
        if self.on_close:
            self.on_close()
        self.destroy()

