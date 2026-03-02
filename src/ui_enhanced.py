"""
Enhanced ChatUI with modern features and improved UX.

Features:
- Message bubbles with different colors
- Typing indicator animation
- Markdown support in messages
- Copy code buttons
- Timestamps on messages
- Model info panel
- Smooth animations
"""

import os
import re
import time
from datetime import datetime
from typing import Callable, Optional
import customtkinter as ctk
from tkinter import filedialog, messagebox
import config


class ModernMessageBubble(ctk.CTkFrame):
    """Modern message bubble with timestamp and copy functionality."""

    def __init__(self, parent, sender: str, text: str, timestamp: Optional[str] = None,
                 is_user: bool = False, on_copy: Optional[Callable] = None):
        super().__init__(parent, fg_color="transparent")

        self.is_user = is_user
        self.on_copy = on_copy

        # Colors
        if is_user:
            bg_color = "#4a9eff"
            text_color = "#ffffff"
        else:
            bg_color = "#2d2d3a"
            text_color = "#e0e0e0"

        # Container
        bubble = ctk.CTkFrame(
            self,
            fg_color=bg_color,
            corner_radius=15,
            border_width=0
        )
        bubble.pack(fill="both", expand=True, padx=10, pady=5)

        # Header with sender and timestamp
        header = ctk.CTkFrame(bubble, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 5))

        sender_text = "👤 You" if is_user else "🤖 Assistant"
        sender_label = ctk.CTkLabel(
            header,
            text=sender_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=text_color
        )
        sender_label.pack(side="left")

        if timestamp:
            time_label = ctk.CTkLabel(
                header,
                text=timestamp,
                font=ctk.CTkFont(size=9),
                text_color="#888888" if is_user else "#666666"
            )
            time_label.pack(side="right")

        # Message content
        content = ctk.CTkTextbox(
            bubble,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="transparent",
            border_width=0,
            wrap="word",
            height=1,
        )
        content.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        content.insert("0.0", text)
        content.configure(state="disabled")

        # Copy button for code blocks
        if "```" in text and not is_user:
            copy_btn = ctk.CTkButton(
                bubble,
                text="📋 Copy",
                width=70,
                height=25,
                corner_radius=8,
                font=ctk.CTkFont(size=10),
                fg_color="#3d3d4d",
                hover_color="#4d4d5d",
                command=lambda: self._copy_text(text)
            )
            copy_btn.pack(pady=(0, 10))

    def _copy_text(self, text: str):
        """Copy text to clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)
        if self.on_copy:
            self.on_copy()


class TypingIndicator(ctk.CTkFrame):
    """Animated typing indicator."""

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")

        self.bubble = ctk.CTkFrame(
            self,
            fg_color="#2d2d3a",
            corner_radius=15
        )
        self.bubble.pack(padx=10, pady=5)

        self.dots = []
        for i in range(3):
            dot = ctk.CTkLabel(
                self.bubble,
                text="●",
                font=ctk.CTkFont(size=12),
                text_color="#888888"
            )
            dot.pack(side="left", padx=2)
            self.dots.append(dot)

        self._animate()

    def _animate(self):
        """Blink animation for dots."""
        for i, dot in enumerate(self.dots):
            color = "#ffffff" if i == 0 else "#888888"
            dot.configure(text_color=color)
        self.after(400, self._animate_step1)

    def _animate_step1(self):
        for i, dot in enumerate(self.dots):
            color = "#ffffff" if i == 1 else "#888888"
            dot.configure(text_color=color)
        self.after(400, self._animate_step2)

    def _animate_step2(self):
        for i, dot in enumerate(self.dots):
            color = "#ffffff" if i == 2 else "#888888"
            dot.configure(text_color=color)
        self.after(400, self._animate)


class ModelInfoPanel(ctk.CTkFrame):
    """Panel showing model information and stats."""

    def __init__(self, parent):
        super().__init__(parent, fg_color="#1e1e2e", corner_radius=10)

        self._create_widgets()

    def _create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="🤖 Model Info",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4a9eff"
        )
        title.pack(pady=(10, 5))

        # Model stats
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.model_label = ctk.CTkLabel(
            stats_frame,
            text="Model: Loading...",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            anchor="w"
        )
        self.model_label.pack(fill="x", pady=2)

        self.context_label = ctk.CTkLabel(
            stats_frame,
            text=f"Context: {config.CONTEXT_SIZE} tokens",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            anchor="w"
        )
        self.context_label.pack(fill="x", pady=2)

        self.gpu_label = ctk.CTkLabel(
            stats_frame,
            text="GPU: Metal (Apple Silicon)",
            font=ctk.CTkFont(size=10),
            text_color="#50fa7b",
            anchor="w"
        )
        self.gpu_label.pack(fill="x", pady=2)

        # Separator
        ctk.CTkFrame(self, height=1, fg_color="#333333").pack(fill="x", padx=10, pady=5)

        # Settings hint
        hint = ctk.CTkLabel(
            self,
            text="💡 Tip: Load documents for RAG",
            font=ctk.CTkFont(size=9),
            text_color="#666666"
        )
        hint.pack(pady=(0, 10))

    def update_model(self, model_name: str):
        """Update model name display."""
        self.model_label.configure(text=f"Model: {model_name}")


class EnhancedChatUI:
    """Enhanced ChatUI with modern features."""

    def __init__(self, root: ctk.CTk, on_send, on_clear, on_load_files,
                 on_new_chat, on_select_chat, on_delete_chat):
        self.root = root
        self.on_send = on_send
        self.on_clear = on_clear
        self.on_load_files = on_load_files
        self.on_new_chat = on_new_chat
        self.on_select_chat = on_select_chat
        self.on_delete_chat = on_delete_chat

        self.typing_indicator = None
        self.message_history = []

        self._setup_window()
        self._create_widgets()

    def _setup_window(self):
        self.root.title(f"🤖 {config.APP_NAME}")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def _create_widgets(self):
        # === SIDEBAR ===
        sidebar_frame = ctk.CTkFrame(
            self.root,
            width=280,
            fg_color="#1a1a2e",
            corner_radius=0
        )
        sidebar_frame.grid(row=0, column=0, sticky="nsew")
        sidebar_frame.grid_propagate(False)

        # Logo/Header
        header = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(20, 10))

        logo = ctk.CTkLabel(
            header,
            text="⚡ Local RAG",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#4a9eff"
        )
        logo.pack()

        subtitle = ctk.CTkLabel(
            header,
            text="Powered by Qwen2.5",
            font=ctk.CTkFont(size=11),
            text_color="#666666"
        )
        subtitle.pack()

        # New Chat Button
        new_chat_btn = ctk.CTkButton(
            sidebar_frame,
            text="+ New Chat",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#4a9eff",
            hover_color="#3b7ac7",
            text_color="#ffffff",
            height=40,
            corner_radius=10,
            command=self.on_new_chat
        )
        new_chat_btn.pack(fill="x", padx=15, pady=10)

        # Conversations list
        conv_label = ctk.CTkLabel(
            sidebar_frame,
            text="💬 Conversations",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888888",
            anchor="w"
        )
        conv_label.pack(fill="x", padx=15, pady=(15, 5))

        self.conv_list = ctk.CTkScrollableFrame(
            sidebar_frame,
            fg_color="transparent",
            scrollbar_button_color="#4a9eff",
            scrollbar_button_hover_color="#3b7ac7"
        )
        self.conv_list.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Model Info Panel
        self.model_panel = ModelInfoPanel(sidebar_frame)
        self.model_panel.pack(fill="x", padx=15, pady=(0, 15))

        # === MAIN CONTENT ===
        main_frame = ctk.CTkFrame(self.root, fg_color="#0d0d15")
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Top bar
        top_bar = ctk.CTkFrame(main_frame, fg_color="#0d0d15", height=60)
        top_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        top_bar.grid_propagate(False)

        # Title and status
        title_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        title_frame.pack(side="left", fill="x", expand=True)

        title = ctk.CTkLabel(
            title_frame,
            text="💬 Chat",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffffff"
        )
        title.pack(side="left")

        self.status = ctk.CTkLabel(
            title_frame,
            text="● Ready",
            font=ctk.CTkFont(size=11),
            text_color="#50fa7b"
        )
        self.status.pack(side="left", padx=(15, 0))

        # Action buttons
        actions_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        actions_frame.pack(side="right")

        ctk.CTkButton(
            actions_frame,
            text="📁 Load",
            width=80,
            height=35,
            corner_radius=8,
            font=ctk.CTkFont(size=11),
            fg_color="#2d2d3a",
            hover_color="#3d3d4d",
            text_color="#e0e0e0",
            command=self._load_files
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            actions_frame,
            text="🗑️ Clear",
            width=80,
            height=35,
            corner_radius=8,
            font=ctk.CTkFont(size=11),
            fg_color="#2d2d3a",
            hover_color="#3d3d4d",
            text_color="#e0e0e0",
            command=self.on_clear
        ).pack(side="left", padx=5)

        # Chat area
        chat_container = ctk.CTkFrame(
            main_frame,
            fg_color="#0d0d15",
            corner_radius=15
        )
        chat_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)

        self.chat_scroll = ctk.CTkScrollableFrame(
            chat_container,
            fg_color="transparent",
            scrollbar_button_color="#4a9eff",
            scrollbar_button_hover_color="#3b7ac7",
            label_text=""
        )
        self.chat_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        self.chat_container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        self.chat_container.grid(row=0, column=0, sticky="ew")

        # Input area
        input_frame = ctk.CTkFrame(main_frame, fg_color="#0d0d15")
        input_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(5, 15))
        input_frame.grid_columnconfigure(0, weight=1)

        # Text input with border
        input_container = ctk.CTkFrame(
            input_frame,
            fg_color="#1e1e2e",
            corner_radius=15,
            border_width=2,
            border_color="#2d2d3a"
        )
        input_container.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.input = ctk.CTkTextbox(
            input_container,
            height=50,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="transparent",
            border_width=0,
            wrap="word"
        )
        self.input.pack(fill="both", expand=True, padx=15, pady=15)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)

        # Send button
        send_btn = ctk.CTkButton(
            input_frame,
            text="Send ➤",
            width=100,
            height=50,
            corner_radius=15,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#4a9eff",
            hover_color="#3b7ac7",
            text_color="#ffffff",
            command=self._send
        )
        send_btn.grid(row=0, column=1)

        # Footer with document info
        self.doc_info = ctk.CTkLabel(
            main_frame,
            text="📚 No documents loaded",
            font=ctk.CTkFont(size=10),
            text_color="#666666"
        )
        self.doc_info.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 10))

    def _on_enter(self, event):
        """Handle Enter key - send message, Shift+Enter for new line."""
        if not (event.state & 0x1):  # Shift key
            self._send()
            return "break"

    def _send(self):
        """Send the current message."""
        text = self.input.get("0.0", "end").strip()
        if text:
            self.input.delete("0.0", "end")
            self.on_send(text)

    def _load_files(self):
        """Open file dialog to load documents."""
        files = filedialog.askopenfilenames(
            parent=self.root,
            title="Select documents",
            filetypes=[
                ("All supported", "*.txt *.md *.pdf *.xlsx *.xls *.pptx *.ppt *.docx *.doc"),
                ("Documents", "*.txt *.md *.pdf *.docx *.doc"),
                ("Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Presentations", "*.pptx *.ppt"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.on_load_files(files)

    def set_status(self, text: str, is_error: bool = False):
        """Update status indicator."""
        color = "#ff5555" if is_error else "#50fa7b"
        icon = "●" if not is_error else "⚠"
        self.status.configure(text=f"{icon} {text}", text_color=color)

    def update_doc_count(self, count: int):
        """Update document count display."""
        if count == 0:
            self.doc_info.configure(text="📚 No documents loaded")
        else:
            self.doc_info.configure(text=f"📚 {count} document{'s' if count > 1 else ''} loaded • RAG active")

    def update_model_info(self, model_name: str = "Qwen2.5:0.5b"):
        """Update model information panel."""
        self.model_panel.update_model(model_name)

    def add_message(self, sender: str, text: str, tag: str = ""):
        """Add a message to the chat."""
        is_user = sender.lower() == "you"
        timestamp = datetime.now().strftime("%H:%M")

        # Create message bubble
        bubble = ModernMessageBubble(
            self.chat_container,
            sender=sender,
            text=text,
            timestamp=timestamp,
            is_user=is_user,
            on_copy=self._copy_toast
        )
        bubble.pack(fill="x", pady=2)

        # Scroll to bottom
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _copy_toast(self):
        """Show toast notification for copy."""
        # Simple feedback - could be enhanced with a toast notification
        pass

    def stream(self, text: str):
        """Stream text to chat (for LLM responses)."""
        # For streaming, we'd need to update the last message bubble
        # This is a simplified version
        pass

    def show_typing(self):
        """Show typing indicator."""
        if not self.typing_indicator:
            self.typing_indicator = TypingIndicator(self.chat_container)
            self.typing_indicator.pack(fill="x", pady=5)
            self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def hide_typing(self):
        """Hide typing indicator."""
        if self.typing_indicator:
            self.typing_indicator.destroy()
            self.typing_indicator = None

    def clear_chat(self):
        """Clear all messages from chat."""
        for widget in self.chat_container.winfo_children():
            widget.destroy()

    def set_enabled(self, enabled: bool):
        """Enable/disable input."""
        state = "normal" if enabled else "disabled"
        self.input.configure(state=state)

    def focus_input(self):
        """Focus on input field."""
        self.input.focus_set()

    def update_sidebar(self, conversations: list, current_id: str):
        """Update conversations list in sidebar."""
        for widget in self.conv_list.winfo_children():
            widget.destroy()

        if not conversations:
            empty = ctk.CTkLabel(
                self.conv_list,
                text="No conversations yet.\nClick '+ New Chat' to start!",
                font=ctk.CTkFont(size=11),
                text_color="#666666"
            )
            empty.pack(pady=20)
            return

        for conv in conversations:
            # Create conversation item
            item = ctk.CTkFrame(
                self.conv_list,
                fg_color="#2d2d3a" if conv.id != current_id else "#4a9eff",
                corner_radius=10,
                height=50
            )
            item.pack(fill="x", pady=3)
            item.pack_propagate(False)

            # Content
            content = ctk.CTkFrame(item, fg_color="transparent")
            content.pack(fill="both", expand=True, padx=10, pady=8)

            title = ctk.CTkLabel(
                content,
                text=conv.title[:22] + "..." if len(conv.title) > 22 else conv.title,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#ffffff" if conv.id == current_id else "#e0e0e0",
                anchor="w"
            )
            title.pack(fill="x")

            # Delete button
            delete_btn = ctk.CTkButton(
                item,
                text="✕",
                width=25,
                height=25,
                corner_radius=5,
                fg_color="transparent",
                hover_color="#ff5555",
                font=ctk.CTkFont(size=10),
                command=lambda c=conv.id: self._confirm_delete(c)
            )
            delete_btn.pack(side="right", padx=5)

            # Bind click
            item.bind("<Button-1>", lambda e, c=conv.id: self.on_select_chat(c))
            content.bind("<Button-1>", lambda e, c=conv.id: self.on_select_chat(c))

    def _confirm_delete(self, conv_id: str):
        """Confirm and delete conversation."""
        if messagebox.askyesno("Delete Chat", "Delete this conversation?"):
            self.on_delete_chat(conv_id)

    def load_messages(self, messages: list):
        """Load messages from conversation."""
        self.clear_chat()
        for msg in messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            self.add_message(role, msg["content"], msg["role"])

    def prompt_file_save(self, filename: str, default_name: str) -> str:
        """Prompt user to choose save location."""
        filepath = filedialog.asksaveasfilename(
            title=f"Save {filename}",
            defaultextension=os.path.splitext(default_name)[1],
            initialfile=default_name,
            filetypes=[
                ("All files", "*.*"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"),
                ("Excel files", "*.xlsx"),
            ]
        )
        return filepath

    def show_code_status(self, status: str, icon: str = "⚡"):
        """Show code execution status."""
        self.add_message("System", f"{icon} {status}")
