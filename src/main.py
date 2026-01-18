#!/usr/bin/env python3
"""
Local Chat - Entry point.
"""

import tkinter as tk
import threading
from engine import LLMEngine
from ui import ChatUI


class App:
    """Main application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.engine = LLMEngine()
        self.generating = False
        
        self.ui = ChatUI(
            self.root,
            on_send=self.send,
            on_clear=self.clear,
            on_load_files=self.load_files
        )
        
        self.ui.add_message("System", "Starting...", "system")
        self._load_model()
    
    def _load_model(self):
        """Loads model in background."""
        def task():
            success = self.engine.load(
                on_progress=lambda m: self.root.after(0, lambda: self.ui.add_message("Debug", m, "system"))
            )
            if success:
                self.root.after(0, lambda: self.ui.set_status("✅ Ready"))
                self.root.after(0, lambda: self.ui.add_message("System", "Model loaded! Start chatting.", "system"))
            else:
                self.root.after(0, lambda: self.ui.set_status("❌ Error", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def send(self, message: str):
        """Sends a message."""
        if self.generating or not self.engine.is_ready:
            return
        
        self.ui.add_message("You", message, "user")
        self.generating = True
        self.ui.set_enabled(False)
        
        threading.Thread(target=self._generate, args=(message,), daemon=True).start()
    
    def _generate(self, message: str):
        """Generates response."""
        self.root.after(0, lambda: self.ui.add_message("Assistant", "", "bot"))
        
        try:
            for token in self.engine.generate(message):
                self.root.after(0, lambda t=token: self.ui.stream(t))
        except Exception as e:
            self.root.after(0, lambda: self.ui.add_message("Error", str(e), "error"))
        finally:
            self.root.after(0, self._done)
    
    def _done(self):
        """Generation complete."""
        self.generating = False
        self.ui.set_enabled(True)
        self.ui.focus_input()
    
    def clear(self):
        """Clears chat and history."""
        self.engine.clear_history()
        self.ui.clear_chat()
        self.ui.add_message("System", "Conversation cleared.", "system")
    
    def load_files(self, files: tuple):
        """Loads new documents into RAG."""
        self.ui.add_message("System", f"Loading {len(files)} file(s)...", "system")
        
        def task():
            success = self.engine.rag.add_documents(
                files,
                on_progress=lambda m: self.root.after(0, lambda: self.ui.add_message("Debug", m, "system"))
            )
            if success:
                self.root.after(0, lambda: self.ui.add_message("System", "✅ Files loaded successfully!", "system"))
            else:
                self.root.after(0, lambda: self.ui.add_message("Error", "Failed to load files", "error"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def run(self):
        """Runs the app."""
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
