#!/usr/bin/env python3
"""
Local Chat - Entry Point with CustomTkinter.
"""

import sys
import os

# macOS fixes
if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    if hasattr(sys, '_MEIPASS'):
        import multiprocessing
        multiprocessing.freeze_support()

import customtkinter as ctk
import threading
from llm import LLMEngine
from ui import ChatUI


class App:
    """Main application."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.engine = LLMEngine()
        self.generating = False
        
        self.ui = ChatUI(
            self.root,
            on_send=self.send,
            on_clear=self.clear,
            on_load_files=self.load_files
        )
        
        self.ui.add_message("System", "🚀 Starting up...", "system")
        self._load_model()
    
    def _load_model(self):
        """Load model in background."""
        def task():
            success = self.engine.load(
                on_progress=lambda m: self.root.after(0, lambda: self.ui.add_message("Debug", m, "system"))
            )
            if success:
                self.root.after(0, lambda: self.ui.set_status("✅ Ready to chat!"))
                self.root.after(0, lambda: self.ui.add_message(
                    "System", 
                    "✨ Ready! Ask me anything or load documents for context.", 
                    "system"
                ))
                doc_count = len(self.engine.rag.documents) if self.engine.rag else 0
                self.root.after(0, lambda: self.ui.update_doc_count(doc_count))
            else:
                self.root.after(0, lambda: self.ui.set_status("❌ Failed to load", is_error=True))
                error_msg = self.engine.error or "Unknown error"
                self.root.after(0, lambda: self.ui.add_message(
                    "Error", 
                    f"Failed to initialize: {error_msg}", 
                    "error"
                ))
        
        threading.Thread(target=task, daemon=True).start()
    
    def send(self, message: str):
        """Send a message."""
        if self.generating or not self.engine.is_ready:
            return
        
        self.ui.add_message("You", message, "user")
        self.generating = True
        self.ui.set_enabled(False)
        self.ui.set_status("🤔 Thinking...")
        
        threading.Thread(target=self._generate, args=(message,), daemon=True).start()
    
    def _generate(self, message: str):
        """Generate response."""
        self.root.after(0, lambda: self.ui.add_message("Assistant", "", "bot"))
        
        try:
            for token in self.engine.generate(message):
                self.root.after(0, lambda t=token: self.ui.stream(t))
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            self.root.after(0, lambda: self.ui.add_message("Error", error_msg, "error"))
            print(f"[App] {error_msg}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, self._done)
    
    def _done(self):
        """Generation complete."""
        self.generating = False
        self.ui.set_enabled(True)
        self.ui.set_status("✅ Ready to chat!")
        self.ui.focus_input()
    
    def clear(self):
        """Clear chat and history."""
        self.engine.clear_history()
        self.ui.clear_chat()
        self.ui.add_message("System", "💬 Conversation cleared. Fresh start!", "system")
    
    def load_files(self, files: tuple):
        """Load documents into RAG."""
        self.ui.add_message("System", f"📥 Loading {len(files)} file(s)...", "system")
        self.ui.set_status("⏳ Loading files...")
        
        def task():
            success = self.engine.rag.add_documents(
                files,
                on_progress=lambda m: self.root.after(0, lambda msg=m: self.ui.add_message("Debug", msg, "system"))
            )
            
            if success:
                doc_count = len(self.engine.rag.documents)
                self.root.after(0, lambda: self.ui.update_doc_count(doc_count))
                self.root.after(0, lambda: self.ui.add_message(
                    "System", 
                    f"✅ Successfully loaded! Now you can ask questions about your documents.", 
                    "system"
                ))
                self.root.after(0, lambda: self.ui.set_status("✅ Ready to chat!"))
            else:
                self.root.after(0, lambda: self.ui.add_message(
                    "Error", 
                    "❌ Failed to load files. Check console for details.", 
                    "error"
                ))
                self.root.after(0, lambda: self.ui.set_status("❌ Load failed", is_error=True))
        
        threading.Thread(target=task, daemon=True).start()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
